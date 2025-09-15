
import { models, defaultModel } from './models';

// -----------------------------------------------------------------------
// #region               Local definitions and variables
// -----------------------------------------------------------------------

const MIN_LAST_SENTENCE_LETTERS = 20;
const MIN_LAST_PARAGRAPH_LETTERS = 80;


const EXP_TABLE = [
    [512, 1392, 3783, 10284, 27954, 75988, 206556, 561476],
    [512, 580, 657, 745, 844, 957, 1084, 1228],
    [512, 520, 528, 537, 545, 554, 562, 571],
    [512, 513, 514, 515, 516, 517, 518, 519],
];


export interface ILoremIpsumOptions {
    language?: string | undefined;
    heat?: number | undefined;
    seed?: number | undefined;
    version?: number | undefined;
    paragraphs?: boolean | undefined | {
        separator?: string | undefined;
        mean?: number | undefined;
        variance?: number | undefined;
        shorterVariance?: number | undefined;
        longerVariance?: number | undefined;
    };
}

interface ILinearLayer {
    type: 'linear';
    weight: number[][];
    bias: number[];
    input_shift: number[];
    input_clamp: number[][];
}

interface IReLULayer {
    type: "relu";
    input_size: number;
}

interface IScaledSoftmaxLayer {
    type: "scaled_softmax";
    weight: number[];
    frac_bits: number;
}


// #endregion ------------------------------------------------------------
// #region                     LoremIpsum class
// -----------------------------------------------------------------------


export class LoremIpsum {

    private randState: number = 0;
    private groups!: number[][];
    private lastWordHash!: number;
    private currentWordHash!: number;
    private groupContext: number[] = new Array(4 * 3).fill(0);
    private invHeat: number = 26;
    private generateUpper!: boolean;
    private generateSpace!: boolean;
    private wordsSinceDot!: number;
    private wordsSinceComma!: number;
    private sentencesInParagraph!: number;
    private enableParagraphs: boolean = false;
    private paragraphProbTable: number[] = new Array(64).fill(0);
    private model: typeof models[keyof typeof models] = models.defaultModel;
    private paragraphSeparator: string = '\n';
    private letterToIndex: { [key: string]: number } = {};

    // #endregion ------------------------------------------------------------
    // #region                     Public interface
    // -----------------------------------------------------------------------

    public constructor(options?: ILoremIpsumOptions) {

        if (!options?.language) {
            this.model = defaultModel;
        } else if (models[options.language]) {
            this.model = models[options.language];
        } else {
            throw new Error(`Language stylization "${options.language}" not found.`);
        }

        for (let i = 0; i < this.model.letters.length; i++) {
            this.letterToIndex[this.model.letters[i]] = i;
            this.letterToIndex[this.model.letters[i].toUpperCase()] = i;
        }

        if (options?.version != null) {
            if (options.version !== 1 && options.version !== 0) {
                throw new Error(`Unsupported version: ${options.version}.`);
            }
        }

        if (options?.seed != null) {
            this.randState = options.seed >>> 0;
        } else {
            let seed = Math.floor(Math.random() * 0x7FFFFFFF);
            seed ^= Date.now() & 0x7FFFFFFF;
            try {
                let p = globalThis.performance.now();
                seed ^= p ^ Math.floor((p - Math.floor(p)) * 0x7FFFFFFF);
                p = globalThis.performance.timeOrigin;
                seed ^= p ^ Math.floor((p - Math.floor(p)) * 0x7FFFFFFF);
            } catch (err) { }
            this.randState = seed >>> 0;
        }

        if (options?.heat != null) {
            let heatPercent = Math.round(options.heat * 100);
            if (heatPercent > 1) {
                this.invHeat = Math.trunc(1600 / heatPercent);
                if (this.invHeat > 127) this.invHeat = 127;
                if (this.invHeat < 1) this.invHeat = 1;
            } else {
                this.invHeat = 127;
            }
        }

        if (options?.paragraphs) {
            this.enableParagraphs = true;
            let opt = options.paragraphs;
            if (opt === true) opt = {};
            this.paragraphSeparator = opt.separator ?? '\n';
            let meanFloat = opt.mean ?? 5;
            let shorterVarianceFloat = opt.shorterVariance ?? opt.variance ?? 2;
            let longerVarianceFloat = opt.longerVariance ?? opt.variance ?? 4;
            let mean10 = Math.round(meanFloat * 10);
            let shorterVariance10 = Math.round(shorterVarianceFloat * 10);
            let longerVariance10 = Math.round(longerVarianceFloat * 10);
            this.paragraphProbTable = new Array(64).fill(255);
            let sum = 0;
            for (let i = 64 - 2; i >= 0; i--) {
                let x = 10 * (i + 1);
                let probInd = this.normalDist(mean10, (x <= mean10) ? shorterVariance10 : longerVariance10, x);
                sum += probInd;
                if (sum > 0) {
                    this.paragraphProbTable[i] = Math.trunc((probInd * 255) / sum);
                } else {
                    this.paragraphProbTable[i] = 255;
                }
            }
        }

        this.resetContext();
    }


    public static languages(): { [key: string]: string } {
        return Object.fromEntries(Object.entries(models).map(([key, value]) => [key, value.name]));
    }


    public generate(length: number): string {
        let result = '';
        let left = length;
        while (left > 1) {
            let nextLetter = this.next(left - 1);
            if (left < nextLetter.length + 1) {
                break;
            }
            result += nextLetter;
            left -= nextLetter.length;
        }
        if (left > 0) {
            result += '.';
        }
        // Update state, so the next call will start a new sentence.
        this.updateContext(0);
        this.generateSpace = true;
        this.generateUpper = true;
        this.wordsSinceDot = 0;
        this.wordsSinceComma = 0;
        return result;
    }


    public next(remainingCharacters: number): string {

        if (remainingCharacters < 1) {
            this.generateSpace = true;
            this.generateUpper = true;
            this.wordsSinceDot = 0;
            this.wordsSinceComma = 0;
            return '.';
        } else if (this.generateSpace) {
            this.generateSpace = false;
            if (this.generateUpper && this.enableParagraphs) {
                this.sentencesInParagraph++;
                if (this.sentencesInParagraph > this.paragraphProbTable.length) {
                    this.sentencesInParagraph = this.paragraphProbTable.length;
                }
                if (remainingCharacters > MIN_LAST_PARAGRAPH_LETTERS) {
                    let prob = this.paragraphProbTable[this.sentencesInParagraph - 1];
                    let rand = this.randLCG() % 255;
                    if (rand < prob) {
                        this.sentencesInParagraph = 0;
                        return this.paragraphSeparator;
                    }
                }
            }
            return ' ';
        }

        let letterIndex = this.generateLetter(remainingCharacters - 1);
        this.updateContext(letterIndex);

        if (letterIndex == 0) {
            this.wordsSinceDot++;
            if (this.wordsSinceDot > 40) {
                this.wordsSinceDot = 40;
            }
            this.wordsSinceComma++;
            if (this.wordsSinceComma > 20) {
                this.wordsSinceComma = 20;
            }
            if (remainingCharacters > MIN_LAST_SENTENCE_LETTERS) {
                let prob = this.model.prob_dot[this.wordsSinceDot - 1][this.wordsSinceComma - 1] + 1;
                let rand = (this.randLCG() >> 16) & 0xFF;
                if (rand < prob) {
                    this.generateSpace = true;
                    this.generateUpper = true;
                    this.wordsSinceDot = 0;
                    this.wordsSinceComma = 0;
                    return '.';
                }
                prob = this.model.prob_comma[this.wordsSinceDot - 1][this.wordsSinceComma - 1] + 1;
                rand = (this.randLCG() >> 16) & 0xFF;
                if (rand < prob) {
                    this.generateSpace = true;
                    this.wordsSinceComma = 0;
                    return ',';
                }
            }
        }

        if (this.generateUpper) {
            this.generateUpper = false;
            return this.model.letters[letterIndex].toUpperCase();
        } else {
            return this.model.letters[letterIndex];
        }
    }


    public setContext(contextText?: string): void {

        this.resetContext();

        if (contextText == null || contextText == '') {
            return; // Just reset the generator state
        }

        let i = 0;

        while (i < contextText.length) {
            let currentChar = contextText[i];
            if (currentChar === '.') {
                this.wordsSinceDot = 0;
                this.wordsSinceComma = 0;
                this.generateSpace = true;
                this.generateUpper = true;
                this.updateContext(0);
                i++;
            } else if (currentChar === ',') {
                this.wordsSinceDot++;
                this.wordsSinceComma = 0;
                this.generateSpace = true;
                this.updateContext(0);
                i++;
            } else if (currentChar === ' ' || contextText.substring(i, i + this.paragraphSeparator.length) === this.paragraphSeparator) {
                if (!(this.generateSpace)) {
                    this.wordsSinceDot++;
                    this.wordsSinceComma++;
                    this.updateContext(0);
                } else {
                    this.generateSpace = false;
                    if (this.generateUpper) {
                        this.sentencesInParagraph++;
                        if (contextText.substring(i, i + this.paragraphSeparator.length) === this.paragraphSeparator) {
                            this.sentencesInParagraph = 0;
                        }
                    }
                }
                i++;
            } else {
                let letterIndex: number | undefined = this.letterToIndex[currentChar];
                if (letterIndex != null) {
                    this.generateSpace = false;
                    this.generateUpper = false;
                    this.updateContext(letterIndex);
                }
                i++;
            }
        }
    }


    // #endregion ------------------------------------------------------------
    // #region                 Neural network execution
    // -----------------------------------------------------------------------


    private executeLinear(layer: ILinearLayer, input: number[]): number[] {

        const input_size = layer.input_shift.length;
        const output_size = layer.bias.length;
        const input_shift = layer.input_shift;
        const input_clamp = layer.input_clamp;
        const weight = layer.weight;
        const bias = layer.bias;

        for (let i = 0; i < input_size; i++) {
            let x = input[i];
            x >>= input_shift[i];
            if (x < input_clamp[0][i]) x = input_clamp[0][i];
            if (x > input_clamp[1][i]) x = input_clamp[1][i];
            input[i] = x;
        }

        const output: number[] = new Array(output_size);

        for (let row = 0; row < output_size; row++) {
            let sum = bias[row];
            for (let col = 0; col < input_size; col++) {
                sum += input[col] * weight[row][col];
            }
            output[row] = sum;
        }

        return output;
    }


    private executeReLU(layer: IReLULayer, inout: number[]): void {

        const input_size = layer.input_size;
        for (let i = 0; i < input_size; i++) {
            if (inout[i] < 0) inout[i] = 0;
        }
    }


    private executeScaledSoftmax(layer: IScaledSoftmaxLayer, inout: number[]): void {

        const input_size = layer.weight.length;
        const weight = layer.weight;
        const range_top = (8n << BigInt(layer.frac_bits)) - 1n;
        const tmp = new Array(input_size);
        let max = -(1n << 62n);

        for (let i = 0; i < input_size; i++) {
            const weight_scaled = weight[i] * this.invHeat;
            const x = BigInt(inout[i]) * BigInt(weight_scaled);
            if (x > max) max = x;
            tmp[i] = x;
        }

        for (let i = 0; i < input_size; i++) {
            let y;
            let x;
            let x64 = tmp[i] - max + range_top;
            if (x64 > range_top) x64 = -1n; // Overflow protection
            x = Number(x64 >> BigInt(layer.frac_bits - 9));
            if (x >= 0) {
                y = EXP_TABLE[0][(x >> 9) & 7];
                y = (y * EXP_TABLE[1][(x >> 6) & 7]) >> 9;
                y = (y * EXP_TABLE[2][(x >> 3) & 7]) >> 9;
                y = (y * EXP_TABLE[3][x & 7]) >> (9 + 9);
            } else {
                y = 0;
            }
            inout[i] = y;
        }
    }

    private executeNN(layers: { type: string }[], vector: number[]): number[] {
        vector = vector.slice();
        for (const layer of layers) {
            switch (layer.type) {
                case 'linear':
                    vector = this.executeLinear(layer as ILinearLayer, vector);
                    break;
                case 'relu':
                    this.executeReLU(layer as IReLULayer, vector);
                    break;
                case 'scaled_softmax':
                    this.executeScaledSoftmax(layer as IScaledSoftmaxLayer, vector);
                    break;
                default:
                    throw new Error(`Unknown layer type: ${layer.type}`);
            }
        }
        return vector;
    }


    // #endregion ------------------------------------------------------------
    // #region                 Random number generator
    // -----------------------------------------------------------------------


    private randLCG(): number {
        let result = this.randState;
        this.randState = ((Math.imul(1664525, result) >>> 0) + 1013904223) >>> 0;
        return result >>> 8;
    }


    private randomFromCumsum(cumsum: number[]): number {
        const maxValue = cumsum[cumsum.length - 1];
        if (maxValue <= 0) {
            return this.randLCG() % cumsum.length;
        }
        const rand = this.randLCG() % maxValue;
        let start = 0;
        let end = cumsum.length;
        while (start < end) {
            let mid = (start + end) >> 1;
            if (rand < cumsum[mid]) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }
        return start;
    }


    private randomFromProbs(probs: number[]): number {
        let sum: number = 0;
        for (let i = 0; i < probs.length; i++) {
            sum += probs[i];
            probs[i] = sum;
        }
        return this.randomFromCumsum(probs);
    }


    // #endregion ------------------------------------------------------------
    // #region             Generation and context management
    // -----------------------------------------------------------------------


    private generateLetter(remainingCharacters: number) {
        // Concatenate previous groups and current group, and execute the head NN
        let vect = this.groups[0].concat(this.groups[4], this.groups[8]);
        vect = this.executeNN(this.model.head, vect);

        // Prevent unnecessary spaces (no repeating spaces, spaces near the end, at the beginning or after repeating words)
        if (vect[0] > 0 && (remainingCharacters < 3 || this.lastWordHash == this.currentWordHash || this.currentWordHash == (0xFFFFFFFF | 0))) {
            vect[0] = 0;
        }

        // Get random letter index based on probabilities
        return this.randomFromProbs(vect);
    }


    private resetContext(): void {
        // Fill group context with spaces
        this.groupContext = [
            ...this.model.letters_embedding[0],
            ...this.model.letters_embedding[0],
            ...this.model.letters_embedding[0],
            ...this.model.letters_embedding[0],
        ];

        // Calculate initial group embeddings for spaces
        const groupFeatures = this.executeNN(this.model.group, this.groupContext);
        this.groups = new Array(9).fill(groupFeatures);

        // Initial group hashes
        this.lastWordHash = 0xFFFFFFFF | 0;
        this.currentWordHash = 0xFFFFFFFF | 0;

        // Initial punctuation state
        this.generateUpper = true;
        this.generateSpace = false;
        this.wordsSinceDot = 0;
        this.wordsSinceComma = 0;
        this.sentencesInParagraph = 0;
    }


    private updateContext(letterIndex: number): void {

        // Update word hashes
        if (letterIndex == 0) {
            this.lastWordHash = this.currentWordHash;
            this.currentWordHash = 0xFFFFFFFF | 0;
        } else {
            this.currentWordHash = (this.currentWordHash << 6) | (letterIndex & 0x3F);
        }
        // Shift in the new letter index into the group context
        this.groupContext.splice(0, 3);
        this.groupContext.push(...this.model.letters_embedding[letterIndex]);
        // Remove old group
        this.groups.shift();

        // Execute group NN with current group context
        let newGroup = this.executeNN(this.model.group, this.groupContext);
        this.groups.push(newGroup);
    }


    // #endregion ------------------------------------------------------------
    // #region                    Utility functions
    // -----------------------------------------------------------------------


    private normalDist(mean: number, variance: number, x: number): number {
        x = x - mean;
        if (x < 0) x = -x;
        let exp = Math.trunc(4095 - x * x * (1 << 9) / (10 * 2 * variance));
        if (exp >= 0) {
            let y = EXP_TABLE[0][(exp >> 9) & 0x7];
            y = (y * EXP_TABLE[1][(exp >> 6) & 0x7]) >> 9;
            y = (y * EXP_TABLE[2][(exp >> 3) & 0x7]) >> 9;
            y = (y * EXP_TABLE[3][(exp >> 0) & 0x7]) >> (9 + 9);
            return y;
        } else {
            return 0;
        }
    }


    debugDumpState() {
        console.log(`rand_state: ${this.randState}`);
        console.log(`last_word_hash: ${this.lastWordHash >>> 0}`);
        console.log(`current_word_hash: ${this.currentWordHash >>> 0}`);
        console.log(`inv_heat_u3_4: ${this.invHeat}`);
        console.log(`words_since_dot: ${this.wordsSinceDot}`);
        console.log(`words_since_comma: ${this.wordsSinceComma}`);
        console.log(`sentences_in_paragraph: ${this.sentencesInParagraph}`);
        console.log(`enable_paragraphs: ${this.enableParagraphs ? 'true' : 'false'}`);
        console.log(`generate_upper: ${this.generateUpper ? 'true' : 'false'}`);
        console.log(`generate_upper: ${this.generateUpper ? 'true' : 'false'}`);
        console.log('groups:');
        console.log(this.groups.map((g, i) => `    ${g.join(' ')}`).join('\n'));
        console.log('group_context: ' + this.groupContext.join(' '));
        console.log('paragraph_prob_table:');
        console.log('    ' + this.paragraphProbTable.map((x, i) => (i % 16 == 15) ? `${x}\n    ` : `${x} `).join(''));
    }


    // #endregion ------------------------------------------------------------

}
