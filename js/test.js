
let fs = require('fs');

let modelData = JSON.parse(fs.readFileSync('models/la.json', 'utf8'));

function prepareModel() {
    modelData.letter_to_index = {};
    for (let i = 0; i < modelData.letters.length; i++) {
        modelData.letter_to_index[modelData.letters[i]] = i;
    }
}

prepareModel();


function strToEmbedding(str) {
    let result = [];
    for (let i = 0; i < str.length; i++) {
        result = [...result, ...modelData.letters_embedding[modelData.letter_to_index[str[i]]]];
    }
    return result;
}

const EXP_TABLE = [
    [512, 1392, 3783, 10284, 27954, 75988, 206556, 561476],
    [512, 580, 657, 745, 844, 957, 1084, 1228],
    [512, 520, 528, 537, 545, 554, 562, 571],
    [512, 513, 514, 515, 516, 517, 518, 519]
];


function evalModel(layers, input, softmax_scale) {
    let value = input.slice();
    for (let layer of layers) {
        switch (layer.type) {

            case 'relu': {
                for (let i = 0; i < value.length; i++) {
                    value[i] = Math.max(0, value[i]);
                }
                break;
            }

            case 'linear': {
                for (let i = 0; i < value.length; i++) {
                    value[i] >>= layer.input_shift[i];
                    if (value[i] < layer.input_clamp[0][i]) value[i] = layer.input_clamp[0][i];
                    if (value[i] > layer.input_clamp[1][i]) value[i] = layer.input_clamp[1][i];
                }
                let result = new Array(layer.bias.length);
                for (let row = 0; row < result.length; row++) {
                    let sum = layer.bias[row];
                    for (let col = 0; col < value.length; col++) {
                        sum += value[col] * layer.weight[row][col];
                    }
                    result[row] = sum;
                }
                value = result;
                break;
            }

            case 'scaled_softmax': {
                softmax_scale = softmax_scale || 1;
                let max_x = -(1n << 62n);
                for (let i = 0; i < value.length; i++) {
                    let weight_scaled = layer.weight[i] * softmax_scale;
                    value[i] = BigInt(value[i]) * BigInt(weight_scaled);
                    if (max_x < value[i]) {
                        max_x = value[i];
                    }
                }
                let range_top = (8n << BigInt(layer.frac_bits)) - 1n;
                for (let i = 0; i < value.length; i++) {
                    let x = value[i] - max_x + range_top;
                    x = Number(x >> BigInt(layer.frac_bits - 9))
                    let y;
                    if (x >= 0) {
                        y = EXP_TABLE[0][(x >>> 9) & 7];
                        y = (y * EXP_TABLE[1][(x >>> 6) & 7]) >>> 9;
                        y = (y * EXP_TABLE[2][(x >>> 3) & 7]) >>> 9;
                        y = (y * EXP_TABLE[3][x & 7]) >>> 9;
                    } else {
                        y = 0;
                    }
                    value[i] = y;
                }
                break;
            }

            default:
                throw new Error(`Unknown layer type: ${layer.type}`);
        }
    }
    return value;

}

function random_from_cumsum(probs) {
    let maxValue = probs.at(-1);
    let rand = Math.floor(Math.random() * maxValue);
    let start = 0;
    let end = probs.length;
    while (start < end) {
        let mid = (start + end) >> 1;
        if (rand < probs[mid]) {
            end = mid;
        } else {
            start = mid + 1;
        }
    }
    return start
}

const LETTERS_PER_GROUP = 4;
const GROUPS_PER_CONTEXT = 3;
const LETTERS_PER_CONTEXT = LETTERS_PER_GROUP * GROUPS_PER_CONTEXT;
const LETTER_EMBEDDING_SIZE = 3;

class Generator {

    constructor(modelData) {
        this.modelData = modelData;
        let heat = 60;
        this.invHeatScale = Math.round(1600 / heat);
        this.letterEmbedding = strToEmbedding(' '.repeat(LETTERS_PER_GROUP));
        let emptyGroupEmb = evalModel(modelData.group, this.letterEmbedding);
        this.groupHistory = new Array(LETTERS_PER_CONTEXT - LETTERS_PER_GROUP).fill(emptyGroupEmb);
        this.contextText = ' '.repeat(2 * LETTERS_PER_CONTEXT);
    }

    generateNextLetter(remaining) {
        let g = [];
        for (let i = 0; i < this.groupHistory.length; i += LETTERS_PER_GROUP) {
            g.push(...this.groupHistory[i]);
        }
        let currentGroup = evalModel(modelData.group, this.letterEmbedding);
        this.groupHistory.shift();
        this.groupHistory.push(currentGroup);
        g.push(...currentGroup);

        let prob = evalModel(modelData.head, g, this.invHeatScale);
        let prob_cumsum = new Array(prob.length);
        let cumsum = 0;
        for (let i = 0; i < prob.length; i++) {
            let value = prob[i];
            while (i === 0 && value > 0) {
                // Avoid spaces and very short words at the end of the text
                if (remaining < 3) {
                    value = 0;
                    break;
                }
                // Avoid repeating the last word
                let index1 = this.contextText.lastIndexOf(' ');
                if (index1 < 0) break;
                let index2 = this.contextText.substring(0, index1).lastIndexOf(' ');
                if (index2 < 0) break;
                let lastWord = this.contextText.substring(index2 + 1, index1);
                let thisWord = this.contextText.substring(index1 + 1);
                if (lastWord === thisWord) {
                    value = 0;
                }
                break;
            }
            cumsum += value;
            prob_cumsum[i] = cumsum;
        }
        let letterIndex = random_from_cumsum(prob_cumsum);
        if (letterIndex >= modelData.letters.length) {
            letterIndex = 1;
        }
        this.letterEmbedding.splice(0, LETTER_EMBEDDING_SIZE);
        this.letterEmbedding.push(...modelData.letters_embedding[letterIndex]);
        this.contextText = this.contextText.slice(1) + modelData.letters[letterIndex];
        return modelData.letters[letterIndex];
    }
}

MIN_LAST_SENTENCE_LETTERS = 20;

let gen = new Generator(modelData);
let remaining = 200000;
let text = '';
let beginningOfSentence = true;
let nextDot = random_from_cumsum(modelData.prob_dot);
let nextComma = random_from_cumsum(modelData.prob_comma[nextDot]);
remaining--;
let t = Date.now();
while (remaining > 0) {
    remaining--;
    let letter = gen.generateNextLetter(remaining);
    if (beginningOfSentence) {
        letter = letter.toUpperCase();
        beginningOfSentence = false;
    }
    if (letter === ' ') {
        nextDot--;
        nextComma--;
        if (nextDot === 0 && remaining > MIN_LAST_SENTENCE_LETTERS) {
            nextDot = random_from_cumsum(modelData.prob_dot);
            nextComma = random_from_cumsum(modelData.prob_comma[nextDot]);
            beginningOfSentence = true;
            text += '. ';
            remaining--;
        } else if (nextComma === 0 && remaining > MIN_LAST_SENTENCE_LETTERS) {
            nextComma = random_from_cumsum(modelData.prob_comma[nextDot]);
            text += ', ';
            remaining--;
        } else {
            text += ' ';
        }
    } else {
        text += letter;
    }
}
text += '.';
t = Date.now() - t;

console.log(text);
console.log(`Generated ${text.length} characters in ${t} ms, ${Math.round(text.length / t * 1000)} chars/sec`);
