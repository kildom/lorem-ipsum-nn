
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "lorem-ipsum.h"
#include "lorem-ipsum-int.h"


#ifdef _VSCODE_       /* ----------------------------------------------------------------------- */
#pragma region                              Local definitions and variables
#endif                /* ----------------------------------------------------------------------- */


#define MIN_LAST_SENTENCE_LETTERS 20
#define MIN_LAST_PARAGRAPH_LETTERS 80

#define FLAG_GENERATE_SPACE (1 << 0)
#define FLAG_GENERATE_UPPER (1 << 1)
#define FLAG_ENABLE_PARAGRAPHS (1 << 2)

static const int32_t EXP_TABLE[4][8] = {
    { 512, 1392, 3783, 10284, 27954, 75988, 206556, 561476 },
    { 512, 580, 657, 745, 844, 957, 1084, 1228 },
    { 512, 520, 528, 537, 545, 554, 562, 571 },
    { 512, 513, 514, 515, 516, 517, 518, 519 },
};

static const LoremIpsumModel *const models[] = { LOREM_IPSUM_MODELS };
static const char *const dot_string = ".";
static const char *const comma_string = ",";
static const char *const new_line_string = "\n";
static const char* models_names[LOREM_IPSUM_MODELS_COUNT + 1] = { NULL };

static void reset_context(LoremIpsum* ipsum);
static uint32_t rand_lcg(LoremIpsum* ipsum);
static int32_t generate_letter(LoremIpsum* ipsum, int32_t remaining_characters);
static void update_context(LoremIpsum* ipsum, int32_t letter_index);
static int32_t get_punctuation_prob(const uint8_t* prob, int32_t words_since_dot, int32_t words_since_comma);
static int32_t find_letter_index(const char* const* letters, uint32_t letters_count, const char** text);
static uint32_t normal_dist(int32_t mean, int32_t variance, int32_t x);


#ifdef _VSCODE_
#pragma endregion     /* ----------------------------------------------------------------------- */
#pragma region                                      Public interface
#endif                /* ----------------------------------------------------------------------- */


bool lorem_ipsum_init(LoremIpsum* ipsum, const char* language, uint32_t heat_percent, uint32_t seed, uint32_t ver)
{
    int i;
    memset(ipsum, 0, sizeof(LoremIpsum));

    // Find the model for the specified language
    if (language == NULL) {
        ipsum->t.model = models[0]; // Default to the first model
    } else {
        for (i = 0; i < LOREM_IPSUM_MODELS_COUNT; i++) {
            if (strcmp(language, models[i]->lang) == 0) {
                ipsum->t.model = models[i];
                break;
            }
        }
        if (ipsum->t.model == NULL) {
            return false; // Language not supported
        }
    }

    lorem_ipsum_set_seed(ipsum, seed);
    lorem_ipsum_set_heat(ipsum, heat_percent);
    lorem_ipsum_set_paragraphs(ipsum, LOREM_IPSUM_PARAGRAPHS_DISABLE, 0, 0, NULL);
    reset_context(ipsum);

    return (ver <= 1);
}


const char *const * lorem_ipsum_languages()
{
    if (models_names[0] == NULL) {
        int i;
        for (i = 0; i < LOREM_IPSUM_MODELS_COUNT; i++) {
            models_names[i] = models[i]->lang;
        }
        models_names[LOREM_IPSUM_MODELS_COUNT] = NULL;
    }
    return models_names;
}


size_t lorem_ipsum_generate(LoremIpsum* ipsum, char* buffer, size_t buffer_size)
{
    char* initial_buffer = buffer;
    if (buffer_size <= 0) {
        return 0;
    }
    while (buffer_size > 2) {
        const char* next_char = lorem_ipsum_next(ipsum, buffer_size - 2);
        size_t len = strlen(next_char);
        if (buffer_size < len + 2) {
            break;
        }
        // Do simple copy of UTF-8 bytes of one character (up to 4 bytes) which is faster than using `strcpy`.
        buffer[0] = next_char[0];
        if (next_char[1]) {
            buffer[1] = next_char[1];
            if (next_char[2]) {
                buffer[2] = next_char[2];
                if (next_char[3]) {
                    buffer[3] = next_char[3];
                    // We still have a user provided paragraph separator, so we can copy the rest of the string.
                    if (next_char[4]) {
                        strcpy(buffer + 4, next_char + 4);
                    }
                }
            }
        }
        buffer += len;
        buffer_size -= len;
    }
    if (buffer_size > 1) {
        *buffer++ = '.';
    }
    *buffer++ = '\0';
    // Update state, so the next call will start a new sentence.
    update_context(ipsum, 0);
    ipsum->s.flags |= FLAG_GENERATE_SPACE | FLAG_GENERATE_UPPER;
    ipsum->s.words_since_dot = 0;
    ipsum->s.words_since_comma = 0;
    return buffer - initial_buffer;
}


const char* lorem_ipsum_next(LoremIpsum* ipsum, size_t remaining_characters)
{
    if (remaining_characters < 1) {
        ipsum->s.flags |= FLAG_GENERATE_SPACE | FLAG_GENERATE_UPPER;
        ipsum->s.words_since_dot = 0;
        ipsum->s.words_since_comma = 0;
        return dot_string;
    } else if (ipsum->s.flags & FLAG_GENERATE_SPACE) {
        ipsum->s.flags &= ~FLAG_GENERATE_SPACE;
        if ((ipsum->s.flags & FLAG_GENERATE_UPPER) && (ipsum->s.flags & FLAG_ENABLE_PARAGRAPHS)) {
            ipsum->s.sentences_in_paragraph++;
            if (ipsum->s.sentences_in_paragraph > (int32_t)sizeof(ipsum->s.paragraph_prob_table)) {
                ipsum->s.sentences_in_paragraph = sizeof(ipsum->s.paragraph_prob_table);
            }
            if (remaining_characters > MIN_LAST_PARAGRAPH_LETTERS) {
                uint32_t prob = ipsum->s.paragraph_prob_table[ipsum->s.sentences_in_paragraph - 1];
                uint32_t rand = rand_lcg(ipsum) % 255;
                if (rand < prob) {
                    ipsum->s.sentences_in_paragraph = 0;
                    return ipsum->t.paragraph_separator;
                }
            }
        }
        return ipsum->t.model->lower_letters[0];
    }

    int32_t letter_index = generate_letter(ipsum, remaining_characters - 1);
    update_context(ipsum, letter_index);

    if (letter_index == 0) {
        ipsum->s.words_since_dot++;
        if (ipsum->s.words_since_dot > 40) {
            ipsum->s.words_since_dot = 40;
        }
        ipsum->s.words_since_comma++;
        if (ipsum->s.words_since_comma > 20) {
            ipsum->s.words_since_comma = 20;
        }
        if (remaining_characters > MIN_LAST_SENTENCE_LETTERS) {
            int32_t prob = get_punctuation_prob(ipsum->t.model->prob_dot, ipsum->s.words_since_dot, ipsum->s.words_since_comma);
            int32_t rand = (rand_lcg(ipsum) >> 16) & 0xFF;
            if (rand < prob) {
                ipsum->s.flags |= FLAG_GENERATE_SPACE | FLAG_GENERATE_UPPER;
                ipsum->s.words_since_dot = 0;
                ipsum->s.words_since_comma = 0;
                return dot_string;
            }
            prob = get_punctuation_prob(ipsum->t.model->prob_comma, ipsum->s.words_since_dot, ipsum->s.words_since_comma);
            rand = (rand_lcg(ipsum) >> 16) & 0xFF;
            if (rand < prob) {
                ipsum->s.flags |= FLAG_GENERATE_SPACE;
                ipsum->s.words_since_comma = 0;
                return comma_string;
            }
        }
    }

    if (ipsum->s.flags & FLAG_GENERATE_UPPER) {
        ipsum->s.flags &= ~FLAG_GENERATE_UPPER;
        return ipsum->t.model->upper_letters[letter_index];
    } else {
        return ipsum->t.model->lower_letters[letter_index];
    }
}


void lorem_ipsum_set_context(LoremIpsum* ipsum, const char* context_text)
{
    const LoremIpsumModel* model = ipsum->t.model;

    reset_context(ipsum);

    if (context_text == NULL || context_text[0] == '\0') {
        return; // Just reset the generator state
    }

    size_t paragraph_separator_length = strlen(ipsum->t.paragraph_separator);

    while (*context_text) {
        bool is_paragraph_separator = (strncmp(context_text, ipsum->t.paragraph_separator, paragraph_separator_length) == 0);
        if (*context_text == '.') {
            ipsum->s.words_since_dot = 0;
            ipsum->s.words_since_comma = 0;
            ipsum->s.flags |= FLAG_GENERATE_SPACE | FLAG_GENERATE_UPPER;
            update_context(ipsum, 0);
            context_text++;
        } else if (*context_text == ',') {
            ipsum->s.words_since_dot++;
            ipsum->s.words_since_comma = 0;
            ipsum->s.flags |= FLAG_GENERATE_SPACE;
            update_context(ipsum, 0);
            context_text++;
        } else if (*context_text == ' ' || is_paragraph_separator) {
            if (!(ipsum->s.flags & FLAG_GENERATE_SPACE)) {
                ipsum->s.words_since_dot++;
                ipsum->s.words_since_comma++;
                update_context(ipsum, 0);
            } else {
                ipsum->s.flags &= ~FLAG_GENERATE_SPACE;
                if (ipsum->s.flags & FLAG_GENERATE_UPPER) {
                    ipsum->s.sentences_in_paragraph++;
                    if (is_paragraph_separator) {
                        ipsum->s.sentences_in_paragraph = 0;
                    }
                }
            }
            context_text++;
        } else {
            int32_t letter_index = find_letter_index(model->lower_letters, model->letters_count, &context_text);
            if (letter_index < 0) {
                letter_index = find_letter_index(model->upper_letters, model->letters_count, &context_text);
            }
            if (letter_index >= 0) {
                ipsum->s.flags &= ~(FLAG_GENERATE_UPPER | FLAG_GENERATE_SPACE);
                update_context(ipsum, letter_index);
            } else {
                context_text++;
            }
        }
    }
}


void lorem_ipsum_set_paragraphs(LoremIpsum* ipsum, int32_t mean, int32_t shorter_variance, int32_t longer_variance, const char* separator)
{
    int i;
    uint8_t* table = ipsum->s.paragraph_prob_table;
    int32_t sum = 0;
    ipsum->t.paragraph_separator = separator ? separator : new_line_string;
    if (mean == LOREM_IPSUM_PARAGRAPHS_DISABLE) {
        ipsum->s.flags &= ~FLAG_ENABLE_PARAGRAPHS;
        memset(table, 0, sizeof(ipsum->s.paragraph_prob_table));
        return;
    }
    ipsum->s.flags |= FLAG_ENABLE_PARAGRAPHS;
    if (mean == LOREM_IPSUM_PARAGRAPHS_DEFAULT) {
        mean = 50;
        shorter_variance = 20;
        longer_variance = 40;
    }
    if (shorter_variance <= 0) {
        shorter_variance = 1;
    }
    if (longer_variance <= 0) {
        longer_variance = 1;
    }
    table[sizeof(ipsum->s.paragraph_prob_table) - 1] = 255;
    for (i = sizeof(ipsum->s.paragraph_prob_table) - 2; i >= 0; i--) {
        int32_t x = 10 * (i + 1);
        int32_t prob_ind = normal_dist(mean, (x <= mean) ? shorter_variance : longer_variance, x);
        sum += prob_ind;
        if (sum > 0) {
            table[i] = (prob_ind * 255) / sum;
        } else {
            table[i] = 255;
        }
    }
}


void lorem_ipsum_set_heat(LoremIpsum* ipsum, uint32_t heat_percent)
{
    // Fixed-point representation for inverse heat is u3.4 (7 bits total)
    ipsum->s.inv_heat_u3_4 = heat_percent > 0 ? 1600 / heat_percent : 1600;
    if (ipsum->s.inv_heat_u3_4 > 127) ipsum->s.inv_heat_u3_4 = 127;
    if (ipsum->s.inv_heat_u3_4 < 1) ipsum->s.inv_heat_u3_4 = 1;
}


#ifdef _VSCODE_
#pragma endregion     /* ----------------------------------------------------------------------- */
#pragma region                                  Neural network execution
#endif                /* ----------------------------------------------------------------------- */


static bool execute_linear(const LoremIpsumLinear* layer, void* vinput, void* voutput)
{
    uint32_t i, col, row;
    int32_t* input = (int32_t*)vinput;
    int32_t* output = (int32_t*)voutput;
    const int8_t* weight = layer->weight;
    const int32_t* bias = layer->bias;
    const uint8_t* input_shift = layer->input_shift;
    const int32_t (*input_clamp)[2] = layer->input_clamp;
    uint32_t input_size = layer->input_size;
    uint32_t output_size = layer->output_size;

    for (i = 0; i < input_size; i++) {
        input[i] >>= input_shift[i];
        if (input[i] < input_clamp[i][0]) input[i] = input_clamp[i][0];
        if (input[i] > input_clamp[i][1]) input[i] = input_clamp[i][1];
    }

    for (row = 0; row < output_size; row++) {
        int32_t sum = bias[row];
        for (col = 0; col < input_size; col++) {
            sum += input[col] * weight[input_size * row + col];
        }
        output[row] = sum;
    }

    return false;
}


static bool execute_relu(const LoremIpsumReLU* layer, void* vinput)
{
    uint32_t i;
    uint32_t input_size = layer->input_size;
    int32_t* input = (int32_t*)vinput;

    for (i = 0; i < input_size; i++) {
        if (input[i] < 0) input[i] = 0;
    }

    return true;
}


static bool execute_scaled_softmax(LoremIpsum* ipsum, const LoremIpsumScaledSoftmax* layer, void* vinput, void* voutput)
{
    uint32_t i;
    uint32_t input_size = layer->input_size;
    int32_t* input = (int32_t*)vinput;
    int64_t range_top = ((int64_t)8 << layer->frac_bits) - 1;
    int64_t* tmp = (int64_t*)voutput;
    int64_t max = -((int64_t)1 << 62);

    for (i = 0; i < input_size; i++) {
        int32_t weight_scaled = layer->weight[i] * ipsum->s.inv_heat_u3_4;
        tmp[i] = (int64_t)input[i] * (int64_t)weight_scaled;
        if (tmp[i] > max) max = tmp[i];
    }

    for (i = 0; i < input_size; i++) {
        int32_t y;
        int32_t x;
        int64_t x64 = tmp[i] - max + range_top;
        if (x64 > range_top) x64 = -1; // Overflow protection
        x = (int32_t)(x64 >> (layer->frac_bits - 9));
        if (x >= 0) {
            y = EXP_TABLE[0][(x >> 9) & 7];
            y = (y * EXP_TABLE[1][(x >> 6) & 7]) >> 9;
            y = (y * EXP_TABLE[2][(x >> 3) & 7]) >> 9;
            y = (y * EXP_TABLE[3][x & 7]) >> (9 + 9);
        } else {
            y = 0;
        }
        input[i] = y;
    }

    return true;
}


static void* execute_nn(LoremIpsum* ipsum, const void* const* layers)
{
    bool in_place = false;
    void* current_vector = ipsum->t.vector_buffer0;
    void* next_vector = ipsum->t.vector_buffer1;
    const LoremIpsumUnknownLayer* const* layer = (const LoremIpsumUnknownLayer* const*)layers;
    while (*layer) {
        switch ((*layer)->type) {
            case LOREM_IPSUM_LAYER_LINEAR:
                in_place = execute_linear((const LoremIpsumLinear*)(*layer), current_vector, next_vector);
                break;
            case LOREM_IPSUM_LAYER_RELU:
                in_place = execute_relu((const LoremIpsumReLU*)(*layer), current_vector);
                break;
            case LOREM_IPSUM_LAYER_SCALED_SOFTMAX:
                in_place = execute_scaled_softmax(ipsum, (const LoremIpsumScaledSoftmax*)(*layer), current_vector, next_vector);
                break;
            default:
                // ASSERT this
                break;
        }
        if (!in_place) {
            void* tmp = current_vector;
            current_vector = next_vector;
            next_vector = tmp;
        }
        layer++;
    }
    return current_vector;
}


#ifdef _VSCODE_
#pragma endregion     /* ----------------------------------------------------------------------- */
#pragma region                                  Random number generator
#endif                /* ----------------------------------------------------------------------- */


static uint32_t rand_lcg(LoremIpsum* ipsum)
{
    uint32_t result = ipsum->s.rand_state;
    ipsum->s.rand_state = 1664525u * result + 1013904223u;
    return result >> 8;
}

static uint32_t random_from_cumsum(LoremIpsum* ipsum, int32_t* cumsum, uint32_t count)
{
    int32_t max_value = cumsum[count - 1];
    if (max_value <= 0) {
        return rand_lcg(ipsum) % count;
    }
    int32_t rand = rand_lcg(ipsum) % max_value;
    int start = 0;
    int end = count;
    while (start < end) {
        int mid = (start + end) >> 1;
        if (rand < cumsum[mid]) {
            end = mid;
        } else {
            start = mid + 1;
        }
    }
    return start;
}

static uint32_t random_from_probs(LoremIpsum* ipsum, int32_t* probs, uint32_t count)
{
    uint32_t i;
    int32_t sum = 0;
    for (i = 0; i < count; i++) {
        sum += probs[i];
        probs[i] = sum;
    }
    return random_from_cumsum(ipsum, probs, count);
}


#ifdef _VSCODE_
#pragma endregion     /* ----------------------------------------------------------------------- */
#pragma region                             Generation and context management
#endif                /* ----------------------------------------------------------------------- */


static int32_t generate_letter(LoremIpsum* ipsum, int32_t remaining_characters)
{
    const LoremIpsumModel* model = ipsum->t.model;
    int32_t* vect = (int32_t*)ipsum->t.vector_buffer0;

    // Concatenate previous groups and current group, and execute the head NN
    int32_t* group0 = ipsum->s.groups[(ipsum->s.current_group + (9 - 8)) % 9];
    int32_t* group1 = ipsum->s.groups[(ipsum->s.current_group + (9 - 4)) % 9];
    int32_t* group2 = ipsum->s.groups[ipsum->s.current_group];
    memcpy(&vect[0], group0, sizeof(ipsum->s.groups[0]));
    memcpy(&vect[6], group1, sizeof(ipsum->s.groups[0]));
    memcpy(&vect[12], group2, sizeof(ipsum->s.groups[0]));
    int32_t* prob = (int32_t*)execute_nn(ipsum, model->head);

    // Prevent unnecessary spaces (no repeating spaces, spaces near the end, at the beginning or after repeating words)
    if (prob[0] > 0 && (remaining_characters < 3 || ipsum->s.last_word_hash == ipsum->s.current_word_hash || ipsum->s.current_word_hash == 0xFFFFFFFF)) {
        prob[0] = 0;
    }

    // Get random letter index based on probabilities
    return random_from_probs(ipsum, prob, model->letters_count);
}


static void reset_context(LoremIpsum* ipsum)
{
    int i;
    int32_t* vect = (int32_t*)ipsum->t.vector_buffer0;

    // Fill group context with spaces
    ipsum->s.group_context[0] = ipsum->t.model->letters_embedding[0];
    ipsum->s.group_context[1] = ipsum->t.model->letters_embedding[1];
    ipsum->s.group_context[2] = ipsum->t.model->letters_embedding[2];
    memcpy(&ipsum->s.group_context[3], &ipsum->s.group_context[0], sizeof(int32_t) * 3);
    memcpy(&ipsum->s.group_context[6], &ipsum->s.group_context[0], sizeof(int32_t) * 6);

    // Calculate initial group embeddings for spaces
    memcpy(vect, ipsum->s.group_context, sizeof(ipsum->s.group_context));
    int32_t* group_output = (int32_t*)execute_nn(ipsum, ipsum->t.model->group);
    ipsum->s.current_group = 0;
    for (i = 0; i < 9; i++) {
        memcpy(ipsum->s.groups[i], group_output, sizeof(ipsum->s.groups[i]));
    }

    // Initial group hashes
    ipsum->s.last_word_hash = 0xFFFFFFFF;
    ipsum->s.current_word_hash = 0xFFFFFFFF;

    // Initial punctuation state
    ipsum->s.flags |= FLAG_GENERATE_UPPER;
    ipsum->s.flags &= ~FLAG_GENERATE_SPACE;
    ipsum->s.words_since_dot = 0;
    ipsum->s.words_since_comma = 0;
    ipsum->s.sentences_in_paragraph = 0;
}


static void update_context(LoremIpsum* ipsum, int32_t letter_index)
{
    int32_t* vect = (int32_t*)ipsum->t.vector_buffer0;

    // Update word hashes
    if (letter_index == 0) {
        ipsum->s.last_word_hash = ipsum->s.current_word_hash;
        ipsum->s.current_word_hash = 0xFFFFFFFF;
    } else {
        ipsum->s.current_word_hash = (ipsum->s.current_word_hash << 6) | (letter_index & 0x3F);
    }
    // Shift in the new letter index into the group context
    memmove(&ipsum->s.group_context[0], &ipsum->s.group_context[3], sizeof(int32_t) * 9);
    const uint8_t* embedding = &ipsum->t.model->letters_embedding[3 * letter_index];
    ipsum->s.group_context[9] = embedding[0];
    ipsum->s.group_context[10] = embedding[1];
    ipsum->s.group_context[11] = embedding[2];
    // Make next group active
    ipsum->s.current_group = (ipsum->s.current_group + 1) % 9;

    // Execute group NN with current group context
    memcpy(vect, ipsum->s.group_context, sizeof(ipsum->s.group_context));
    int32_t* group_output = (int32_t*)execute_nn(ipsum, ipsum->t.model->group);
    memcpy(ipsum->s.groups[ipsum->s.current_group], group_output, sizeof(ipsum->s.groups[0]));
}


#ifdef _VSCODE_
#pragma endregion     /* ----------------------------------------------------------------------- */
#pragma region                                      Utility functions
#endif                /* ----------------------------------------------------------------------- */


static int32_t get_punctuation_prob(const uint8_t* prob, int32_t words_since_dot, int32_t words_since_comma)
{
    int32_t row = words_since_dot - 1;
    int32_t col = words_since_comma - 1;
    int32_t index = col;
    if (row < 20) {
        index += row * (row + 1) / 2;
    } else {
        index += row * 20 - 190;
    }
    return prob[index] + 1;
}


static int32_t find_letter_index(const char* const* letters, uint32_t letters_count, const char** text)
{
    uint32_t i;
    const char* start = *text;
    for (i = 0; i < letters_count; i++) {
        int len = strlen(letters[i]);
        if (strncmp(letters[i], start, len) == 0) {
            *text += len;
            return i;
        }
    }
    return -1;
}


static uint32_t normal_dist(int32_t mean, int32_t variance, int32_t x)
{
    x = x - mean;
    if (x < 0) x = -x;
    int32_t exp = 4095 - x * x * (1 << 9) / (10 * 2 * variance);
    if (exp >= 0) {
        uint32_t y = EXP_TABLE[0][(exp >> 9) & 0x7];
        y = (y * EXP_TABLE[1][(exp >> 6) & 0x7]) >> 9;
        y = (y * EXP_TABLE[2][(exp >> 3) & 0x7]) >> 9;
        y = (y * EXP_TABLE[3][(exp >> 0) & 0x7]) >> (9 + 9);
        return y;
    } else {
        return 0;
    }
}


#ifdef LOREM_IPSUM_DEBUG

#include <stdio.h>

void lorem_ipsum_print_state(const LoremIpsum* ipsum)
{
    printf("rand_state: %u\n", ipsum->s.rand_state);
    printf("last_word_hash: %u\n", ipsum->s.last_word_hash);
    printf("current_word_hash: %u\n", ipsum->s.current_word_hash);
    printf("inv_heat_u3_4: %d\n", ipsum->s.inv_heat_u3_4);
    printf("words_since_dot: %d\n", ipsum->s.words_since_dot);
    printf("words_since_comma: %d\n", ipsum->s.words_since_comma);
    printf("sentences_in_paragraph: %d\n", ipsum->s.sentences_in_paragraph);
    printf("enable_paragraphs: %s\n", ipsum->s.flags & FLAG_ENABLE_PARAGRAPHS ? "true" : "false");
    printf("generate_space: %s\n", ipsum->s.flags & FLAG_GENERATE_SPACE ? "true" : "false");
    printf("generate_upper: %s\n", ipsum->s.flags & FLAG_GENERATE_UPPER ? "true" : "false");
    printf("groups:\n");
    for (int i = 0; i < 9; i++) {
        printf("    ");
        for (int j = 0; j < 6; j++) {
            printf("%d ", ipsum->s.groups[(ipsum->s.current_group + 1 + i) % 9][j]);
        }
        printf("\n");
    }
    printf("group_context: ");
    for (int i = 0; i < 12; i++) {
        printf("%d ", ipsum->s.group_context[i]);
    }
    printf("paragraph_prob_table: \n    ");
    for (int i = 0; i < 64; i++) {
        printf("%d%s", ipsum->s.paragraph_prob_table[i], i % 16 == 15 ? "\n    " : " ");
    }
    printf("\n");
}

#endif


#ifdef _VSCODE_
#pragma endregion
#endif                /* ----------------------------------------------------------------------- */
