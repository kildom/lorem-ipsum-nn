
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

#include "lorem-ipsum.h"
#include "lorem-ipsum-int.h"

#define MIN_LAST_SENTENCE_LETTERS 20

#define FLAG_GENERATE_SPACE (1 << 0)
#define FLAG_GENERATE_UPPER (1 << 1)


static const LoremIpsumModel *const models[] = { LOREM_IPSUM_MODELS };
static const char *const dot_string = ".";
static const char *const comma_string = ",";
static const char* models_names[LOREM_IPSUM_MODELS_COUNT + 1] = { NULL };


static void reset_context(LoremIpsum* ipsum);


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

    ipsum->s.last_word_hash = 0xFFFFFFFF;
    ipsum->s.current_word_hash = 0xFFFFFFFF;

    lorem_ipsum_set_seed(ipsum, seed);
    lorem_ipsum_set_heat(ipsum, heat_percent);
    reset_context(ipsum);

    return (ver <= 1);
}

#undef PRINT

#ifdef PRINT
#define pprintf printf
#else
#define pprintf(...) do {} while (0)
#endif

void print_vect(const char* name, int32_t* vect, int size) {
    pprintf("%s: ", name);
    for (int i = 0; i < size; i++) {
        pprintf("%d ", vect[i]);
    }
    pprintf("\n");
}

void print_vect64(const char* name, int64_t* vect, int size) {
    pprintf("%s: ", name);
    for (int i = 0; i < size; i++) {
        pprintf("%ld ", vect[i]);
    }
    pprintf("\n");
}

static bool execute_linear(LoremIpsum* ipsum, const LoremIpsumLinear* layer, void* vinput, void* voutput)
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

static bool execute_relu(LoremIpsum* ipsum, const LoremIpsumReLU* layer, void* vinput, void* voutput)
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
    static const int32_t EXP_TABLE[4][8] = {
        { 512, 1392, 3783, 10284, 27954, 75988, 206556, 561476 },
        { 512, 580, 657, 745, 844, 957, 1084, 1228 },
        { 512, 520, 528, 537, 545, 554, 562, 571 },
        { 512, 513, 514, 515, 516, 517, 518, 519 },
    };

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

    print_vect64("SoftMax value:", tmp, input_size);
    pprintf("inv heat: %d\n", ipsum->s.inv_heat_u3_4);

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
                in_place = execute_linear(ipsum, (const LoremIpsumLinear*)(*layer), current_vector, next_vector);
                break;
            case LOREM_IPSUM_LAYER_RELU:
                in_place = execute_relu(ipsum, (const LoremIpsumReLU*)(*layer), current_vector, next_vector);
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

static uint32_t rand_lcg(LoremIpsum* ipsum)
{
    uint32_t result = ipsum->s.rand_state;
    ipsum->s.rand_state = (uint32_t)1664525 * result + (uint32_t)1013904223;
    pprintf("LCG: %u\n", result);
    return result >> 1;
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

static void reset_context(LoremIpsum* ipsum)
{
    int i;
    int32_t* vect = (int32_t*)ipsum->t.vector_buffer0;

    printf("\n----- RESET -----\n");

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
    for (i = 0; i < 12; i++) {
        memcpy(ipsum->s.groups[i], group_output, sizeof(ipsum->s.groups[i]));
    }

    // Initial group hashes
    ipsum->s.last_word_hash = 0xFFFFFFFF;
    ipsum->s.current_word_hash = 0xFFFFFFFF;

    // Initial punctuation state
    ipsum->s.flags = FLAG_GENERATE_UPPER;
    ipsum->s.words_since_dot = 0;
    ipsum->s.words_since_comma = 0;
}

static int32_t generate_only(LoremIpsum* ipsum, int32_t remaining_characters)
{
    const LoremIpsumModel* model = ipsum->t.model;
    int32_t* vect = (int32_t*)ipsum->t.vector_buffer0;

    // // Execute group NN with current group context
    // memcpy(vect, ipsum->s.group_context, sizeof(ipsum->s.group_context));
    // //memset(ipsum->t.vector_buffer0, 0, sizeof(ipsum->t.vector_buffer0));
    // print_vect("Group input", vect, sizeof(ipsum->s.group_context) / sizeof(int32_t));
    // int32_t* group_output = (int32_t*)execute_nn(ipsum, model->group);
    // print_vect("Group output", group_output, 6);
    // memcpy(ipsum->s.groups[ipsum->s.current_group], group_output, sizeof(ipsum->s.groups[0]));
    // pprintf("Group output: ");
    // for (int i = 0; i < 6; i++) {
    //     pprintf("%d ", group_output[i]);
    // }
    // for (int i = 0; i < 12; i++) {
    //     pprintf("%d ", ipsum->s.group_context[i]);
    // }
    // pprintf("\n");

    //pprintf("Group %d %d %d %d %d %d\n", group_output[0], group_output[1], group_output[2], group_output[3], group_output[4], group_output[5]);

    // Concatenate previous groups and current group, and execute the head NN layer
    int32_t* group0 = ipsum->s.groups[(ipsum->s.current_group + (12 - 8)) % 12];
    int32_t* group1 = ipsum->s.groups[(ipsum->s.current_group + (12 - 4)) % 12];
    int32_t* group2 = ipsum->s.groups[ipsum->s.current_group];
    memcpy(&vect[0], group0, sizeof(ipsum->s.groups[0]));
    memcpy(&vect[6], group1, sizeof(ipsum->s.groups[0]));
    memcpy(&vect[12], group2, sizeof(ipsum->s.groups[0]));
    print_vect("Head input", vect, 18);
    int32_t* prob = (int32_t*)execute_nn(ipsum, model->head);
    print_vect("Head output", prob, 23);

    // Prevent unnecessary spaces (no repeating spaces, spaces near the end, at the beginning or after repeating words)
    if (prob[0] > 0 && (remaining_characters < 3 || ipsum->s.last_word_hash == ipsum->s.current_word_hash || ipsum->s.current_word_hash == 0xFFFFFFFF)) {
        prob[0] = 0;
    }

    for (int i = 0; i < model->letters_count; i++) {
        pprintf("%s %d\n", model->lower_letters[i], prob[i]);
    }

    // Get random letter index based on probabilities
    return random_from_probs(ipsum, prob, model->letters_count);
}

static void update_context(LoremIpsum* ipsum, int32_t letter_index)
{
    int32_t* vect = (int32_t*)ipsum->t.vector_buffer0;

    //printf("(%s)", ipsum->t.model->lower_letters[letter_index]);
    // Update word hashes
    if (letter_index == 0) {
        ipsum->s.last_word_hash = ipsum->s.current_word_hash;
        ipsum->s.current_word_hash = 0xFFFFFFFF;
    } else {
        ipsum->s.current_word_hash = (ipsum->s.current_word_hash << 6) | (letter_index & 0x3F);
    }
    // Shift in the new letter index into the group context
    memmove(&ipsum->s.group_context[0], &ipsum->s.group_context[3], sizeof(int32_t) * 9);
    uint8_t* embedding = &ipsum->t.model->letters_embedding[3 * letter_index];
    ipsum->s.group_context[9] = embedding[0];
    ipsum->s.group_context[10] = embedding[1];
    ipsum->s.group_context[11] = embedding[2];
    // Make next group active
    ipsum->s.current_group = (ipsum->s.current_group + 1) % 12;

    // Execute group NN with current group context
    memcpy(vect, ipsum->s.group_context, sizeof(ipsum->s.group_context));
    //memset(ipsum->t.vector_buffer0, 0, sizeof(ipsum->t.vector_buffer0));
    print_vect("Group input", vect, sizeof(ipsum->s.group_context) / sizeof(int32_t));
    int32_t* group_output = (int32_t*)execute_nn(ipsum, ipsum->t.model->group);
    print_vect("Group output", group_output, 6);
    memcpy(ipsum->s.groups[ipsum->s.current_group], group_output, sizeof(ipsum->s.groups[0]));
}

void lorem_ipsum_set_heat(LoremIpsum* ipsum, uint32_t heat_percent)
{
    // Fixed-point representation for inverse heat is u3.4 (7 bits total)
    ipsum->s.inv_heat_u3_4 = heat_percent > 0 ? 1600 / heat_percent : 1600;
    if (ipsum->s.inv_heat_u3_4 > 127) ipsum->s.inv_heat_u3_4 = 127;
    if (ipsum->s.inv_heat_u3_4 < 1) ipsum->s.inv_heat_u3_4 = 1;
}

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


const char* lorem_ipsum_next(LoremIpsum* ipsum, size_t remaining_characters)
{
    if (remaining_characters < 1) {
        ipsum->s.flags |= FLAG_GENERATE_SPACE | FLAG_GENERATE_UPPER;
        ipsum->s.words_since_dot = 0;
        ipsum->s.words_since_comma = 0;
        return dot_string;
    } else if (ipsum->s.flags & FLAG_GENERATE_SPACE) {
        ipsum->s.flags &= ~FLAG_GENERATE_SPACE;
        return ipsum->t.model->lower_letters[0];
    }

    int32_t letter_index = generate_only(ipsum, remaining_characters - 1);
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
            int32_t rand = rand_lcg(ipsum) & 0xFF;
            if (rand < prob) {
                ipsum->s.flags |= FLAG_GENERATE_SPACE | FLAG_GENERATE_UPPER;
                ipsum->s.words_since_dot = 0;
                ipsum->s.words_since_comma = 0;
                return dot_string;
            }
            prob = get_punctuation_prob(ipsum->t.model->prob_comma, ipsum->s.words_since_dot, ipsum->s.words_since_comma);
            rand = rand_lcg(ipsum) & 0xFF;
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


static int32_t find_letter_index(const char* const* letters, uint32_t letters_count, const char** text)
{
    int i;
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


void lorem_ipsum_set_context(LoremIpsum* ipsum, char* context_text)
{
    const LoremIpsumModel* model = ipsum->t.model;

    reset_context(ipsum);

    if (context_text == NULL || context_text[0] == '\0') {
        return; // Just reset the generator state
    }

    while (*context_text) {
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
        } else if (*context_text == ' ') {
            if (!(ipsum->s.flags & FLAG_GENERATE_SPACE)) {
                ipsum->s.words_since_dot++;
                ipsum->s.words_since_comma++;
                update_context(ipsum, 0);
            } else {
                ipsum->s.flags &= ~FLAG_GENERATE_SPACE;
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
        buffer[0] = next_char[0];
        if (next_char[1]) {
            buffer[1] = next_char[1];
            if (next_char[2]) {
                buffer[2] = next_char[2];
                if (next_char[3]) {
                    buffer[3] = next_char[3];
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
    return buffer - initial_buffer;
}

#if 1

#include <stdio.h>

/** Functions dumps to stdout content of a buffer bytes as a hex.
*/
void dump_hex(const void* vbuffer, int length)
{
    const uint8_t* buffer = vbuffer;
    for (int i = 0; i < length; i++) {
        if (i % 16 == 0) {
            if (i > 0) {
                printf("\n");
            }
            printf("%04x: ", i);
        }
        printf("%02x ", buffer[i]);
    }
    printf("\n");
}

void test() {
    int32_t gen;
    LoremIpsum ipsum;
    lorem_ipsum_init(&ipsum, "pl", 50, 3, 0);

    char buffer[100] = {0};
    char buffer2[100] = {0};
    char buffer3[100] = {0};
    size_t count = lorem_ipsum_generate(&ipsum, buffer, sizeof(buffer));
    // for (int i = 0; i < 30; i++) {
    //     strcat(buffer, lorem_ipsum_next(&ipsum, 1000));
    // }
    // dump_hex(&ipsum.s, sizeof(ipsum.s));
    // uint32_t seed = ipsum.s.rand_state;
    // for (int i = 0; i < 30; i++) {
    //     strcat(buffer2, lorem_ipsum_next(&ipsum, 1000));
    // }
    // lorem_ipsum_set_context(&ipsum, buffer);
    // lorem_ipsum_set_seed(&ipsum, seed);
    // dump_hex(&ipsum.s, sizeof(ipsum.s));
    // for (int i = 0; i < 30; i++) {
    //     strcat(buffer3, lorem_ipsum_next(&ipsum, 1000));
    // }
    printf("|%s| %d\n", buffer, count);

    // int32_t* in = (int32_t*)ipsum.vector_buffer0;
    // in[0] = 0; in[1] = 255; in[2] = 0;
    // in[3] = 0; in[4] = 255; in[5] = 0;
    // in[6] = 0; in[7] = 255; in[8] = 0;
    // in[9] = 147; in[10] = 21; in[11] = 0;
    // int32_t* out = execute_nn(&ipsum, ipsum.model->group);
    // printf("Group output: ");
    // for (int i = 0; i < 6; i++) {
    //     printf("%d ", out[i]);
    // }
    // printf("\n");

    // int32_t gen = generate_only(&ipsum, 100);
    // int32_t* prob = (int32_t*)ipsum.vector_buffer0;
    // for (unsigned i = 0; i < ipsum.model->letters_count; i++) {
    //     printf("%s: %d\n", ipsum.model->lower_letters[i], prob[i]);
    // }
    // printf("%d [%s]\n", gen, ipsum.model->lower_letters[gen]);
    // while (remaining_characters >= 0) {
    //     //gen = i < 12 ? 0 : generate_only(&ipsum, 100);
    //     //gen = generate_only(&ipsum, 100);
    //     //update_context(&ipsum, gen);

    //     printf("%s", lorem_ipsum_next(&ipsum, remaining_characters + 100));

    //     // printf("Group context: ");
    //     // for (int j = 0; j < 4; j++) {
    //     //     //printf("%s ", ipsum.model->lower_letters[ipsum.group_context[j]]);
    //     // }
    //     // printf("\n");

    //     remaining_characters--;
    // }
    // printf("\n%d\n", ipsum.s.rand_state);
}

#endif
