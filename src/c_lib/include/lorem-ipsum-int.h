#ifndef _LOREM_IPSUM_INT_H
#define _LOREM_IPSUM_INT_H

#include <stdint.h>

enum LoremIpsumLayerType {
    LOREM_IPSUM_LAYER_LINEAR,
    LOREM_IPSUM_LAYER_RELU,
    LOREM_IPSUM_LAYER_SCALED_SOFTMAX,
};

typedef struct LoremIpsumUnknownLayer {
    uint32_t type;
} LoremIpsumUnknownLayer;

typedef struct LoremIpsumLinear {
    uint32_t type;
    const int8_t* weight;
    const int32_t* bias;
    const uint8_t* input_shift;
    const int32_t (*input_clamp)[2];
    uint32_t input_size;
    uint32_t output_size;
} LoremIpsumLinear;

typedef struct LoremIpsumReLU {
    uint32_t type;
    uint32_t input_size;
} LoremIpsumReLU;

typedef struct LoremIpsumScaledSoftmax {
    uint32_t type;
    const int32_t* weight;
    uint32_t frac_bits;
    uint32_t input_size;
} LoremIpsumScaledSoftmax;

typedef struct LoremIpsumModel {
    const char* lang;
    const char* name;
    const char* const* lower_letters;
    const char* const* upper_letters;
    uint32_t letters_count;
    const uint8_t* letters_embedding;
    uint32_t letter_embedding_length;
    const void* const* group;
    const void* const* head;
    const uint8_t* prob_dot;
    const uint8_t* prob_comma;
} LoremIpsumModel;

#endif // _LOREM_IPSUM_INT_H
