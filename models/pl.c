#include <stdint.h>
#include "lorem-ipsum-int.h"

static const char* const model_pl_lower_letters[] = {
    " ", "a", "\xc4\x85", "b", "c", "\xc4\x87", "d", "e", "\xc4\x99", "f", "g", "h", "i", "j", "k", "l", "\xc5\x82", "m", "n", "\xc5\x84", "o", "\xc3\xb3", "p", "r", "s", "\xc5\x9b", "t", "u", "w", "y", "z", "\xc5\xba", "\xc5\xbc",
};

static const char* const model_pl_upper_letters[] = {
    " ", "A", "\xc4\x84", "B", "C", "\xc4\x86", "D", "E", "\xc4\x98", "F", "G", "H", "I", "J", "K", "L", "\xc5\x81", "M", "N", "\xc5\x83", "O", "\xc3\x93", "P", "R", "S", "\xc5\x9a", "T", "U", "W", "Y", "Z", "\xc5\xb9", "\xc5\xbb",
};

static const uint8_t model_pl_letters_embedding[] = {
    0, 0, 0,
    167, 0, 34,
    183, 22, 37,
    0, 96, 208,
    0, 255, 0,
    255, 0, 30,
    0, 166, 106,
    145, 0, 0,
    207, 2, 30,
    0, 40, 191,
    0, 120, 166,
    0, 0, 101,
    218, 0, 60,
    196, 15, 138,
    0, 0, 184,
    154, 5, 191,
    0, 248, 208,
    0, 0, 197,
    0, 0, 255,
    203, 29, 33,
    109, 0, 45,
    134, 0, 0,
    0, 20, 158,
    0, 20, 101,
    0, 146, 0,
    158, 32, 89,
    0, 167, 210,
    147, 18, 44,
    39, 22, 198,
    180, 0, 44,
    100, 0, 154,
    125, 86, 76,
    129, 137, 154,
};

static const int8_t model_pl_group_0_linear_weight[] = {
    -2, 3, -1, -127, -43, -47, -48, 66, -4, -38, 39, -37,
    12, 15, 10, 8, 9, 8, -1, 4, 2, -59, 11, -127,
    0, -1, -1, -5, -1, -2, -99, -36, -127, 38, 21, 9,
    1, 2, -1, -4, 2, -2, 6, 29, 23, 58, -46, 127,
    -4, 1, -4, -4, -2, -2, 4, 0, -6, -6, 48, -127,
    5, 5, 10, -3, -11, -21, 34, 55, 39, -127, -101, 114,
    4, 4, 2, 0, 3, 3, 10, -127, -30, 14, -62, -7,
    1, 6, -3, 3, 6, -4, 13, 0, -16, 127, 34, -46,
    17, 1, 10, 23, 17, 9, -21, 45, 45, -24, -13, -127,
    6, 5, 5, 10, 6, 6, -14, 10, 6, 50, 46, 127,
    -17, -3, -9, -28, 6, -6, -16, 73, 114, -86, 17, 127,
    -10, -1, -24, 11, 5, -20, 127, 71, 61, -20, -43, -55,
    5, 8, -2, 9, 8, 11, 7, -127, 50, 22, 14, 80,
    -2, 4, -2, -8, 6, 0, 17, -2, -122, -27, -127, 47,
    -5, 0, -4, 0, 0, 1, -34, -78, -18, -127, 2, 8,
    28, 38, -12, -123, -6, 99, 46, -45, 19, 16, 36, -127,
};

static const int32_t model_pl_group_0_linear_bias[] = {
    -8139, 2655, -1054, -13951, 5015, -1650, 357, -12584, 3517, -16228, -24868, -18078, -14143, 2624, 3541, -26802,
};

static const uint8_t model_pl_group_0_linear_input_shift[] = {
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

static const int32_t model_pl_group_0_linear_input_clamp[][2] = {
    { -253993, 254120 },
    { -253993, 254120 },
    { -253993, 254120 },
    { -509991, 510246 },
    { -509991, 510246 },
    { -509991, 510246 },
    { -509991, 510246 },
    { -509991, 510246 },
    { -509991, 510246 },
    { -509991, 510246 },
    { -509991, 510246 },
    { -509991, 510246 },
};

static const struct LoremIpsumLinear model_pl_group_0_linear = {
    .type = LOREM_IPSUM_LAYER_LINEAR,
    .weight = model_pl_group_0_linear_weight,
    .bias = model_pl_group_0_linear_bias,
    .input_shift = model_pl_group_0_linear_input_shift,
    .input_clamp = model_pl_group_0_linear_input_clamp,
    .input_size = 12,
    .output_size = 16,
};

static const struct LoremIpsumReLU model_pl_group_1_relu = {
    .type = LOREM_IPSUM_LAYER_RELU,
    .input_size = 16,
};

static const int8_t model_pl_group_2_linear_weight[] = {
    51, 42, 16, 20, -127, -103, 5, 72, 16, -3, -84, 34, 25, -111, -16, -3,
    -77, -8, 127, 7, 34, -8, 4, 38, 30, 92, -21, 20, 33, -77, -62, -20,
    -14, -84, -64, -13, -15, 80, -38, 17, 9, 120, -36, -16, 101, 78, 127, 68,
    35, -127, -3, 16, -3, -3, 67, -8, 69, -8, -66, 4, 64, 19, -81, 41,
    68, 17, 43, -63, 127, -2, -46, -85, 15, 10, 13, -2, -23, 24, -67, 25,
    31, 88, 18, 62, -9, 3, -127, -4, -27, 13, -31, -72, 32, -58, 67, -47,
};

static const int32_t model_pl_group_2_linear_bias[] = {
    -47905, -535760, -430, -57917, 81144, -455162,
};

static const uint8_t model_pl_group_2_linear_input_shift[] = {
    0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 2, 1, 0, 1,
};

static const int32_t model_pl_group_2_linear_input_clamp[][2] = {
    { -1206795, 1221036 },
    { -988418, 1000082 },
    { -739703, 748432 },
    { -2575200, 2605589 },
    { -1516358, 1534252 },
    { -2113275, 2138213 },
    { -570558, 577291 },
    { -1886082, 1908339 },
    { -1349586, 1365512 },
    { -1174935, 1188800 },
    { -1899898, 1922318 },
    { -1590845, 1609618 },
    { -479717, 485378 },
    { -800634, 810082 },
    { -500901, 506812 },
    { -942403, 953524 },
};

static const struct LoremIpsumLinear model_pl_group_2_linear = {
    .type = LOREM_IPSUM_LAYER_LINEAR,
    .weight = model_pl_group_2_linear_weight,
    .bias = model_pl_group_2_linear_bias,
    .input_shift = model_pl_group_2_linear_input_shift,
    .input_clamp = model_pl_group_2_linear_input_clamp,
    .input_size = 16,
    .output_size = 6,
};

static const struct LoremIpsumReLU model_pl_group_3_relu = {
    .type = LOREM_IPSUM_LAYER_RELU,
    .input_size = 6,
};

static const void* const model_pl_group[] = {
    &model_pl_group_0_linear,
    &model_pl_group_1_relu,
    &model_pl_group_2_linear,
    &model_pl_group_3_relu,
    (void*)0,
};

static const int8_t model_pl_head_0_linear_weight[] = {
    -108, -54, 100, -92, 24, -24, 38, 76, 50, -6, -45, 24, 23, 62, 91, 10, -127, 5,
    -20, -12, 22, -15, -4, -45, -4, 7, -22, 4, 10, 13, 25, -127, -40, 13, 20, -14,
    -18, 11, -3, -7, -7, 7, 7, 4, 3, 7, 6, 11, 7, 7, -8, -127, -5, 53,
    -36, 76, -38, -33, 15, 64, 62, -64, -46, 6, -11, -18, -29, 127, -22, 16, 34, -39,
    1, 13, 8, -10, 14, 18, 8, -5, -19, -6, -26, 9, -56, -59, 10, -91, -127, -44,
    -47, 9, -10, 6, -4, -5, 17, 30, 67, 19, -18, -15, 56, -18, -83, -127, 12, 19,
    -61, 20, -3, 11, -12, 5, 41, 20, 40, 6, 55, 48, -35, 46, 12, 127, -60, -57,
    -23, 6, -7, 9, -7, -61, 13, 1, 2, -4, -12, 6, 15, -46, -22, 12, -22, -127,
    10, 3, -8, -9, 5, 23, -11, -1, -6, 1, -1, 4, 18, -127, -85, 32, 7, -105,
    -85, -2, -80, 23, 10, -37, 58, 7, 73, 25, 2, 8, 36, -104, -126, -127, 26, 19,
    4, 7, -4, 3, 0, -9, 3, 5, 6, 1, 4, -1, -19, 26, -36, 5, -9, -127,
    30, 24, -65, -8, -10, 25, 27, 32, -39, 17, 0, -14, 14, -28, -4, 127, -88, 109,
    -24, -73, 29, -25, 6, 42, 38, 18, -21, -7, 8, 43, -24, 127, 13, 49, 57, 29,
    -18, 11, -1, 3, 3, -19, 0, 24, -26, 2, 4, 27, -69, 89, 6, -127, 5, 10,
    9, 6, 2, -2, -2, 29, -3, 3, -14, -3, -1, 2, -127, 13, -5, -3, -20, -23,
    -49, -24, -2, -17, -65, -50, 72, 74, 46, 84, 82, 73, 123, 12, 2, 81, -127, -67,
};

static const int32_t model_pl_head_0_linear_bias[] = {
    -65516039, -16830599, 2181998, -114615480, 14066325, -24348941, -8005836, 13897410, 2197230, 25352503, 12671722, -54063575, -95874025, -19418929, 8179049, -22527106,
};

static const uint8_t model_pl_head_0_linear_input_shift[] = {
    5, 4, 6, 3, 4, 5, 4, 3, 5, 2, 4, 3, 0, 0, 0, 0, 0, 0,
};

static const int32_t model_pl_head_0_linear_input_clamp[][2] = {
    { -22129, 83354 },
    { -34179, 128738 },
    { -19546, 73623 },
    { -45658, 171974 },
    { -74094, 279062 },
    { -14581, 54929 },
    { -44262, 166713 },
    { -68363, 257482 },
    { -39096, 147251 },
    { -91323, 343956 },
    { -74094, 279062 },
    { -58340, 219732 },
    { -708281, 2667512 },
    { -546948, 2059906 },
    { -1251186, 4712167 },
    { -365317, 1375852 },
    { -1185576, 4465066 },
    { -466758, 1757897 },
};

static const struct LoremIpsumLinear model_pl_head_0_linear = {
    .type = LOREM_IPSUM_LAYER_LINEAR,
    .weight = model_pl_head_0_linear_weight,
    .bias = model_pl_head_0_linear_bias,
    .input_shift = model_pl_head_0_linear_input_shift,
    .input_clamp = model_pl_head_0_linear_input_clamp,
    .input_size = 18,
    .output_size = 16,
};

static const struct LoremIpsumReLU model_pl_head_1_relu = {
    .type = LOREM_IPSUM_LAYER_RELU,
    .input_size = 16,
};

static const int8_t model_pl_head_2_linear_weight[] = {
    16, 64, -127, -118, -104, -16, 45, 21, -66, -57, -2, -67, -56, -55, -1, 14,
    19, -65, -13, -20, -26, -127, -37, -120, 33, -23, -54, -13, -2, -70, 67, 3,
    1, 21, -2, 26, -45, -83, -34, -127, -79, -26, -111, 1, -25, -50, 17, 14,
    -7, -125, -117, -26, -44, -72, -16, -1, -12, 49, 11, -14, -127, -47, 24, 7,
    -13, -75, -127, -66, -95, -24, -23, 4, -24, 8, -5, -15, -33, -73, -36, -3,
    -12, -13, -20, 11, -47, -6, -9, 0, 0, -17, -5, -3, -33, -127, -7, 1,
    -2, -65, -83, 0, -32, -51, -31, -6, 10, 4, 11, -32, -76, -127, 2, 0,
    -32, 13, -52, 4, -53, -49, -50, -127, -39, -60, -53, 12, -13, -6, -13, 8,
    -14, 18, -12, -20, -24, 8, -45, -127, -18, -89, -57, -3, -13, -45, -1, 0,
    7, -8, -49, -9, -19, -127, -12, 3, -6, 7, 6, -3, -25, -15, -17, -2,
    -36, -127, -118, -14, -55, 2, 7, -4, -5, -5, 7, -10, -52, -40, -27, -9,
    -79, 1, 11, -120, -68, -21, 13, -21, 7, -7, 11, -127, 16, -11, -44, -30,
    11, -17, 33, -127, -10, -52, 9, -22, -11, 10, -12, -8, 0, -100, -38, -54,
    -83, -20, -113, -19, -82, -17, 0, -1, -7, -4, 1, -12, 0, -127, -38, -11,
    -2, -8, -64, -127, -48, -7, -15, -5, 3, 2, 1, 5, -42, -12, -11, -9,
    -50, -127, -52, -35, 30, -59, -37, -13, 5, -39, 12, -18, -111, -21, -54, 3,
    -36, 9, -50, -127, -5, -63, -15, -8, 17, -49, -39, -49, -77, -45, -33, 5,
    -127, -107, -126, -29, -60, -98, 3, -7, -24, 28, 3, 5, -49, 28, -41, 6,
    -2, -52, -127, -67, -27, -45, -14, -12, 8, -4, -1, -11, -19, -47, -20, -8,
    -6, -12, -46, 18, -82, -35, -17, 10, -7, -50, 8, 1, -116, -127, -10, -3,
    -24, -87, -6, -58, 60, -52, -6, -127, 34, 3, -34, -17, -20, -24, 4, -21,
    1, -20, -32, -76, 11, -41, -26, -38, -3, -9, -127, -22, -1, -25, 1, -9,
    -22, -47, -32, 40, -60, -81, -4, -7, -6, 42, 2, -22, -127, -51, -3, -9,
    -19, -63, 8, -127, 39, -66, 14, -4, 19, -6, 0, -41, -33, 2, -28, -11,
    9, -120, -43, 4, -47, -52, -13, -1, -15, 28, 5, -27, -127, -120, -12, 1,
    -7, -127, -53, 27, -105, -59, 16, -5, -10, 16, 9, -37, -107, -119, 8, 20,
    29, 0, -87, -25, -22, -68, -5, 5, 0, 48, 5, -24, -115, -127, -37, -5,
    -9, -23, -47, -17, 3, -127, -17, -56, 3, -21, -14, -9, 0, -40, -19, -11,
    -127, -112, -22, 9, -56, -92, 16, -26, 5, 3, 2, -45, -94, 8, -39, -26,
    -4, 13, -68, 46, -111, -30, -76, -73, -25, -80, -127, -20, -24, -24, -28, -72,
    -127, -21, -47, -48, 14, -44, -46, -14, 18, 0, -23, -32, -9, 17, 16, -26,
    -19, -17, -127, 9, -4, 2, -25, -4, 8, -22, 2, -65, -46, -76, -11, -3,
    -26, -127, -37, -4, -34, -4, -15, -7, 10, -2, 4, -13, -42, -45, -30, -2,
};

static const int32_t model_pl_head_2_linear_bias[] = {
    3392329, 10933583, -12154785, -19950366, 5599656, -987716, -4622101, 19981306, 2381907, -11140414, -3959496, -22514732, 2893637, 2709010, -1114065, -1616722, 4068392, -1086354, 6367125, -9172598, 7934410, -7828306, -8061439, -4972499, -4043913, -22489019, -15967443, 861777, 7536304, 28419120, 10030992, -1788052, -7116781,
};

static const uint8_t model_pl_head_2_linear_input_shift[] = {
    8, 5, 6, 5, 6, 5, 7, 5, 5, 7, 4, 6, 6, 6, 4, 7,
};

static const int32_t model_pl_head_2_linear_input_clamp[][2] = {
    { -154016, 1187444 },
    { -223766, 1725193 },
    { -141206, 1088676 },
    { -191040, 1472879 },
    { -79591, 613657 },
    { -298315, 2299927 },
    { -138579, 1068425 },
    { -147837, 1139798 },
    { -151861, 1170822 },
    { -146592, 1130200 },
    { -230866, 1779924 },
    { -315427, 2431855 },
    { -265225, 2044807 },
    { -216465, 1668897 },
    { -103093, 794838 },
    { -317032, 2444236 },
};

static const struct LoremIpsumLinear model_pl_head_2_linear = {
    .type = LOREM_IPSUM_LAYER_LINEAR,
    .weight = model_pl_head_2_linear_weight,
    .bias = model_pl_head_2_linear_bias,
    .input_shift = model_pl_head_2_linear_input_shift,
    .input_clamp = model_pl_head_2_linear_input_clamp,
    .input_size = 16,
    .output_size = 33,
};

static const int32_t model_pl_head_3_scaled_softmax_weight[] = {
    769623, 546694, 736473, 1045238, 883931, 5391262, 934788, 798865, 976023, 2954919, 1557748, 1257375, 1056476, 1774869, 1300262, 829148, 897822, 812914, 1003286, 3730433, 628830, 1306495, 1126352, 1363981, 1157082, 1344351, 839896, 819094, 564395, 820052, 1069001, 5530952, 1778980,
};

static const struct LoremIpsumScaledSoftmax model_pl_head_3_scaled_softmax = {
    .type = LOREM_IPSUM_LAYER_SCALED_SOFTMAX,
    .weight = model_pl_head_3_scaled_softmax_weight,
    .frac_bits = 47,
    .input_size = 33,
};

static const void* const model_pl_head[] = {
    &model_pl_head_0_linear,
    &model_pl_head_1_relu,
    &model_pl_head_2_linear,
    &model_pl_head_3_scaled_softmax,
    (void*)0,
};

static const uint8_t model_pl_prob_dot[] = {
    0,
    13, 8,
    6, 14, 13,
    3, 11, 25, 17,
    3, 8, 17, 25, 21,
    3, 9, 16, 20, 25, 25,
    3, 7, 15, 25, 24, 31, 30,
    3, 8, 16, 24, 29, 34, 32, 32,
    3, 9, 17, 26, 31, 32, 30, 42, 35,
    3, 7, 20, 26, 31, 36, 38, 35, 35, 37,
    4, 9, 17, 26, 33, 41, 39, 43, 46, 33, 41,
    3, 10, 18, 26, 39, 40, 45, 46, 43, 48, 41, 45,
    3, 10, 19, 29, 36, 41, 43, 45, 39, 45, 48, 58, 49,
    3, 10, 16, 27, 38, 40, 47, 39, 41, 47, 42, 54, 38, 46,
    3, 9, 21, 28, 40, 40, 41, 43, 42, 43, 32, 50, 62, 32, 55,
    5, 12, 20, 29, 42, 43, 45, 47, 46, 46, 60, 59, 52, 45, 78, 55,
    3, 11, 20, 31, 37, 47, 50, 46, 51, 49, 41, 46, 50, 36, 62, 47, 60,
    5, 10, 19, 32, 41, 44, 44, 46, 41, 50, 38, 55, 59, 58, 79, 27, 22, 94,
    3, 8, 21, 34, 38, 40, 46, 41, 55, 45, 44, 62, 68, 62, 74, 86, 58, 56, 115,
    5, 10, 23, 36, 39, 42, 49, 57, 43, 57, 60, 42, 56, 81, 60, 73, 65, 46, 139, 178,
    5, 8, 23, 30, 38, 40, 50, 45, 63, 50, 55, 56, 43, 58, 82, 55, 33, 89, 90, 101,
    3, 9, 23, 33, 43, 43, 47, 60, 49, 55, 69, 51, 51, 64, 64, 65, 74, 61, 58, 170,
    5, 11, 20, 38, 40, 48, 45, 46, 52, 49, 55, 47, 54, 43, 82, 61, 59, 99, 72, 109,
    4, 13, 18, 37, 50, 57, 49, 50, 52, 49, 60, 45, 45, 50, 44, 71, 106, 106, 139, 212,
    2, 10, 22, 46, 41, 39, 43, 55, 57, 57, 52, 33, 33, 44, 80, 55, 36, 159, 145, 127,
    3, 8, 19, 38, 41, 63, 51, 60, 55, 42, 57, 52, 56, 62, 73, 88, 78, 84, 84, 170,
    6, 6, 23, 37, 52, 47, 55, 57, 47, 53, 46, 48, 60, 82, 69, 68, 145, 63, 69, 255,
    7, 17, 21, 33, 46, 41, 60, 43, 48, 41, 58, 54, 48, 89, 68, 80, 42, 127, 113, 191,
    7, 9, 29, 40, 41, 39, 57, 64, 62, 73, 30, 56, 86, 54, 32, 79, 90, 145, 170, 191,
    8, 4, 22, 37, 46, 50, 57, 77, 47, 69, 46, 54, 75, 40, 72, 66, 127, 72, 153, 255,
    3, 19, 30, 45, 70, 57, 70, 51, 61, 76, 33, 78, 54, 44, 53, 39, 36, 0, 127, 0,
    9, 13, 34, 49, 52, 41, 62, 65, 63, 59, 74, 57, 71, 91, 63, 84, 84, 113, 127, 255,
    9, 16, 25, 48, 80, 57, 67, 83, 60, 33, 87, 63, 69, 104, 67, 0, 84, 212, 127, 255,
    5, 24, 41, 30, 73, 59, 55, 53, 84, 84, 73, 99, 127, 63, 56, 76, 127, 127, 255, 127,
    7, 28, 27, 49, 52, 70, 49, 80, 67, 79, 48, 63, 63, 63, 46, 145, 36, 170, 255, 255,
    18, 33, 61, 84, 93, 88, 98, 94, 115, 97, 0, 79, 38, 84, 84, 191, 0, 127, 0, 255,
    0, 52, 46, 36, 116, 140, 92, 93, 119, 69, 92, 84, 101, 76, 63, 127, 0, 127, 84, 255,
    101, 63, 66, 113, 74, 135, 97, 97, 127, 113, 95, 182, 212, 170, 109, 170, 255, 255, 255, 255,
    84, 170, 72, 191, 127, 135, 145, 191, 157, 127, 101, 204, 127, 0, 0, 170, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

static const uint8_t model_pl_prob_comma[] = {
    17,
    19, 25,
    13, 24, 32,
    12, 19, 31, 38,
    10, 21, 29, 38, 42,
    10, 19, 26, 36, 38, 43,
    10, 17, 24, 32, 38, 48, 42,
    11, 16, 24, 31, 37, 39, 45, 40,
    11, 16, 25, 29, 34, 41, 45, 45, 40,
    10, 14, 24, 28, 34, 33, 41, 37, 41, 40,
    10, 16, 21, 29, 36, 35, 37, 43, 32, 35, 35,
    11, 18, 24, 27, 35, 37, 32, 38, 32, 33, 43, 39,
    11, 16, 21, 28, 32, 33, 35, 35, 38, 37, 37, 46, 40,
    10, 16, 21, 28, 33, 35, 42, 36, 42, 36, 36, 37, 39, 39,
    10, 17, 23, 27, 30, 37, 34, 36, 31, 39, 30, 41, 48, 24, 39,
    7, 15, 20, 26, 32, 34, 35, 37, 33, 34, 37, 26, 34, 52, 39, 38,
    12, 14, 22, 25, 32, 37, 28, 35, 36, 29, 36, 26, 24, 38, 29, 38, 41,
    12, 18, 20, 26, 33, 35, 24, 40, 39, 28, 40, 36, 37, 35, 49, 36, 25, 64,
    12, 13, 21, 24, 27, 35, 32, 31, 35, 29, 26, 35, 30, 18, 49, 28, 46, 54, 104,
    13, 12, 22, 30, 29, 31, 34, 34, 38, 33, 33, 30, 43, 44, 45, 34, 32, 56, 0, 255,
    7, 12, 19, 25, 31, 25, 27, 33, 21, 37, 33, 25, 20, 30, 51, 53, 30, 0, 84, 255,
    10, 14, 18, 26, 23, 34, 35, 24, 31, 40, 28, 36, 24, 32, 40, 88, 10, 92, 76, 255,
    12, 12, 25, 28, 23, 33, 37, 24, 40, 30, 32, 31, 22, 40, 19, 36, 19, 54, 101, 255,
    13, 18, 18, 23, 24, 37, 33, 41, 20, 20, 35, 29, 25, 20, 32, 38, 109, 0, 153, 255,
    7, 8, 20, 25, 27, 36, 26, 30, 33, 34, 32, 38, 25, 29, 33, 47, 63, 0, 0, 255,
    11, 13, 20, 19, 25, 33, 26, 30, 35, 19, 31, 21, 29, 44, 47, 44, 27, 20, 0, 255,
    8, 16, 19, 25, 27, 28, 26, 29, 23, 25, 22, 19, 42, 40, 20, 12, 0, 63, 127, 255,
    6, 10, 18, 22, 18, 23, 35, 47, 26, 21, 42, 15, 26, 29, 39, 16, 16, 0, 50, 255,
    6, 11, 12, 24, 25, 35, 19, 31, 11, 23, 38, 58, 34, 38, 12, 69, 56, 42, 0, 255,
    7, 8, 24, 23, 27, 33, 27, 30, 43, 39, 5, 40, 26, 23, 12, 0, 0, 50, 127, 255,
    10, 17, 22, 15, 30, 34, 16, 39, 5, 50, 15, 18, 34, 36, 50, 63, 63, 0, 0, 255,
    13, 9, 13, 16, 29, 26, 39, 29, 39, 30, 14, 20, 13, 15, 84, 63, 63, 50, 255, 255,
    6, 11, 22, 10, 30, 34, 5, 46, 14, 15, 12, 0, 0, 25, 22, 0, 0, 0, 0, 255,
    5, 3, 18, 13, 9, 19, 6, 29, 27, 11, 22, 69, 84, 20, 0, 0, 0, 0, 255, 255,
    15, 12, 17, 0, 8, 22, 17, 38, 11, 46, 14, 33, 0, 0, 27, 84, 0, 0, 255, 255,
    9, 38, 0, 6, 16, 7, 0, 0, 20, 38, 0, 22, 22, 0, 0, 127, 0, 0, 0, 255,
    50, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255,
    0, 56, 36, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 63, 0, 255, 255, 255, 255,
    0, 0, 0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

const struct LoremIpsumModel lorem_ipsum_pl = {
    .lang = "pl",
    .name = "Polish",
    .lower_letters = model_pl_lower_letters,
    .upper_letters = model_pl_upper_letters,
    .letters_count = 33,
    .letters_embedding = model_pl_letters_embedding,
    .letter_embedding_length = 3,
    .group = model_pl_group,
    .head = model_pl_head,
    .prob_dot = model_pl_prob_dot,
    .prob_comma = model_pl_prob_comma,
};
