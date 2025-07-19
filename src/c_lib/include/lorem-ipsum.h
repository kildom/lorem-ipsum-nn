#ifndef _LOREM_IPSUM_H
#define _LOREM_IPSUM_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include "lorem-ipsum-int.h"
#include "lorem-ipsum-models.h"

#define LOREM_IPSUM_PARAGRAPHS_DISABLE (-0x7FFFFFFF)
#define LOREM_IPSUM_PARAGRAPHS_DEFAULT (-0x7FFFFFFE)


/** @brief Lorem Ipsum generator structure.
 *
 * Initialize it with `lorem_ipsum_init()` function. No deinitialization is needed.
 *
 * Contains all necessary data to generate text, no dynamic memory allocation is used.
 *
 * Structure fields are private and should not be accessed directly.
 */
typedef struct LoremIpsum
{
    struct {
        uint32_t rand_state;
        int32_t groups[12][6];
        int32_t current_group;
        uint32_t last_word_hash;
        uint32_t current_word_hash;
        int32_t group_context[4 * 3];
        int32_t inv_heat_u3_4;
        int32_t flags;
        int32_t words_since_dot;
        int32_t words_since_comma;
        int32_t sentences_in_paragraph;
        uint8_t paragraph_prob_table[64];
    } s;
    struct {
        const struct LoremIpsumModel* model;
        const char* paragraph_separator;
        int64_t vector_buffer0[LOREM_IPSUM_ALPHABET_MAX_SIZE > 16 ? LOREM_IPSUM_ALPHABET_MAX_SIZE : 16];
        int64_t vector_buffer1[LOREM_IPSUM_ALPHABET_MAX_SIZE > 16 ? LOREM_IPSUM_ALPHABET_MAX_SIZE : 16];
    } t;
} LoremIpsum;


/** @brief Returns the list of available language stylization.
 *
 * @return A list of strings. The list is ended with a NULL pointer.
 */
const char *const * lorem_ipsum_languages();


/** @brief Initializes the Lorem Ipsum generator.
 *
 * It does not use dynamic memory allocation, everything is contained in the LoremIpsum structure.
 * You do not need to uninitialize it.
 *
 * @param ipsum Pointer to the LoremIpsum structure.
 * @param language Stylize output to specific language. Pass NULL for default style which is Latin if enabled, or first available otherwise.
 * @param heat_percent Heat to apply to the generation (in percent). Normally it should be around 50%-60% for a good balance
 *        between randomness and coherence. Increase it for more random output, but it may look less coherent.
 *        Decreasing it may lead to less diversity in the generated text and repetitive output.
 * @param seed Initial 32-bit seed for the pseudo random number generator. You can use rand() function if you want a random seed.
 * @param ver Version of the model to use. With current implementation, pass 0 if you don't care about deterministic output
 *            across different versions. Pass 1 to ensure deterministic output even if the model changes in the future.
 * @return `false` if the language stylization or version is not supported, `true` otherwise.
 */
bool lorem_ipsum_init(LoremIpsum* ipsum, const char* language, uint32_t heat_percent, uint32_t seed, uint32_t ver);


/** @brief Generates a random text.
 *
 * Generates entire text in one call. At the beginning, it will reset the generator state
 * (see `lorem_ipsum_set_context(ipsum, NULL)`). It will end text with a period and null character.
 *
 * For non-ASCII alphabet, number of bytes written to the buffer may be smaller than the buffer size.
 * The function returns actually written bytes to the buffer, including the null character.
 *
 * For more flexibility, use `lorem_ipsum_next()` to generate text character by character.
 *
 * @param ipsum Pointer to the LoremIpsum structure.
 * @param buffer Pointer to the buffer where the generated text will be written.
 * @param buffer_size Size of the buffer in bytes.
 * @return Number of bytes written to the buffer, including the null character.
 */
size_t lorem_ipsum_generate(LoremIpsum* ipsum, char* buffer, size_t buffer_size);


/** @brief Generates the next character.
 *
 * @param ipsum Pointer to the LoremIpsum structure.
 * @param remaining_characters Number of characters left to generate after current one.
 *        This parameter is used to control coherent ending.
 *        If this is 0, it will generate a period to end current sentence.
 *        If it is very low, it will not end current word to avoid very short words at the end.
 *        If it is low, it will not end current sentence to avoid very short sentences at the end.
 * @return Pointer to the next character in the generated text.
 *         It will remain valid until the next generation call.
 *         For non-ASCII alphabet, it will return a null-terminated UTF-8 character.
 */
const char* lorem_ipsum_next(LoremIpsum* ipsum, size_t remaining_characters);


/** @brief Sets the context for the Lorem Ipsum generator.
 *
 * It will reset the generator before setting the context.
 * Characters outside defined alphabet will be ignored.
 *
 * You can use it to: reset state, continue generating text, or prefix generated text with
 * predefined text, e.g. "Lorem ipsum ".
 *
 * If your context is long, you can start with the last sentence of your context text.
 *
 * @param ipsum Pointer to the LoremIpsum structure.
 * @param context_text Pointer to the text that will be used as a new context.
 *                     It can be NULL, in which case it will just reset generator state.
 */
void lorem_ipsum_set_context(LoremIpsum* ipsum, const char* context_text);


/** @brief Sets the seed for the pseudo random number generator.
 *
 * @param ipsum Pointer to the LoremIpsum structure.
 * @param seed 32-bit seed for the pseudo random number generator.
 */
static inline void lorem_ipsum_set_seed(LoremIpsum* ipsum, uint32_t seed);


/** @brief Sets the heat for the Lorem Ipsum generator.
 *
 * @param ipsum Pointer to the LoremIpsum structure.
 * @param heat_percent Heat to apply to the generation (in percent).
 *        See `lorem_ipsum_init()` for more details.
 */
void lorem_ipsum_set_heat(LoremIpsum* ipsum, uint32_t heat_percent);


/** @brief Sets the paragraph generation parameters.
 *
 * The paragraphs are separated by a '\n' character.
 *
 * By default, paragraphs generation is disabled. This function enables it and sets parameters.
 * To disable it again, call this function with `mean` set to `LOREM_IPSUM_PARAGRAPHS_DISABLE`.
 * 
 * Paragraph length (in sentences) distribution is based on normal distribution.
 * The distribution is centered around `mean` value and it can be asymmetric.
 * For paragraphs shorter than `mean`, it uses `shorter_variance`.
 * For paragraphs longer than `mean`, it uses `longer_variance`.
 *
 * All parameters are in 1/10 of a sentence, so 10 is 1 sentence, 20 is 2 sentences, etc.
 *
 * @param ipsum Pointer to the LoremIpsum structure.
 * @param mean Mean length of paragraphs (in 1/10 of a sentence),
 *             or `LOREM_IPSUM_PARAGRAPHS_DISABLE` to disable paragraph generation,
 *             or `LOREM_IPSUM_PARAGRAPHS_DEFAULT` to use default values which are:
 *             50 for mean, 20 for shorter_variance, and 40 for longer_variance.
 *             If you want to use default values, pass `LOREM_IPSUM_PARAGRAPHS_DEFAULT`.
 * @param shorter_variance Variance for shorter paragraphs (in 1/10 of a sentence).
 * @param separator Paragraph separator string. It is not copied, so it must be valid
 *                  as long as paragraphs are generated. NULL for default separator
 *                  which is "\n" character.
 * @param longer_variance Variance for longer paragraphs (in 1/10 of a sentence).
 */
void lorem_ipsum_set_paragraphs(LoremIpsum* ipsum, int32_t mean, int32_t shorter_variance, int32_t longer_variance, const char* separator);


/** @brief Returns pointer to the internal state buffer.
 *
 * You can use it to save content of this buffer.
 * Later, you can restore it on initialized generator by writing back to that buffer.
 * After restoring, you can continue generating text from the same state.
 *
 * @param ipsum Pointer to the LoremIpsum structure.
 * @param size  Pointer to a variable that will receive the size of the state buffer,
 *              or NULL, if you don't need the size.
 * @return Pointer to the internal state buffer.
 */
static inline void* lorem_ipsum_get_state_buffer(LoremIpsum* ipsum, uint32_t* size);


static inline void lorem_ipsum_set_seed(LoremIpsum* ipsum, uint32_t seed)
{
    ipsum->s.rand_state = seed;
}


static inline void* lorem_ipsum_get_state_buffer(LoremIpsum* ipsum, uint32_t* size)
{
    if (size) *size = sizeof(ipsum->s);
    return (void*)&ipsum->s;
}


#endif // _LOREM_IPSUM_H
