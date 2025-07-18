
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lorem-ipsum.h"


static const char usage_text[] = "\n"
    "\n"
    "USAGE: %s [TEXT_LENGTH] [LANG] [OPTIONS]\n"
    "\n"
    "Options:\n"
    "\n"
    "  [TEXT_LENGTH]       Specify the length of the generated text in characters\n"
    "                      (default: 200). The text is written in UTF-8, so for\n"
    "                      non-ASCII alphabet, number of bytes may be greater.\n"
    "\n"
    "  [LANG]              Specify language stylization (default: %s).\n"
    "\n"
    "  -h, --heat PERCENT  Generation heat (default: 60%%). Increase it for more\n"
    "                      random output, but it may be less coherent. Decrease it\n"
    "                      for less diverse output, but it may be more repetitive.\n"
    "\n"
    "  -s, --seed [SEED]   Set random number generator seed (default: current time\n"
    "                      and \"/dev/urandom\" if available). If SEED is not\n"
    "                      provided, the chosen random value will be displayed\n"
    "                      alongside the generated text.\n"
    "\n"
    "  -v, --version [NUM] Set expected model version (default: 0 for newest\n"
    "                      version). You can use it to get deterministic output\n"
    "                      even if the model changes in the future. Use this option\n"
    "                      without argument to see current version.\n"
    "\n"
    "  -p, --prefix [NUM]  Prefix the text with original Lorem Ipsum passage.\n"
    "                      The NUM is the number of words (default: 5).\n"
    "\n"
    "  -ps, --paragraph-separator SEPARATOR\n"
    "                      Enable paragraphs with the specified SEPARATOR.\n"
    "                      You can use escape sequences: \\\\ \\r \\n \\xNN\n"
    "\n"
    "  -pm, --paragraph-mean MEAN\n"
    "  -psv, --paragraph-shorter-variance VARIANCE\n"
    "  -plv, --paragraph-longer-variance VARIANCE\n"
    "                      Enable paragraphs with the specified parameters.\n"
    "                      Parameters are expressed in number of sentences.\n"
    "                      You can use fractions with precision of '0.1'.\n"
    "                      The probability of a paragraph length is modeled using\n"
    "                      normal distribution with the specified MEAN.\n"
    "                      The distribution is asymmetric with different VARIANCE\n"
    "                      for paragraphs shorter than the mean and paragraphs\n"
    "                      longer than the mean.\n"
    "\n"
    "  -l, --languages     List available language stylizations and exit.\n"
    "\n"
    "  -h, --help          Show this help message and exit.\n"
    "\n";

static const char lorem_ipsum_passage[] = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
    "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, "
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute "
    "irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim "
    "id est laborum.";


static void usage(const char* self, int error) {
    const char* name = strrchr(self, '/');
    FILE* out = error ? stderr : stdout;
    fprintf(out, usage_text, name ? name + 1 : self, lorem_ipsum_languages()[0]);
    if (error) {
        exit(error);
    }
}

uint32_t parse_cli_uint(const char* str, const char* fail_or_error_with_arg) {
    char* endptr = NULL;
    uint32_t value = (uint32_t)strtoul(str, &endptr, 10);
    while (*str == ' ') str++;
    if (endptr == NULL || (*endptr != '\0' && *endptr != '%' && *endptr != ' ') || endptr == str) {
        if (fail_or_error_with_arg) {
            fprintf(stderr, "Error: Invalid number '%s' after '%s'.\n", str, fail_or_error_with_arg);
            exit(12);
        }
        return 0xFFFFFFFF;
    }
    return value;
}

int main(int argc, char* argv[]) {
    int i, j;

    static LoremIpsum ipsum;
    const char *const * languages = lorem_ipsum_languages();
    uint32_t version = 0;
    bool initialize_seed = true;
    bool show_seed = false;
    uint32_t seed = 0;
    uint32_t heat = 60;
    uint32_t prefix = 0;
    uint32_t text_length = 200;
    const char* lang = languages[0];

    for (i = 1; i < argc; i++) {
        const char* arg = argv[i];
        if (arg[0] == '-') {
            bool has_arg = (i + 1 < argc) && (argv[i + 1][0] != '-');
            while (arg[0] == '-') {
                arg++;
            }
            if (strcmp(arg, "h") == 0 || strcmp(arg, "help") == 0 || strcmp(arg, "heat") == 0) {
                if (has_arg) {
                    heat = parse_cli_uint(argv[i + 1], argv[i]);
                    i++;
                } else {
                    usage(argv[0], 0);
                    return 0;
                }
            } else if (strcmp(arg, "s") == 0 || strcmp(arg, "seed") == 0) {
                if (has_arg) {
                    seed = parse_cli_uint(argv[i + 1], argv[i]);
                    initialize_seed = false;
                    i++;
                } else {
                    show_seed = true;
                }
            } else if (strcmp(arg, "v") == 0 || strcmp(arg, "version") == 0) {
                if (has_arg) {
                    version = parse_cli_uint(argv[i + 1], argv[i]);
                    i++;
                } else {
                    printf("Current implementation version: 1\n");
                    return 0;
                }
            } else if (strcmp(arg, "p") == 0 || strcmp(arg, "prefix") == 0) {
                if (has_arg) {
                    prefix = parse_cli_uint(argv[i + 1], argv[i]);
                    i++;
                } else {
                    prefix = 5;
                }
            } else if (strcmp(arg, "l") == 0 || strcmp(arg, "languages") == 0) {
                printf("Available languages:");
                for (j = 0; languages[j]; j++) {
                    printf(" %s", languages[j]);
                }
                printf("\n");
                return 0;
            } else if (strcmp(arg, "") == 0) {
                // Skip options separator "--"
            } else {
                fprintf(stderr, "Error: Unknown option '%s'.\n", argv[i]);
                usage(argv[0], 13);
            }
        } else if (strcmp(arg, "/?") == 0) {
            usage(argv[0], 0);
            return 0;
        } else {
            uint32_t try_text_length = parse_cli_uint(arg, NULL);
            if (try_text_length != 0xFFFFFFFF) {
                text_length = try_text_length;
            } else {
                lang = NULL;
                for (j = 0; languages[j]; j++) {
                    if (strcmp(arg, languages[j]) == 0) {
                        lang = arg;
                        break;
                    }
                }
                if (lang == NULL) {
                    fprintf(stderr, "Error: Unknown language '%s'.\n", arg);
                    fprintf(stderr, "Available languages:");
                    for (j = 0; languages[j]; j++) {
                        fprintf(stderr, " %s", languages[j]);
                    }
                    fprintf(stderr, "\n");
                    usage(argv[0], 11);
                }
            }
        }
    }

    if (initialize_seed) {
        srand(time(NULL));
        seed = rand() + time(NULL);
        FILE* urandom = fopen("/dev/urandom", "rb");
        if (urandom) {
            uint32_t urandom_seed;
            int x = fread(&urandom_seed, sizeof(urandom_seed), 1, urandom);
            fclose(urandom);
            seed += urandom_seed;
            seed += x;
        }
        if (show_seed) {
            printf("Chosen random seed: %u\n", seed);
        }
    }

    if (!lorem_ipsum_init(&ipsum, lang, heat, seed, version)) {
        fprintf(stderr, "Error: Failed to initialize Lorem Ipsum generator.\n");
        return 1;
    }

    static char buffer[sizeof(lorem_ipsum_passage)];

    if (prefix > 0) {
        memcpy(buffer, lorem_ipsum_passage, sizeof(lorem_ipsum_passage));
        char* end = buffer;
        while (prefix > 0 && *end != '\0') {
            if (*end == ' ') {
                prefix--;
            }
            end++;
        }
        *end = '\0';
        if (strlen(buffer) > text_length) {
            buffer[text_length] = '\0';
        }
        printf("%s", buffer);
        lorem_ipsum_set_context(&ipsum, buffer);
        text_length -= strlen(buffer);
    }

    while (text_length > 0) {
        i = 0;
        while (i < (int)sizeof(buffer) - 6 && text_length > 0) {
            text_length--;
            const char* letter = lorem_ipsum_next(&ipsum, text_length);
            int len = strlen(letter);
            memcpy(&buffer[i], letter, len);
            i += len;
        }
        buffer[i] = '\0';
        printf("%s", buffer);
    }

    printf("\n");

    return 0;
}

