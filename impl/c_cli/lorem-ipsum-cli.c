
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "lorem-ipsum.h"

//static LoremIpsum ipsum;

void test();

int main(int argc, char* argv[]) {
    const char *const * languages = lorem_ipsum_languages();
    printf("Languages supported:");
    while (*languages) {
        printf(" %s", *languages);
        languages++;
    }
    printf("\n");
    //lorem_ipsum_init(&ipsum, NULL, 50, rand(), 0);
    test();
    (void)argc;
    (void)argv;
    return 0;
}

