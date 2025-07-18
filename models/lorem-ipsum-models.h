#include "lorem-ipsum-int.h"

#if defined(LOREM_IPSUM_LA_ENABLED) || defined(LOREM_IPSUM_EN_ENABLED) || defined(LOREM_IPSUM_PL_ENABLED)
#define LOREM_IPSUM_LA_DISABLED
#define LOREM_IPSUM_EN_DISABLED
#define LOREM_IPSUM_PL_DISABLED
#endif

#if defined(LOREM_IPSUM_LA_ENABLED) || !defined(LOREM_IPSUM_LA_DISABLED)
#define LOREM_IPSUM_LA_INSTANCE &lorem_ipsum_la,
#define _LOREM_IPSUM_LA_AS 22
#define _LOREM_IPSUM_LA_CN 1
extern const struct LoremIpsumModel lorem_ipsum_la;
#else
#define LOREM_IPSUM_LA_INSTANCE
#define _LOREM_IPSUM_LA_AS 0
#define _LOREM_IPSUM_LA_CN 0
#endif

#if defined(LOREM_IPSUM_EN_ENABLED) || !defined(LOREM_IPSUM_EN_DISABLED)
#define LOREM_IPSUM_EN_INSTANCE &lorem_ipsum_en,
#define _LOREM_IPSUM_EN_AS (27 > (_LOREM_IPSUM_LA_AS) ? 27 : (_LOREM_IPSUM_LA_AS))
#define _LOREM_IPSUM_EN_CN _LOREM_IPSUM_LA_CN + 1
extern const struct LoremIpsumModel lorem_ipsum_en;
#else
#define LOREM_IPSUM_EN_INSTANCE
#define _LOREM_IPSUM_EN_AS _LOREM_IPSUM_LA_AS
#define _LOREM_IPSUM_EN_CN _LOREM_IPSUM_LA_CN
#endif

#if defined(LOREM_IPSUM_PL_ENABLED) || !defined(LOREM_IPSUM_PL_DISABLED)
#define LOREM_IPSUM_PL_INSTANCE &lorem_ipsum_pl,
#define _LOREM_IPSUM_PL_AS (33 > (_LOREM_IPSUM_EN_AS) ? 33 : (_LOREM_IPSUM_EN_AS))
#define _LOREM_IPSUM_PL_CN _LOREM_IPSUM_EN_CN + 1
extern const struct LoremIpsumModel lorem_ipsum_pl;
#else
#define LOREM_IPSUM_PL_INSTANCE
#define _LOREM_IPSUM_PL_AS _LOREM_IPSUM_EN_AS
#define _LOREM_IPSUM_PL_CN _LOREM_IPSUM_EN_CN
#endif

#define LOREM_IPSUM_MODELS LOREM_IPSUM_LA_INSTANCE LOREM_IPSUM_EN_INSTANCE LOREM_IPSUM_PL_INSTANCE
#define LOREM_IPSUM_ALPHABET_MAX_SIZE _LOREM_IPSUM_PL_AS
#define LOREM_IPSUM_MODELS_COUNT (_LOREM_IPSUM_PL_CN)
