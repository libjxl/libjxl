// DOCUMENT GENERATOR
// TO BE USED ONLY DURING DOXY BUILD PROCESS!

#include "type_generator.h"

#define DOC_GENERATOR

#define ESCAPE(code...) code

// preserve C interface when possible
// there files are supposed to be used as actual include files
// not just doxy files!


// escape a few things
#define ENDIF #endif
#define IF #if

// its stupid i know, but we shouldn't be using it for doc generation anyways
// we need to undefine these otherwise, they will get tokenized and replaced with their matching values.
#undef __cplusplus
#undef c_plusplus

#undef EXTERN_C
#define EXTERN_C(code...) \
    IF defined(__cplusplus) || defined(c_plusplus) \
        extern "C" { \
    ENDIF \
    code \
    IF defined(__cplusplus) || defined(c_plusplus) \
        } \
    ENDIF \
    /**/

