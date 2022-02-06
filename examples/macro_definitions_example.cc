#include <iostream>
// include the actual definitions
#include "jxl/types.h"

// now override the builder, so that we can peek the general structure
// note the use of [[maybe_unused]] is not escapeable for these sorts of senario's
// however if there is a possibility to not use [[maybe_unused]] it is recommended.


// load custom generator and define the CUSTOM_GENERATOR PP
#define CUSTOM_GENERATOR
#include "macro_definitions_example.h"

int main(int argc, char ** argv)
{
    // now when we include our type definitions, we will use our Custom Generator to create tangable code.
    //  in this instance to output the basic layout of the file
    #include "jxl/types.h"
    return 0;
}

// make sure to clean up,
//   not strictly necissary but good practice
#undef CUSTOM_GENERATOR
#include "typebuilder/clear_generator.h"
