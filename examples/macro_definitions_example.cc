#include <iostream>
// include the actual definitions
#include "jxl/types.h"



// now override the builder, so that we can peek the general structure
// note the use of [[maybe_unused]] is not escapeable for these sorts of senario's
// however if there is a possibility to not use [[maybe_unused]] it is recommended.

#define OVERRIDE_TYPE_BUILDER
#define Enum(type, code) \
    { \
        using def_enum [[maybe_unused]] = type; \
        [[maybe_unused]] const char* accessibility = "private"; \
        std::cout << "BEGIN_Enum (" #type ")" << std::endl; \
        code \
        std::cout << "END_Enum (" #type ")" << std::endl; \
    }
#define EnumValue(name) std::cout << "    " #name << std::endl;
#define EnumDefinedValue(name, value) std::cout << "    " #name " = " #value << std::endl;

#define Struct(type, code) \
    {   \
        using def_type [[maybe_unused]] = type; \
        [[maybe_unused]] const char* accessibility = "private"; \
        std::cout << "BEGIN_Class (" #type ")" << std::endl; \
        code \
        std::cout << "END_Class (" #type ")" << std::endl; \
    }
#define Class(type, code) \
    {   \
        using def_type [[maybe_unused]] = type; \
        [[maybe_unused]] const char* accessibility = "private"; \
        std::cout << "BEGIN_Class (" #type ")" << std::endl; \
        code \
        std::cout << "END_Class (" #type ")" << std::endl; \
    }

#define Member(valuetype, member) \
    std::cout << "    Member: " << accessibility << " " #valuetype " " #member << std::endl; 
#define Property(property) \
    std::cout << "    Getter: " << accessibility << " " << typeid(decltype(def_type::get##property))::name() << " get" #property "() => " #property << std::endl; \
    std::cout << "    Setter: " << accessibility << " void set" #property "("  << typeid(decltype(def_type::set##property))::name() << " value) => " #property << std::endl;
#define PropertyNamed(property, name) \
    std::cout << "    Getter: " << accessibility << " " << typeid(decltype(def_type::get##name))::name() << " get" #name "() => " #property << std::endl; \
    std::cout << "    Setter: " << accessibility << " void set" #name "("  << typeid(decltype(def_type::set##name))::name() << " value) => " property << std::endl;

#define Public() accessibility = "public";
#define Private() accessibility = "private";

int main(int argc, char ** argv)
{
    std::cout << "Pulling jxl/types.h structure data" << std::endl;
    #include "jxl/internal/meta_types.h" // because jxl/types.h exports a C interface, we need to move the actual definitions into a seperate file
    return 0;
}

// make sure the unset our custom builder, so as to not interfer with other files
//    note: in this senario this does nothing, but when using this within a header file, be sure to include the bellow lines!
#undef OVERRIDE_TYPE_BUILDER
#include "jxl/internal/clear_type_builder.h"
