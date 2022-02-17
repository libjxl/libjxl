//#include "clear_generator.h"
// define this when using a custom generator
//#define CUSTOM_GENERATOR

#if defined(__cplusplus) || defined(c_plusplus)
/** Define as Exported to C (only when language is set to c++ otherwise not encapsulated by extern "C") (currently C++)
 * @param ... the body/code of the extern export
*/
#define EXTERN_C(...) \
    extern "C" \
    { \
        __VA_ARGS__ \
    }
#else
/** Define as Exporten to C (only when language is set to c++ otherwise not encapsulated by extern "C") (Not currently C++)
 * @param ... the body/code of the extern export
*/
#define EXTERN_C(...) \
    __VA_ARGS__
#endif

/// define as private
#define Private private:
/// define as public
#define Public public:
/// define a (method, constructor) as exported (only for JXL library)
#define Export JXL_EXPORT
/// define a (method, constructor, type) as deprecated
#define DEPRECATED JXL_DEPRECATED
/// define a (method, constructor) as exported (only for JXL_THREADS library)
#define Threads_Export JXL_THREADS_EXPORT
/// define a (method, constructor) as static
#define Inline inline
/// define a (member, method, constructor) as static
#define Static static


/** Define a Delegate
 * @param return_valuetype
 * @param name
 * @param args
*/
#define Delegate(return_valuetype, name, args) \
    typedef return_valuetype (*name) args;

/** Define a struct
 * @param name      struct name
 * @param ...       body/code
*/
#define Struct(name, ...) \
    struct name \
    { \
        __VA_ARGS__ \
    };

/** Define a struct (using typedef, prevents the need for 'struct name')
 * @param name      struct name
 * @param ...       body/code
*/
#define StructDef(name, ...) \
    typedef struct \
    { \
        __VA_ARGS__ \
    } name;

/** Define a struct (using typedef with a specified second name, prevents the need for 'struct name')
 * @param typedef_name  the defined type name (then one you dont need 'struct typename' for)
 * @param struct_name   struct name
 * @param ...           body/code
*/
#define StructDef2(typedef_name, struct_name, ...) \
    typedef struct struct_name \
    { \
        __VA_ARGS__ \
    } typedef_name;

/** Define an inline struct
 * @param ...      body/code
*/
#define InlineStruct(...) \
    struct \
    { \
        __VA_ARGS__ \
    };

/** Define an inline struct
 * @param name      struct name
 * @param ...       body/code
*/
#define NamedInlineStruct(name, ...) \
    struct \
    { \
        __VA_ARGS__ \
    } name;

/** Define a type definition/alias
 * @param name      definition/alias name
 * @param ...       definition/alias type/value
*/
#define Type(name, ...) \
    using name = __VA_ARGS__;

/** Define a type definition/alias (using typedef, not recommended)
 * @param name      definition/alias name
 * @param ...       definition/alias type/value
*/
#define TypeDef(name, ...) \
    typedef __VA_ARGS__ name;

/** Define a type Member with value/initializer
 * @param valuetype     member value type
 * @param name          member name
 * @param ...           member value/initializer
*/
#define MemberWithValue(valuetype, name, ...) \
    valuetype name = __VA_ARGS__;


/** Define a type Member
 * @param valuetype     member value type
 * @param name          member name
*/
#define Member(valuetype, name) \
    valuetype name;

/** Define a fixed array member
 * @param valuetype     member value type of fixed array
 * @param name          member name
 * @param size          member fixed array size
*/
#define FixedArray(valuetype, name, size) \
    valuetype name[size];

/** Define a Function/Method
 * @param return_valuetype      return type
 * @param name                  name
 * @param args                  args (must contain parenthesis)
*/
#define Method(return_valuetype, name, args) \
    return_valuetype name args;

/** Define a Function/Method with a global body! (make sure you know what you're doing)
 * 
 * @param return_valuetype      return type
 * @param name                  name
 * @param args                  args (must contain parenthesis)
 * @param ...                   code/body
 */
#define BodyMethod(return_valuetype, name, args, ...) \
    return_valuetype name args \
    { \
        __VA_ARGS__ \
    }

/** Defines an enum type
 * @param name      enum type name
 * @param ...       enum type body / code
*/
#define Enum(name, ...) \
    enum name \
    { \
        __VA_ARGS__ \
    };

/** Defines an enum type (using typedef, prevents the need for 'enum name')
 * @param name      enum type name
 * @param ...       enum type body / code
*/
#define EnumDef(name, ...) \
    typedef enum \
    { \
        __VA_ARGS__ \
    } name;

/** Defines an enum value
 * @param name      enum value
*/
#define Value(name) \
    name,
/** Defines an enum value with the specif   ied value
 * @param name      enum value name
 * @param ...       enum values, value
*/
#define DefinedValue(name, ...) \
    name = __VA_ARGS__,

/// unsafe code (used to allow more complex behaivour whilst preserving Metadata processing functionality)
#define RawCode(...) __VA_ARGS__ 
