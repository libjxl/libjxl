//#include "clear_generator.h"
// define this when using a custom generator
//#define CUSTOM_GENERATOR

#if defined(__cplusplus) || defined(c_plusplus)
/** Define as Exported to C (only when language is set to c++ otherwise not encapsulated by extern "C") (currently C++)
 * @param code the body/code of the extern export
*/
#define EXTERN_C(code...) extern "C" { code }
#else
/** Define as Exporten to C (only when language is set to c++ otherwise not encapsulated by extern "C") (Not currently C++)
 * @param code the body/code of the extern export
*/
#define EXTERN_C(code...) code
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


/** Define an inline unnamed union
 * @param body      body/code
*/
#define Union(body...) \
    union \
    { \
        body \
    };

/** Define an inline named union
 * @param name      union name
 * @param body      body/code
*/
#define UnionNamed(name, body...) \
    union name\
    { \
        body \
    };

/** Define a struct
 * @param name      struct name
 * @param body      body/code
*/
#define Struct(name, body...) \
    struct name \
    { \
        body \
    };

/** Define a struct (using typedef, prevents the need for 'struct name')
 * @param name      struct name
 * @param body      body/code
*/
#define StructDef(name, body...) \
    typedef struct \
    { \
        body \
    } name;

/** Define a struct (using typedef with a specified second name, prevents the need for 'struct name')
 * @param typedef_name  the defined type name (then one you dont need 'struct typename' for)
 * @param struct_name   struct name
 * @param body          body/code
*/
#define StructDef2(typedef_name, struct_name, body...) \
    typedef struct struct_name \
    { \
        body \
    } typedef_name;

/** Define an inline struct
 * @param body      body/code
*/
#define InlineStruct(body...) \
    struct \
    { \
        body \
    };
/** Define an inline struct
 * @param name      struct name
 * @param body      body/code
*/
#define NamedInlineStruct(name, body...) \
    struct \
    { \
        body \
    } name;

/** Define a class
 * @param name      class name
 * @param body      body/code
*/
#define Class(name, body...) \
    class name \
    { \
        body \
    };

/** Define a class (using typedef, prevents the need for 'class name')
 * @param name      class name
 * @param body      body/code
*/
#define ClassDef(name, body...) \
    typedef class \
    { \
        body \
    } name;

/** Define an class
 * @param body      body/code
*/
#define InlineClass(body...) \
    class \
    { \
        body \
    };

/** Define a type definition/alias
 * @param name      definition/alias name
 * @param value     definition/alias type/value
*/
#define Type(name, value...) \
    using name = value;

/** Define a type definition/alias (using typedef, not recommended)
 * @param name      definition/alias name
 * @param value     definition/alias type/value
*/
#define TypeDef(name, value...) \
    typedef value name;

/** Define a type Member with value/initializer
 * @param valuetype     member value type
 * @param name          member name
 * @param value         member value/initializer
*/
#define MemberWithValue(valuetype, name, value...) \
    valuetype name = value;


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
 * @param code                  code/body
*/
#define Method(return_valuetype, name, args, code...) \
    return_valuetype name args;

/** Define a Function/Method with a global body! (make sure you know what you're doing)
 * 
 * @param return_valuetype      return type
 * @param name                  name
 * @param args                  args (must contain parenthesis)
 * @param code                  code/body
 */
#define BodyMethod(return_valuetype, name, args, code...) \
    return_valuetype name args \
    { \
        code \
    }

/** Define type Constructor
 * @param name      type name
 * @param args      args (must contain parenthesis)
 * @param code      code/body
*/
#define Constructor(name, args, code...) \
    name args;

/** Define a Property
 * @param valuetype             value type
 * @param name                  name
 * @param getter_accessability  accessability modifier for getter
 * @param setter_accessability  accessability modifier for setter
*/
#define Property(valuetype, name, getter_accessability, setter_accessability) \
    private: valuetype name; \
    getter_accessability valuetype get##name() const; \
    setter_accessability void set##name(valuetype val);

/** Define a ReadOnlyProperty
 * @param valuetype             value type
 * @param name                  name
 * @param getter_accessability  accessability modifier for getter
*/
#define ReadOnlyProperty(valuetype, name, getter_accessability) \
    private: valuetype name; \
    getter_accessability valuetype get##name() const;

/** Defines an enum type
 * @param name      enum type name
 * @param body      enum type body / code
*/
#define Enum(name, body...) \
    enum name \
    { \
        body \
    };

/** Defines an enum type (using typedef, prevents the need for 'enum name')
 * @param name      enum type name
 * @param body      enum type body / code
*/
#define EnumDef(name, body...) \
    typedef enum \
    { \
        body \
    } name;

/** Defines an enum value
 * @param name      enum value
*/
#define Value(name) \
    name,
/** Defines an enum value with the specif   ied value
 * @param name      enum value name
 * @param value     enum values, value
*/
#define DefinedValue(name, value...) \
    name = value,

/** Define a Typed Enum
 * @param name      pesudo enum name
 * @param valuetype pesudo enum inner type
 * @param body      pesudo enum type body / code
*/
#define TypedEnum(name, valuetype, body...) \
    class name \
    { \
        using type [[maybe_unused]] = valuetype; \
        body \
    };
/** Define a Typed Enum (using typedef, prevents the need for 'class name')
 * @param name      pesudo enum name
 * @param valuetype pesudo enum inner type
 * @param body      pesudo enum type body / code
*/
#define TypedEnumDef(name, valuetype, body...) \
    typedef class \
    { \
        using type [[maybe_unused]] = valuetype; \
        body \
    } name;


/** Define a Typed Enum Value
 * @param name      pesudo enum value name
 * @param value     pesudo enum values, value
*/
#define TypedValue(name, value...) \
    private: const type _i##name = value; \
    public: const type& name = _i##name;

/// unsafe code (used to allow more complex behaivour whilst preserving Metadata processing functionality)
#define RawCode(code...) code 
