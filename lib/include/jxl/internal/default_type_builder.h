#if !defined(OVERRIDE_TYPE_BUILDER)

#define Enum(type, code) \
    typedef enum \
    { \
      code \
    } type;

#define EnumValue(name) name,
#define EnumDefinedValue(name, value) name = value,


#define Struct(type, code) \
    typedef struct \
    {   \
        code \
    } type;

#define Class(type, code) \
    typdef class \
    { \
        code \
    } type;



#define Member(valuetype, member) \
    valuetype member;

#define Property(property) \
    decltype(property) get##property(); \
    void set##property(decltype(property) value);

#define PropertyNamed(property, name) \
    decltype(property) get##name(); \
    void set##name(decltype(property) value);

#define Public() public:
#define Private() private:

#endif
