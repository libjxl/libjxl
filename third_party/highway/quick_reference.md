# API synopsis / quick reference

[[_TOC_]]

## Headers

*   For static dispatch (choosing the single CPU target at compile-time):
    include `hwy/highway.h`.
*   For dynamic dispatch (generating implementations for multiple targets and
    choosing the best available at runtime): include `hwy/foreach_target.h` and
    see the skeleton example in examples/.

## Preprocessor macros

*   `HWY_ATTR` must be prefixed to any function declaration/definition into
    which Highway functions are (transitively) inlined, i.e. all functions
    inside begin/end_target unless they do not call Highway functions.

*   `HWY_ALIGN`: Ensures an array is aligned and suitable for Load()/Store()
    functions. Example: `HWY_ALIGN T lanes[d.N];`

## Vector and descriptor types

SIMD vectors consist of one or more 'lanes' of the same built-in type `T =
uint##_t, int##_t, float or double` for `## = 8, 16, 32, 64`. Highway provides
data types for vectors of unspecified sizes `N`, where N is a power of two
in the range [1, kMaxVectorSize / sizeof(T)].

Highway relies on "descriptors" (zero-sized class templates) to select the
function overload matching a given `T` and `N`. Users typically define a
descriptor lvalue `d` using aliases:

*   `const HWY_FULL(T) d;` for the largest available N;
*   `const HWY_CAPPED(T, N) d;` for up to `N` lanes.

The API does not document the names of the vector types. Instead, user code
relies on type deduction (`auto`). Examples:

*   Zero-initialize: `auto v0 = Zero(d);`
*   Uninitialized/output variables: `auto v = Undefined(d); Func(&v);`
*   Function arguments or return types: `HWY_VEC(D) Func2(HWY_VEC(D) arg);`

## Operations

Let `V` denote a `D::N`-lane vector (where `D::N` is a power of two).
Operations limited to certain types have prefixes `V`: `u8/16` or `uif` for
unsigned/signed/floating-point types.

### Initialization

*   <code>V **Zero**(D)</code>: returns D::N-lane vector or scalar with all bits
    set to 0.
*   <code>V **Set**(D, T)</code>: returns D::N-lane vector or scalar with all
    lanes equal to the given value of type `T`.
*   <code>V **Iota**(D, T)</code>: returns D::N-lane vector or scalar where the
    lane with index `i` has the given value of type `T` plus `i`. The least
    significant lane has index 0.
*   <code>V **Undefined**(D)</code>: returns uninitialized D::N-lane vector or
    scalar.

### Arithmetic

*   <code>V **operator+**(V a, V b)</code>: returns `a[i] + b[i]` (mod 2^bits).
*   <code>V **operator-**(V a, V b)</code>: returns `a[i] - b[i]` (mod 2^bits).

*   `V`: `ui8/16` \
    <code>V **SaturatedAdd**(V a, V b)</code> returns `a[i] + b[i]` saturated
    to the minimum/maximum representable value.
*   `V`: `ui8/16` \
    <code>V **SaturatedSub**(V a, V b)</code> returns `a[i] - b[i]` saturated
    to the minimum/maximum representable value.

*   `V`: `u8/16` \
    <code>V **AverageRound**(V a, V b)</code> returns `(a[i] + b[i] + 1) / 2`.

*   `V`: `i8/16/32`, `f` \
    <code>V **Abs**(V a)</code> returns the absolute value of `a[i]`;
    for integers, `LimitsMin()` maps to `LimitsMax() + 1`.

*   `V`: `ui8/16/32`, `f` \
    <code>V **Min**(V a, V b)</code>: returns `min(a[i], b[i])`.

*   `V`: `ui8/16/32`, `f` \
    <code>V **Max**(V a, V b)</code>: returns `max(a[i], b[i])`.

*   `V`: `ui8/16/32`, `f` \
    <code>V **Clamp**(V a, V lo, V hi)</code>: returns `a[i]` clamped to
    `[lo[i], hi[i]]`.

*   `V`: `f` \
    <code>V **operator/**(V a, V b)</code>: returns `a[i] / b[i]` in each lane.

*   `V`: `f` \
    <code>V **Sqrt**(V a)</code>: returns `sqrt(a[i])`.

*   `V`: `f32` \
    <code>V **ApproximateReciprocalSqrt**(V a)</code>: returns an approximation
    of `1.0 / sqrt(a[i])`. `sqrt(a) ~= ApproximateReciprocalSqrt(a) * a`. x86
    and PPC provide 12-bit approximations but the error on ARM is closer to 1%.

*   `V`: `f32` \
    <code>V **ApproximateReciprocal**(V a)</code>: returns an approximation of
    `1.0 / a[i]`.

*   `V`: `f32` \
    <code>V **AbsDiff**(V a, V b)</code>: returns `|a[i] - b[i]|` in each
    lane.

#### Multiply

*   `V`: `ui16/32` \
    <code>V <b>operator*</b>(V a, V b)</code>: returns the lower half of
    `a[i] * b[i]` in each lane.

*   `V`: `f` \
    <code>V <b>operator*</b>(V a, V b)</code>: returns `a[i] * b[i]` in each
    lane.

*   `V`: `i16` \
    <code>V **MulHigh**(V a, V b)</code>: returns the upper half of
    `a[i] * b[i]` in each lane.

*   `V`: `ui32` \
    <code>V **MulEven**(V a, V b)</code>: returns double-wide result of
    `a[i] * b[i]` for every even `i`, in lanes `i` (lower) and `i + 1` (upper).

#### Fused multiply-add

When supported, these functions are more precise and faster than separate
multiplication followed by addition. The `*Sub` variants are somewhat slower
on ARM; it is preferable to replace them with MulAdd using a negated constant.

*   `V`: `f` \
    <code>V **MulAdd**(V a, V b, V c)</code>: returns `a[i] * b[i] + c[i]`.

*   `V`: `f` \
    <code>V **NegMulAdd**(V a, V b, V c)</code>: returns `-a[i] * b[i] + c[i]`.

*   `V`: `f` \
    <code>V **MulSub**(V a, V b, V c)</code>: returns `a[i] * b[i] - c[i]`.

*   `V`: `f` \
    <code>V **NegMulSub**(V a, V b, V c)</code>: returns
    `-a[i] * b[i] - c[i]`.

#### Shifts

**Note**: it is generally fastest to shift by a compile-time constant number of
bits. ARM requires the count be less than the lane size.

*   `V`: `ui16/32/64` \
    <code>V **ShiftLeft**&lt;int&gt;(V a)</code> returns `a[i] <<` a
    compile-time constant count.

*   `V`: `u16/32/64`, `i16/32` \
    <code>V **ShiftRight**&lt;int&gt;(V a)</code> returns `a[i] >>` a compile-time
    constant count. Inserts zero or sign bit(s) depending on `V`.

**Note**: independent shifts are only available if `HWY_CAPS & HWY_CAP_VARIABLE_SHIFT`:

*   `V`: `ui32/64` \
    <code>V **operator<<**(V a, V b)</code> returns `a[i] << b[i]`, which is
    zero when `b[i] >= sizeof(T)*8`.

*   `V`: `u32/64`, `i32` \
    <code>V **operator>>**(V a, V b)</code> returns `a[i] >> b[i]`, which is
    zero when `b[i] >= sizeof(T)*8`. Inserts zero or sign bit(s).

**Note**: the following are only provided if `!(HWY_CAPS & HWY_CAP_VARIABLE_SHIFT)`:

*   `V`: `ui16/32/64` \
    <code>V **ShiftLeftSame**(V a, int bits)</code> returns `a[i] << bits`.

*   `V`: `u16/32/64`, `i16/32` \
    <code>V **ShiftRightSame**(V a, int bits)</code> returns `a[i] >> bits`.
    Inserts 0 or sign bit(s).

#### Floating-point rounding

*   `V`: `f` \
    <code>V **Round**(V a)</code>: returns `a[i]` rounded towards the nearest
    integer, with ties to even.

*   `V`: `f` \
    <code>V **Trunc**(V a)</code>: returns `a[i]` rounded towards zero
    (truncate).

*   `V`: `f` \
    <code>V **Ceil**(V a)</code>: returns `a[i]` rounded towards positive
    infinity (ceiling).

*   `V`: `f` \
    <code>V **Floor**(V a)</code>: returns `a[i]` rounded towards negative
    infinity.

### Logical

These operate on individual bits within each lane.

*   <code>V **operator&**(V a, V b)</code>: returns `a[i] & b[i]`.

*   <code>V **operator|**(V a, V b)</code>: returns `a[i] | b[i]`.

*   <code>V **operator^**(V a, V b)</code>: returns `a[i] ^ b[i]`.

*   <code>V **AndNot**(V a, V b)</code>: returns `~a[i] & b[i]`.

For floating-point types, builtin operators are not always available, so we
provide non-operator functions:
*   <code>V **And**(V a, V b)</code>: returns `a[i] & b[i]`.

*   <code>V **Or**(V a, V b)</code>: returns `a[i] | b[i]`.

*   <code>V **Xor**(V a, V b)</code>: returns `a[i] ^ b[i]`.

*   <code>V **AndNot**(V a, V b)</code>: returns `~a[i] & b[i]`.

### Masks

Let `M` denote a mask capable of storing true/false for each lane.

*   <code>M **MaskFromVec**(V v)</code>: returns false in lane `i` if
    `v[i] == 0`, or true if `v[i]` has all bits set.

*   <code>V **VecFromMask**(M m)</code>: returns 0 in lane `i` if
    `m[i] == false`, otherwise all bits set.

*   <code>V **IfThenElse**(M mask, V yes, V no)</code>:
    returns `mask[i] ? yes[i] : no[i]`.
*   <code>V **IfThenElseZero**(M mask, V yes)</code>:
    returns `mask[i] ? yes[i] : 0`.
*   <code>V **IfThenZeroElse**(M mask, V no)</code>:
    returns `mask[i] ? 0 : no[i]`.

*   <code>V **ZeroIfNegative**(V v)</code>: returns `v[i] < 0 ? 0 : v[i]`.

*   <code>bool **AllTrue**(M m)</code>: returns whether all `m[i]` are
    true.
*   <code>bool **AllFalse**(M m)</code>: returns whether all `m[i]` are
    false.

*   <code>uint64_t **BitsFromMask**(M m)</code>: returns `sum{1 << i}`
    for all indices `i` where `m[i]` is true.

*   <code>size_t **CountTrue**(M m)</code>: returns how many of `m[i]` are
    true [0, N]. This is typically more expensive than AllTrue/False.

### Comparisons

These return a mask (see above) indicating whether the condition is true.

*   <code>M **operator==**(V a, V b)</code>: returns `a[i] == b[i]`.

*   `V`: `if` \
    <code>M **operator&lt;**(V a, V b)</code>: returns `a[i] < b[i]`.
*   `V`: `if` \
    <code>M **operator&gt;**(V a, V b)</code>: returns `a[i] > b[i]`.

*   `V`: `f` \
    <code>M **operator&lt;=**(V a, V b)</code>: returns `a[i] <= b[i]`.
*   `V`: `f` \
    <code>M **operator&gt;=**(V a, V b)</code>: returns `a[i] >= b[i]`.

*   `V`: `ui` \
    <code>M **TestBit**(V v, V bit)</code>: returns `(v[i] & bit[i]) == bit[i]`.
    `bit[i]` must have exactly one bit set.

### Memory

Memory operands are little-endian, otherwise their order would depend on the
lane configuration. Pointers are the addresses of `N` consecutive `T` values,
either naturally-aligned (`aligned`) or possibly unaligned (`p`).

#### Load

*   <code>VT&lt;D&gt; **Load**(D, const D::T* aligned)</code>: returns
    `aligned[i]`.
*   <code>VT&lt;D&gt; **LoadU**(D, const D::T* p)</code>: returns `p[i]`.

*   <code>VT&lt;D&gt; **LoadDup128**(D, const D::T* p)</code>: returns one
    128-bit block loaded from `p` and broadcasted into all 128-bit block\[s\].
    This enables a specialized `U32FromU8` that avoids a 3-cycle overhead on
    AVX2/AVX-512. This may be faster than broadcasting single values, and is
    more convenient than preparing constants for the maximum vector length.

#### Gather

**Note**: only available if `HWY_CAPS & HWY_CAP_GATHER`:

*   `V`,`VI`: (`uif32,i32`), (`uif64,i64`) \
    <code>VT&lt;D&gt; **GatherOffset**(D, const D::T* base, VI offsets)</code>.
    Returns elements of base selected by signed/possibly repeated *byte*
    `offsets[i]`.

*   `V`,`VI`: (`uif32,i32`), (`uif64,i64`) \
    <code>VT&lt;D&gt; **GatherIndex**(D, const D::T* base, VI indices)</code>.
    Returns vector of `base[indices[i]]`. Indices are signed and need not be
    unique.

#### Store

*   <code>void **Store**(VT&lt;D&gt; a, D, D::T* aligned)</code>: copies `a[i]`
    into `aligned[i]`, which must be naturally aligned. Writes exactly
    N * sizeof(T) bytes.
*   <code>void **StoreU**(VT&lt;D&gt; a, D, D::T* p)</code>: as Store, but
    without the alignment requirement.

### Cache control

*   <code>void **Stream**(VT&lt;D&gt; a, D, const D::T* aligned)</code>: copies `a[i]`
    into `aligned[i]` with non-temporal hint on x86 (for good performance, call
    for all consecutive vectors within the same cache line). (Over)writes a
    multiple of HWY_STREAM_MULTIPLE bytes.

*   `T`: `u32/64` \
    <code>void **Stream**(T, T* aligned)</code>: copies `T` into `*aligned` with
    non-temporal hint on x86.

*   <code>void **LoadFence**()</code>: delays subsequent loads until prior loads
    are visible. Also a full fence on Intel CPUs. No effect on non-x86.

*   <code>void **StoreFence**()</code>: ensures previous non-temporal stores are
    visible. No effect on non-x86.

*   <code>void **FlushCacheline**(const void* p)</code>: invalidates and flushes
    the cache line containing "p". No effect on non-x86.

*   <code>void **Prefetch**(const T* p)</code>: begins loading the cache line
    containing "p".

### Type conversion

*   <code>VT&lt;D&gt; **BitCast**(D, V)</code>: returns the bits of `V`
    reinterpreted as type `HWY_VEC(D)`.

*   `V`,`D`: (`u8,i16`), (`u8,i32`), (`u16,i32`), (`i8,i16`), (`i8,i32`),
    (`i16,i32`), (`f32,f64`) \
    <code>VT&lt;D&gt; **PromoteTo**(D, V part)</code>: returns `part[i]`
    zero- or sign-extended to the wider `D::T` type.

*   `V`,`D`: (`u8,u32`) \
    <code>VT&lt;D&gt; **U32FromU8**(V)</code>: special-case `u8` to `u32` conversion
    when all blocks of `V` are identical, e.g. from `LoadDup128`.

*   `V`,`D`: (`u32,u8`) \
    <code>VT&lt;D&gt; **U8FromU32**(V)</code>: special-case `u32` to `u8` conversion
    when all lanes of `V` are already clamped to `[0, 256)`.

*   `V`,`D`: (`i16,i8`), (`i32,i8`), (`i32,i16`), (`i16,u8`), (`i32,u8`),
    (`i32,u16`), (`f64,f32`) \
    <code>VT&lt;D&gt; **DemoteTo**(D, V a)</code>: returns `a[i]` after packing
    with signed/unsigned saturation, i.e. a vector with narrower type `D::T`.

*   `V`,`D`: (`i32`,`f32`), (`i64`,`f64`) \
    <code>VT&lt;D&gt; **ConvertTo**(D, V)</code>: converts an integer value to
    same-sized floating point.

*   `V`,`D`: (`f32`,`i32`), (`f64`,`i64`) \
    <code>VT&lt;D&gt; **ConvertTo**(D, V)</code>: rounds floating point towards
    zero and converts the value to same-sized integer.

*   `V`: `f32`; `Ret`: `i32` \
    <code>Ret **NearestInt**(V a)</code>: returns the integer nearest to `a[i]`.

### Swizzle

*   <code>T **GetLane**(V)</code>: returns lane 0 within `V`. This is useful
    for extracting `SumOfLanes` results.

*   <code>V2 **Upper/LowerHalf**(V)</code>: returns upper or lower half of
    the vector `V`.

*   <code>V **OddEven**(V a, V b)</code>: returns a vector whose odd lanes are
    taken from `a` and the even lanes from `b`.

**Note**: if vectors are larger than 128 bits, the following operations split
their operands into independently processed 128-bit *blocks*.

*   `V`: `ui16/32/64`, `f` \
    <code>V **Broadcast**&lt;int i&gt;(V)</code>: returns individual *blocks*,
    each with lanes set to `input_block[i]`, `i = [0, 16/sizeof(T))`.

*   `Ret`: double-width `u/i`; `V`: `u8/16/32`, `i8/16/32` \
    <code>Ret **ZipLower**(V a, V b)</code>: returns the same bits as InterleaveLower,
    except that `Ret` is a vector with double-width lanes (required in order to
    use this operation with `scalar`).

**Note**: the following are only available for full vectors (`N > 1), and split
their operands into independently processed 128-bit *blocks*:

*   `Ret`: double-width u/i; `V`: `u8/16/32`, `i8/16/32` \
    <code>Ret **ZipUpper**(V a, V b)</code>: returns the same bits as InterleaveUpper,
    except that `Ret` is a vector with double-width lanes (required in order to
    use this operation with `scalar`).

*   `V`: `ui` \
    <code>V **ShiftLeftBytes**&lt;int&gt;(V)</code>: returns the result of
    shifting independent *blocks* left by `int` bytes \[1, 15\].

*   `V`: `ui` \
    <code>V **ShiftLeftLanes**&lt;int&gt;(V)</code>: returns the result of
    shifting independent *blocks* left by `int` lanes \[1, 15\].

*   `V`: `ui` \
    <code>V **ShiftRightBytes**&lt;int&gt;(V)</code>: returns the result of
    shifting independent *blocks* right by `int` bytes \[1, 15\].

*   `V`: `ui` \
    <code>V **ShiftRightLanes**&lt;int&gt;(V)</code>: returns the result of
    shifting independent *blocks* right by `int` lanes \[1, 15\].

*   `V`: `ui` \
    <code>V **CombineShiftRightBytes**&lt;int&gt;(V hi, V lo)</code>: returns
    the result of shifting two concatenated *blocks* `hi[i] || lo[i]` right by
    `int` bytes \[1, 15\].

*   `V`: `ui`; `VI`: `ui` \
    <code>V **TableLookupBytes**(V bytes, VI from)</code>: returns *blocks* with
    `bytes[from[i]]`, or zero if bit 7 of byte `from[i]` is set.

*   `V`: `uif32` \
    <code>V **Shuffle2301**(V)</code>: returns *blocks* with 32-bit halves
    swapped inside 64-bit halves.

*   `V`: `uif32` \
    <code>V **Shuffle1032**(V)</code>: returns *blocks* with 64-bit halves
    swapped.

*   `V`: `uif64` \
    <code>V **Shuffle01**(V)</code>: returns *blocks* with 64-bit halves
    swapped.

*   `V`: `uif32` \
    <code>V **Shuffle0321**(V)</code>: returns *blocks* rotated right (toward
    the lower end) by 32 bits.

*   `V`: `uif32` \
    <code>V **Shuffle2103**(V)</code>: returns *blocks* rotated left (toward the
    upper end) by 32 bits.

*   `V`: `uif32` \
    <code>V **Shuffle0123**(V)</code>: returns *blocks* with lanes in reverse
    order.

*   <code>V **InterleaveLower**(V a, V b)</code>: returns *blocks* with alternating
    lanes from the lower halves of `a` and `b` (`a[0]` in the least-significant
    lane).

*   <code>V **InterleaveUpper**(V a, V b)</code>: returns *blocks* with alternating
    lanes from the upper halves of `a` and `b` (`a[N/2]` in the
    least-significant lane).

**Note**: the following operations cross block boundaries, which is typically
more expensive on AVX2/AVX-512 than within-block operations.

*   <code>V **ConcatLowerLower**(V hi, V lo)</code>: returns the concatenation of the
    lower halves of `hi` and `lo` without splitting into blocks.

*   <code>V **ConcatUpperUpper**(V hi, V lo)</code>: returns the concatenation of the
    upper halves of `hi` and `lo` without splitting into blocks.

*   <code>V **ConcatLowerUpper**(V hi, V lo)</code>: returns the inner half of the
    concatenation of `hi` and `lo` without splitting into blocks. Useful for
    swapping the two blocks in 256-bit vectors.

*   <code>V **ConcatUpperLower**(V hi, V lo)</code>: returns the outer quarters of the
    concatenation of `hi` and `lo` without splitting into blocks. Unlike the
    other variants, this does not incur a block-crossing penalty on AVX2.

*   `V`: `uif32` \
    <code>V **TableLookupLanes**(V a, VI)</code> returns a vector of
    `a[indices[i]]`, where `VI` is from `SetTableIndices(D, &indices[0])`.

*   <code>VI **SetTableIndices**(D, int* idx)</code> prepares for
    `TableLookupLanes` with lane indices `idx = [0, d.N)` (need not be unique).

### Misc

**Note**: the following are only available for full vectors (`N > 1`):

*   `V`: `u8`; `Ret`: `u64` \
    <code>Ret **SumsOfU8x8**(V)</code>: returns the sums of 8 consecutive
    bytes in each 64-bit lane.

*   `V`: `uif32/64` \
    <code>V **SumOfLanes**(V v)</code>: returns the sum of all lanes in
    each lane; to obtain the result, use `GetLane(horz_sum_result)`. This is a
    "reduction" (horizontally across lanes), which is less efficient than
    normal ("vertical") SIMD operations.

## Advanced macros

Let `Target` denote an instruction set: `SCALAR/SSE4/AVX2/AVX3/PPC8/NEON/WASM`.

*   `HWY_Target=##` are powers of two uniquely identifying `Target`.

*   `HWY_TARGETS` indicates which targets to generate for dynamic dispatch, and
    which headers to include. It is determined by the configuration macros
    `HWY_{BASELINE|BROKEN|DISABLED}_TARGETS[_ONLY]`.

*   `HWY_STATIC_TARGET` is the best non-disabled/broken baseline `HWY_Target`,
    and matches `HWY_TARGET` in static dispatch mode. This is useful even in
    dynamic dispatch mode for deducing and printing the compiler flags.

*   `HWY_TARGET`: which `HWY_Target` is currently being compiled. This is
    initially identical to `HWY_STATIC_TARGET` and remains so in static dispatch
    mode. For dynamic dispatch, this changes before each re-inclusion and
    finally reverts to `HWY_STATIC_TARGET`. Can be used in `#if` expressions to
    provide an alternative to functions which are not supported by HWY_SCALAR.

*   `HWY_LANES(T)`: how many lanes of type `T` in a full vector (>= 1). Used by
    HWY_FULL/CAPPED. Note: cannot be used in #if because it uses sizeof.

*   `HWY_CAPS` includes zero or more of the following:
    - `HWY_CAP_GATHER`: whether the current target supports GatherIndex/Offset.
    - `HWY_CAP_VARIABLE_SHIFT`: whether the current target supports variable
      shifts, i.e. per-lane shift amounts (v1 << v2).
    - `HWY_CAP_INT64`: whether the current target supports 64-bit integers.
    - `HWY_CAP_CMP64`: whether the current target supports 64-bit signed
      comparisons.
    - `HWY_CAP_DOUBLE`: whether the current target supports double-precision
      vectors.
    - `HWY_CAP_GE256`: the current target supports vectors of >= 256 bits.
    - `HWY_CAP_GE512`: the current target supports vectors of >= 512 bits.

*   `HWY_IDE` is 0 except when parsed by IDEs; adding it to conditions such as
    `#if HWY_TARGET != HWY_SCALAR || HWY_IDE` avoids code appearing greyed out.

## Advanced configuration macros

*   `HWY_DISABLE_CACHE_CONTROL` makes the cache-control functions no-ops.
*   `HWY_DISABLE_BMI2_FMA` prevents emitting BMI/BMI2/FMA instructions.

The following `*_TARGETS` are zero or more `HWY_Target` bits and can be defined
as an expression, e.g. `#define HWY_DISABLED_TARGETS (HWY_SSE4 | HWY_AVX3)`.

*   `HWY_BROKEN_TARGETS` are excluded from `HWY_BASELINE_TARGETS` and
    `HWY_TARGETS`. If undefined, defaults to a blacklist of known compiler bugs.
    Defining to 0 disables the blacklist.

*   `HWY_DISABLED_TARGETS` are excluded from `HWY_BASELINE_TARGETS` and
    `HWY_TARGETS`. If undefined, defaults to zero. This allows explicitly
    disabling targets without interfering with the blacklist.

*   `HWY_BASELINE_TARGETS` minus any broken/disabled targets are interpreted as
    the targets for which the compiler is allowed to generate instructions
    (implying the target CPU would have to support them). We exclude these
    targets except the best of them (which supersedes the others) from
    `HWY_TARGETS` to reduce compile time and binary size. `HWY_STATIC_TARGET` is
    the best of these targets.

    If undefined, defaults to the set whose predefined macros are defined (i.e.
    those for which the corresponding flag, e.g. -mavx2, was passed to the
    compiler).

    To ensure all possible targets are included, which is useful for tests,
    define this to `HWY_SCALAR` **globally** - doing so only for individual
    translation units, e.g. the caller of a `Choose*()` function, may crash if
    the called TU did not generate all the requested targets.

*   `HWY_BASELINE_TARGET_ONLY` governs whether to add to `HWY_TARGETS` any
    additional enabled targets. If defined, only `HWY_STATIC_TARGET` is
    included, which effectively disables dynamic dispatch and reduces code size.
    Otherwise, targets which are enabled and better than any baseline target are
    also included, and thus generated if foreach_target.h is used.

## Compiler support

Clang and GCC require e.g. -mavx2 flags in order to use SIMD intrinsics.
However, this enables AVX2 instructions in the entire translation unit, which
may violate the one-definition rule and cause crashes. Instead, we use
target-specific attribute annotations: any function using SIMD must be prefixed
with `HWY_ATTR`. These are supported by GCC 4.9 or Clang 3.9 and unnecessary in
MSVC.

Immediates (compile-time constants) are specified as template arguments to avoid
constant-propagation issues with Clang on ARM.

## Type traits

*   `IsFloat<T>()` returns true if the T is a floating-point type.
*   `IsSigned<T>()` returns true if the T is a signed or floating-point type.
*   `LimitsMin/Max<T>()` return the smallest/largest value representable in T.
*   `SizeTag<N>` is an empty struct, used to select overloaded functions
    appropriate for N bytes.
