// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef JXL_FIELDS_H_
#define JXL_FIELDS_H_

// Forward/backward-compatible 'bundles' with auto-serialized 'fields'.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cinttypes>
#include <cmath>  // abs
#include <cstdarg>

#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/field_encodings.h"

namespace jxl {

// Integer coders: BitsCoder (raw), U32Coder (table), U64Coder (varint).

// Reads/writes a given (fixed) number of bits <= 32.
class BitsCoder {
 public:
  static size_t MaxEncodedBits(const size_t bits) { return bits; }

  static Status CanEncode(const size_t bits, const uint32_t value,
                          size_t* JXL_RESTRICT encoded_bits) {
    *encoded_bits = bits;
    if (value >= (1ULL << bits)) {
      return JXL_FAILURE("Value %u too large for %zu bits", value, bits);
    }
    return true;
  }

  static uint32_t Read(const size_t bits, BitReader* JXL_RESTRICT reader) {
    return reader->ReadBits(bits);
  }

  // Returns false if the value is too large to encode.
  static Status Write(const size_t bits, const uint32_t value,
                      BitWriter* JXL_RESTRICT writer) {
    if (value >= (1ULL << bits)) {
      return JXL_FAILURE("Value %d too large to encode in %zu bits", value,
                         bits);
    }
    writer->Write(bits, value);
    return true;
  }
};

// Encodes u32 using a lookup table and/or extra bits, governed by a per-field
// encoding `enc` which consists of four distributions `d` chosen via a 2-bit
// selector (least significant = 0). Each d may have two modes:
// - direct: if d.IsDirect(), the value is d.Direct();
// - offset: the value is derived from d.ExtraBits() extra bits plus d.Offset();
// This encoding is denser than Exp-Golomb or Gamma codes when both small and
// large values occur.
//
// Examples:
// Direct: U32Enc(Val(8), Val(16), Val(32), Bits(6)), value 32 => 10b.
// Offset: U32Enc(Val(0), BitsOffset(1, 1), BitsOffset(2, 3), BitsOffset(8, 8))
//   defines the following prefix code:
//   00 -> 0
//   01x -> 1..2
//   10xx -> 3..7
//   11xxxxxxxx -> 8..263
class U32Coder {
 public:
  static size_t MaxEncodedBits(U32Enc enc);
  static Status CanEncode(U32Enc enc, uint32_t value,
                          size_t* JXL_RESTRICT encoded_bits);
  static uint32_t Read(U32Enc enc, BitReader* JXL_RESTRICT reader);

  // Returns false if the value is too large to encode.
  static Status Write(U32Enc enc, uint32_t value,
                      BitWriter* JXL_RESTRICT writer);

 private:
  static Status ChooseSelector(U32Enc enc, uint32_t value,
                               uint32_t* JXL_RESTRICT selector,
                               size_t* JXL_RESTRICT total_bits);
};

// Encodes 64-bit unsigned integers with a fixed distribution, taking 2 bits
// to encode 0, 6 bits to encode 1 to 16, 10 bits to encode 17 to 272, 15 bits
// to encode up to 4095, and on the order of log2(value) * 1.125 bits for
// larger values.
class U64Coder {
 public:
  static constexpr size_t MaxEncodedBits() {
    return 2 + 12 + 6 * (8 + 1) + (4 + 1);
  }

  static uint64_t Read(BitReader* JXL_RESTRICT reader);

  // Returns false if the value is too large to encode.
  static Status Write(uint64_t value, BitWriter* JXL_RESTRICT writer);

  // Can always encode, but useful because it also returns bit size.
  static Status CanEncode(uint64_t value, size_t* JXL_RESTRICT encoded_bits);
};

// IEEE 754 half-precision (binary16). Refuses to read/write NaN/Inf.
class F16Coder {
 public:
  static constexpr size_t MaxEncodedBits() { return 16; }

  // Returns false if the bit representation is NaN or infinity
  static Status Read(BitReader* JXL_RESTRICT reader, float* JXL_RESTRICT value);

  // Returns false if the value is too large to encode.
  static Status Write(float value, BitWriter* JXL_RESTRICT writer);
  static Status CanEncode(float value, size_t* JXL_RESTRICT encoded_bits);
};

// A "bundle" is a forward- and backward compatible collection of fields.
// They are used for SizeHeader/FrameHeader/GroupHeader. Bundles can be
// extended by appending(!) fields. Optional fields may be omitted from the
// bitstream by conditionally visiting them. When reading new bitstreams with
// old code, we skip unknown fields at the end of the bundle. This requires
// storing the amount of extra appended bits, and that fields are visited in
// chronological order of being added to the format, because old decoders
// cannot skip some future fields and resume reading old fields. Similarly,
// new readers query bits in an "extensions" field to skip (groups of) fields
// not present in old bitstreams. Note that each bundle must include an
// "extensions" field prior to freezing the format, otherwise it cannot be
// extended.
//
// To ensure interoperability, there will be no opaque fields.
//
// HOWTO:
// - basic usage: define a struct with member variables ("fields") and a
//   VisitFields(v) member function that calls v->U32/Bool etc. for each
//   field, specifying their default values. The ctor must call
//   Bundle::Init(this).
//
// - print a trace of visitors: ensure each bundle has a static Name() member
//   function, and change Bundle::Print* to return true.
//
// - optional fields: in VisitFields, add if (v->Conditional(your_condition))
//   { v->Bool(default, &field); }. This prevents reading/writing field
//   if !your_condition, which is typically computed from a prior field.
//   WARNING: to ensure all fields are initialized, do not add an else branch;
//   instead add another if (v->Conditional(!your_condition)).
//
// - repeated fields: for dynamic sizes, use e.g. std::vector and in
//   VisitFields, if (v->IfReading()) field.resize(size) before accessing field.
//   For static or bounded sizes, use an array or std::array. In all cases,
//   simply visit each array element as if it were a normal field.
//
// - nested bundles: add a bundle as a normal field and in VisitFields call
//   JXL_RETURN_IF_ERROR(v->VisitNested(&nested));
//
// - allow future extensions: define a "uint64_t extensions" field and call
//   v->BeginExtensions(&extensions) after visiting all non-extension fields,
//   and `return v->EndExtensions();` after the last extension field.
//
// - encode an entire bundle in one bit if ALL its fields equal their default
//   values: add a "mutable bool all_default" field and as the first visitor:
//   if (v->AllDefault(*this, &all_default)) {
//     // Overwrite all serialized fields, but not any nonserialized_*.
//     InitFields();
//     return true; }
//   Note: if extensions are present, AllDefault() == false. To avoid depending
//   on fields.h, InitFields is a private non-inlined member function that calls
//   Bundle::Init(this).

class Bundle {
 public:
  // Print the type of each visitor called.
  static constexpr bool PrintVisitors() { return false; }
  // Print default value for each field and AllDefault result.
  static constexpr bool PrintAllDefault() { return false; }
  // Print values decoded for each field in Read.
  static constexpr bool PrintRead() { return false; }
  // Print size for each field and CanEncode total_bits.
  static constexpr bool PrintSizes() { return false; }

  template <class T>
  static void Init(T* JXL_RESTRICT t) {
    InitVisitor visitor;
    if (!visitor.Visit(t, PrintVisitors() ? "-- Init\n" : "")) {
      JXL_ASSERT(false);  // Init should never fail.
    }
  }

  // Returns whether ALL fields (including `extensions`, if present) are equal
  // to their default value.
  template <class T>
  static bool AllDefault(const T& t) {
    AllDefaultVisitor visitor;
    const char* name =
        (PrintVisitors() || PrintAllDefault()) ? "[[AllDefault\n" : "";
    if (!visitor.VisitConst(t, name)) {
      JXL_ASSERT(false);  // AllDefault should never fail.
    }

    if (PrintAllDefault()) printf("  %d]]\n", visitor.AllDefault());
    return visitor.AllDefault();
  }

  // Returns max number of bits required to encode a T.
  template <class T>
  static size_t MaxBits(const T& t) {
    MaxBitsVisitor visitor;
#if JXL_ENABLE_ASSERT
    Status ret =
#else
    (void)
#endif  // JXL_ENABLE_ASSERT
        visitor.VisitConst(t, PrintVisitors() ? "-- MaxBits\n" : "");
    JXL_ASSERT(ret);
    return visitor.MaxBits();
  }

  // Returns whether a header's fields can all be encoded, i.e. they have a
  // valid representation. If so, "*total_bits" is the exact number of bits
  // required. Called by Write.
  template <class T>
  static Status CanEncode(const T& t, size_t* JXL_RESTRICT extension_bits,
                          size_t* JXL_RESTRICT total_bits) {
    CanEncodeVisitor visitor;
    const char* name = (PrintVisitors() || PrintSizes()) ? "[[CanEncode\n" : "";
    JXL_RETURN_IF_ERROR(visitor.VisitConst(t, name));
    JXL_RETURN_IF_ERROR(visitor.GetSizes(extension_bits, total_bits));
    if (PrintSizes()) printf("  %zu]]\n", *total_bits);
    return true;
  }

  template <class T>
  static Status Read(BitReader* reader, T* JXL_RESTRICT t) {
    ReadVisitor visitor(reader);
    JXL_RETURN_IF_ERROR(visitor.Visit(t, PrintVisitors() ? "-- Read\n" : ""));
    return visitor.OK();
  }

  template <class T>
  static Status Write(const T& t, BitWriter* JXL_RESTRICT writer, size_t layer,
                      AuxOut* aux_out) {
    size_t extension_bits, total_bits;
    JXL_RETURN_IF_ERROR(CanEncode(t, &extension_bits, &total_bits));

    BitWriter::Allotment allotment(writer, total_bits);
    WriteVisitor visitor(extension_bits, writer);
    JXL_RETURN_IF_ERROR(
        visitor.VisitConst(t, PrintVisitors() ? "-- Write\n" : ""));
    JXL_RETURN_IF_ERROR(visitor.OK());
    ReclaimAndCharge(writer, &allotment, layer, aux_out);
    return true;
  }

 private:
  // A bundle can be in one of three states concerning extensions: not-begun,
  // active, ended. Bundles may be nested, so we need a stack of states.
  class ExtensionStates {
   public:
    static constexpr size_t kMaxDepth = 64;

    void Push() {
      // Initial state = not-begun.
      begun_ <<= 1;
      ended_ <<= 1;
    }

    // Clears current state; caller must check IsEnded beforehand.
    void Pop() {
      begun_ >>= 1;
      ended_ >>= 1;
    }

    // Returns true if state == active || state == ended.
    Status IsBegun() const { return (begun_ & 1) != 0; }
    // Returns true if state != not-begun && state != active.
    Status IsEnded() const { return (ended_ & 1) != 0; }

    void Begin() {
      JXL_ASSERT(!IsBegun());
      JXL_ASSERT(!IsEnded());
      begun_ += 1;
    }

    void End() {
      JXL_ASSERT(IsBegun());
      JXL_ASSERT(!IsEnded());
      ended_ += 1;
    }

   private:
    // Current state := least-significant bit of begun_ and ended_.
    uint64_t begun_ = 0;
    uint64_t ended_ = 0;
  };

  // Visitors generate Init/AllDefault/Read/Write logic for all fields. Each
  // bundle's VisitFields member function calls visitor->U32 etc. We do not
  // overload operator() because a function name is easier to search for.

  template <class Derived>
  class VisitorBase {
   public:
    explicit VisitorBase(bool print_bundles = false)
        : print_bundles_(print_bundles) {}
    ~VisitorBase() { JXL_ASSERT(depth_ == 0); }

    // This is the only call site of T::VisitFields. Adds tracing and ensures
    // EndExtensions was called.
    template <class T>
    Status Visit(T* t, const char* visitor_name) {
      fputs(visitor_name, stdout);  // No newline; no effect if empty
      if (print_bundles_) {
        Trace("%s\n", print_bundles_ ? T::Name() : "");
      }

      depth_ += 1;
      JXL_ASSERT(depth_ <= ExtensionStates::kMaxDepth);
      extension_states_.Push();

      Derived* self = static_cast<Derived*>(this);
      const Status ok = t->VisitFields(self);

      if (ok) {
        // If VisitFields called BeginExtensions, must also call
        // EndExtensions.
        JXL_ASSERT(!extension_states_.IsBegun() || extension_states_.IsEnded());
      } else {
        // Failed, undefined state: don't care whether EndExtensions was
        // called.
      }

      extension_states_.Pop();
      JXL_ASSERT(depth_ != 0);
      depth_ -= 1;

      return ok;
    }

    // For visitors accepting a const T, need to const-cast so we can call the
    // non-const T::VisitFields. NOTE: T is not modified except the
    // `all_default` field by CanEncodeVisitor.
    template <class T>
    Status VisitConst(const T& t, const char* message) {
      return Visit(const_cast<T*>(&t), message);
    }

    // Helper to construct U32Enc from U32Distr.
    void U32(const U32Distr d0, const U32Distr d1, const U32Distr d2,
             const U32Distr d3, const uint32_t default_value,
             uint32_t* JXL_RESTRICT value) {
      Derived* self = static_cast<Derived*>(this);
      // The rarely-used function that takes a U32Enc must have a different
      // name, else it would HIDE the base class U32() member.
      self->U32WithEnc(U32Enc(d0, d1, d2, d3), default_value, value);
    }

    // Derived types (overridden by InitVisitor because it is unsafe to read
    // from *value there)

    void Bool(bool default_value, bool* JXL_RESTRICT value) {
      uint32_t bits = *value ? 1 : 0;
      Derived* self = static_cast<Derived*>(this);
      self->Bits(1, static_cast<uint32_t>(default_value), &bits);
      JXL_DASSERT(bits <= 1);
      *value = bits == 1;
    }

    template <typename EnumT>
    Status Enum(const EnumT default_value, EnumT* JXL_RESTRICT value) {
      Derived* self = static_cast<Derived*>(this);
      uint32_t u32 = static_cast<uint32_t>(*value);
      // 00 -> 0
      // 01 -> 1
      // 10xxxx -> 2..17
      // 11yyyyyy -> 18..81
      self->U32(Val(0), Val(1), BitsOffset(4, 2), BitsOffset(6, 18),
                static_cast<uint32_t>(default_value), &u32);
      *value = static_cast<EnumT>(u32);
      return EnumValid(*value);
    }

    void S32(const U32Distr d0, const U32Distr d1, const U32Distr d2,
             const U32Distr d3, const int32_t default_value,
             int32_t* JXL_RESTRICT value) {
      uint32_t u32 = U32FromS32(*value);
      Derived* self = static_cast<Derived*>(this);
      self->U32(d0, d1, d2, d3, U32FromS32(default_value), &u32);
      *value = S32FromU32(u32);
    }

    void S64(const int64_t default_value, int64_t* JXL_RESTRICT value) {
      uint64_t u64 = U64FromS64(*value);
      Derived* self = static_cast<Derived*>(this);
      self->U64(U64FromS64(default_value), &u64);
      *value = S64FromU64(u64);
    }

    // Returns whether VisitFields should visit some subsequent fields.
    // "condition" is typically from prior fields, e.g. flags.
    // Overridden by InitVisitor and MaxBitsVisitor.
    Status Conditional(bool condition) { return condition; }

    // Overridden by InitVisitor, AllDefaultVisitor and CanEncodeVisitor.
    template <class Fields>
    Status AllDefault(const Fields& fields, bool* JXL_RESTRICT all_default) {
      Derived* self = static_cast<Derived*>(this);
      self->Bool(true, all_default);
      return *all_default;
    }

    // Returns the result of visiting a nested Bundle.
    // Overridden by InitVisitor.
    template <class Fields>
    Status VisitNested(Fields* fields) {
      Derived* self = static_cast<Derived*>(this);
      return self->Visit(fields, "");
    }

    // Overridden by ReadVisitor. Enables dynamically-sized fields.
    Status IsReading() const { return false; }

    // Overriden by ReadVisitor and WriteVisitor.
    // Called before any conditional visit based on "extensions".
    // Overridden by ReadVisitor, CanEncodeVisitor and WriteVisitor.
    void BeginExtensions(uint64_t* JXL_RESTRICT extensions) {
      Derived* self = static_cast<Derived*>(this);
      self->U64(0, extensions);

      extension_states_.Begin();
    }

    // Called after all extension fields (if any). Although non-extension
    // fields could be visited afterward, we prefer the convention that
    // extension fields are always the last to be visited. Overridden by
    // ReadVisitor.
    Status EndExtensions() {
      extension_states_.End();
      return true;
    }

   protected:
    // Prints indentation, <format_in>.
    JXL_FORMAT(2, 3)  // 1-based plus one because member function
    void Trace(const char* format_in, ...) const {
      // New format string with indentation included
      char format[200];
      memset(format, ' ', depth_ * 2);  // indentation
      const size_t pos = depth_ * 2;
      strncpy(format + pos, format_in, sizeof(format) - pos - 1);
      format[sizeof(format) - 1] = '\0';

      char buf[2000];
      va_list args;
      va_start(args, format_in);
      vsnprintf(buf, sizeof(buf), format, args);
      va_end(args);
      fputs(buf, stdout);  // unlike puts, does not add newline
    }

   private:
    size_t depth_ = 0;  // for indentation.
    ExtensionStates extension_states_;
    const bool print_bundles_;
  };

  struct InitVisitor : public VisitorBase<InitVisitor> {
    void Bits(const size_t /*unused*/, const uint32_t default_value,
              uint32_t* JXL_RESTRICT value) {
      *value = default_value;
    }

    void U32WithEnc(const U32Enc /*unused*/, const uint32_t default_value,
                    uint32_t* JXL_RESTRICT value) {
      *value = default_value;
    }

    void U64(const uint64_t default_value, uint64_t* JXL_RESTRICT value) {
      *value = default_value;
    }

    void Bool(bool default_value, bool* JXL_RESTRICT value) {
      *value = default_value;
    }

    template <typename T>
    Status Enum(const T default_value, T* JXL_RESTRICT value) {
      *value = default_value;
      return EnumValid(*value);
    }

    void S32(U32Distr /*unused*/, U32Distr /*unused*/, U32Distr /*unused*/,
             U32Distr /*unused*/, const int32_t default_value,
             int32_t* JXL_RESTRICT value) {
      *value = default_value;
    }

    void S64(const int64_t default_value, int64_t* JXL_RESTRICT value) {
      *value = default_value;
    }

    void F16(const float default_value, float* JXL_RESTRICT value) {
      *value = default_value;
    }

    // Always visit conditional fields to ensure they are initialized.
    Status Conditional(bool condition) { return true; }

    template <class Fields>
    Status AllDefault(const Fields& fields, bool* JXL_RESTRICT all_default) {
      // Just initialize this field and don't skip initializing others.
      Bool(true, all_default);
      return false;
    }

    template <class Fields>
    Status VisitNested(Fields* fields) {
      // Avoid re-initializing nested bundles (their ctors already called
      // Bundle::Init for their fields).
      return true;
    }
  };

  class AllDefaultVisitor : public VisitorBase<AllDefaultVisitor> {
   public:
    AllDefaultVisitor() : VisitorBase<AllDefaultVisitor>(PrintAllDefault()) {}

    void Bits(const size_t bits, const uint32_t default_value,
              const uint32_t* JXL_RESTRICT value) {
      if (PrintAllDefault()) {
        Trace("  u(%zu) = %u, default %u\n", bits, *value, default_value);
      }

      all_default_ &= *value == default_value;
    }

    void U32WithEnc(const U32Enc /*unused*/, const uint32_t default_value,
                    const uint32_t* JXL_RESTRICT value) {
      if (PrintAllDefault()) {
        Trace("  U32 = %u, default %u\n", *value, default_value);
      }

      all_default_ &= *value == default_value;
    }

    void U64(const uint64_t default_value, const uint64_t* JXL_RESTRICT value) {
      if (PrintAllDefault()) {
        Trace("  U64 = %" PRIu64 ", default %" PRIu64 "\n", *value,
              default_value);
      }

      all_default_ &= *value == default_value;
    }

    void F16(const float default_value, const float* JXL_RESTRICT value) {
      if (PrintAllDefault()) {
        Trace("  F16 = %.6f, default %.6f\n", *value, default_value);
      }
      all_default_ &= std::abs(*value - default_value) < 1E-6f;
    }

    template <class Fields>
    Status AllDefault(const Fields& fields, bool* JXL_RESTRICT all_default) {
      // Visit all fields so we can compute the actual all_default_ value.
      return false;
    }

    bool AllDefault() const { return all_default_; }

   private:
    bool all_default_ = true;
  };

  class ReadVisitor : public VisitorBase<ReadVisitor> {
   public:
    explicit ReadVisitor(BitReader* reader)
        : VisitorBase(PrintRead()), reader_(reader) {}

    void Bits(const size_t bits, const uint32_t default_value,
              uint32_t* JXL_RESTRICT value) {
      *value = BitsCoder::Read(bits, reader_);
      if (PrintRead()) Trace("  u(%zu) = %u\n", bits, *value);
    }

    void U32WithEnc(const U32Enc dist, const uint32_t default_value,
                    uint32_t* JXL_RESTRICT value) {
      *value = U32Coder::Read(dist, reader_);
      if (PrintRead()) Trace("  U32 = %u\n", *value);
    }

    void U64(const uint64_t default_value, uint64_t* JXL_RESTRICT value) {
      *value = U64Coder::Read(reader_);
      if (PrintRead()) Trace("  U64 = %" PRIu64 "\n", *value);
    }

    void F16(const float default_value, float* JXL_RESTRICT value) {
      ok_ &= F16Coder::Read(reader_, value);
      if (PrintRead()) Trace("  F16 = %f\n", *value);
    }

    Status IsReading() const { return true; }

    void BeginExtensions(uint64_t* JXL_RESTRICT extensions) {
      VisitorBase<ReadVisitor>::BeginExtensions(extensions);
      if (*extensions != 0) {
        // Read the additional U64 indicating the number of extension bits
        // (more compact than sending the total size).
        extension_bits_ = U64Coder::Read(reader_);  // >= 0
        // Used by EndExtensions to skip past any _remaining_ extensions.
        pos_after_ext_size_ = reader_->TotalBitsConsumed();
        JXL_ASSERT(pos_after_ext_size_ != 0);
      }
    }

    Status EndExtensions() {
      JXL_RETURN_IF_ERROR(VisitorBase<ReadVisitor>::EndExtensions());
      // Happens if extensions == 0: don't read size, done.
      if (pos_after_ext_size_ == 0) return true;

      // Skip new fields this (old?) decoder didn't know about, if any.
      const size_t bits_read = reader_->TotalBitsConsumed();
      const uint64_t end = pos_after_ext_size_ + extension_bits_;
      if (bits_read > end) {
        return JXL_FAILURE("Read more extension bits than budgeted");
      }
      const size_t remaining_bits = end - bits_read;
      if (remaining_bits != 0) {
        JXL_WARNING("Skipping %zu-bit extension(s)", remaining_bits);
        reader_->SkipBits(remaining_bits);
      }
      return true;
    }

    Status OK() const { return ok_; }

   private:
    bool ok_ = true;
    BitReader* const reader_;
    uint64_t extension_bits_ = 0;    // May be 0 even if extensions present.
    size_t pos_after_ext_size_ = 0;  // 0 iff extensions == 0.
  };

  class MaxBitsVisitor : public VisitorBase<MaxBitsVisitor> {
   public:
    void Bits(const size_t bits, const uint32_t default_value,
              const uint32_t* JXL_RESTRICT value) {
      max_bits_ += BitsCoder::MaxEncodedBits(bits);
    }

    void U32WithEnc(const U32Enc enc, const uint32_t default_value,
                    const uint32_t* JXL_RESTRICT value) {
      max_bits_ += U32Coder::MaxEncodedBits(enc);
    }

    void U64(const uint64_t default_value, const uint64_t* JXL_RESTRICT value) {
      max_bits_ += U64Coder::MaxEncodedBits();
    }

    void F16(const float default_value, const float* JXL_RESTRICT value) {
      max_bits_ += F16Coder::MaxEncodedBits();
    }

    template <class Fields>
    Status AllDefault(const Fields& fields, bool* JXL_RESTRICT all_default) {
      Bool(true, all_default);
      return false;
    }

    // Always visit conditional fields to get a (loose) upper bound.
    Status Conditional(bool condition) { return true; }

    void BeginExtensions(uint64_t* JXL_RESTRICT extensions) {
      // Skip - extensions are not included in "MaxBits" because their length
      // is potentially unbounded.
    }

    Status EndExtensions() { return true; }

    size_t MaxBits() const { return max_bits_; }

   private:
    size_t max_bits_ = 0;
  };

  class CanEncodeVisitor : public VisitorBase<CanEncodeVisitor> {
   public:
    CanEncodeVisitor() : VisitorBase<CanEncodeVisitor>(PrintSizes()) {}

    void Bits(const size_t bits, const uint32_t default_value,
              const uint32_t* JXL_RESTRICT value) {
      size_t encoded_bits = 0;
      ok_ &= BitsCoder::CanEncode(bits, *value, &encoded_bits);
      if (PrintSizes()) Trace("u(%zu) = %u\n", bits, *value);
      encoded_bits_ += encoded_bits;
    }

    void U32WithEnc(const U32Enc enc, const uint32_t default_value,
                    const uint32_t* JXL_RESTRICT value) {
      size_t encoded_bits = 0;
      ok_ &= U32Coder::CanEncode(enc, *value, &encoded_bits);
      if (PrintSizes()) Trace("U32(%zu) = %u\n", encoded_bits, *value);
      encoded_bits_ += encoded_bits;
    }

    void U64(const uint64_t default_value, const uint64_t* JXL_RESTRICT value) {
      size_t encoded_bits = 0;
      ok_ &= U64Coder::CanEncode(*value, &encoded_bits);
      if (PrintSizes()) {
        Trace("U64(%zu) = %" PRIu64 "\n", encoded_bits, *value);
      }
      encoded_bits_ += encoded_bits;
    }

    void F16(const float default_value, const float* JXL_RESTRICT value) {
      size_t encoded_bits = 0;
      ok_ &= F16Coder::CanEncode(*value, &encoded_bits);
      if (PrintSizes()) {
        Trace("F16(%zu) = %.6f\n", encoded_bits, *value);
      }
      encoded_bits_ += encoded_bits;
    }

    template <class Fields>
    Status AllDefault(const Fields& fields, bool* JXL_RESTRICT all_default) {
      *all_default = Bundle::AllDefault(fields);
      Bool(true, all_default);
      return *all_default;
    }

    void BeginExtensions(uint64_t* JXL_RESTRICT extensions) {
      VisitorBase<CanEncodeVisitor>::BeginExtensions(extensions);
      if (*extensions != 0) {
        JXL_ASSERT(pos_after_ext_ == 0);
        pos_after_ext_ = encoded_bits_;
        JXL_ASSERT(pos_after_ext_ != 0);  // visited "extensions"
      }
    }
    // EndExtensions = default.

    Status GetSizes(size_t* JXL_RESTRICT extension_bits,
                    size_t* JXL_RESTRICT total_bits) {
      JXL_RETURN_IF_ERROR(ok_);
      *extension_bits = 0;
      *total_bits = encoded_bits_;
      // Only if extension field was nonzero will we encode the size.
      if (pos_after_ext_ != 0) {
        JXL_ASSERT(encoded_bits_ >= pos_after_ext_);
        *extension_bits = encoded_bits_ - pos_after_ext_;
        // Also need to encode *extension_bits and bill it to *total_bits.
        size_t encoded_bits = 0;
        ok_ &= U64Coder::CanEncode(*extension_bits, &encoded_bits);
        *total_bits += encoded_bits;
      }
      return true;
    }

   private:
    bool ok_ = true;
    size_t encoded_bits_ = 0;
    // Snapshot of encoded_bits_ after visiting the extension field, but NOT
    // including the hidden "extension_bits" u64.
    uint64_t pos_after_ext_ = 0;
  };

  class WriteVisitor : public VisitorBase<WriteVisitor> {
   public:
    WriteVisitor(const size_t extension_bits, BitWriter* JXL_RESTRICT writer)
        : extension_bits_(extension_bits), writer_(writer) {}

    void Bits(const size_t bits, const uint32_t default_value,
              const uint32_t* JXL_RESTRICT value) {
      ok_ &= BitsCoder::Write(bits, *value, writer_);
    }
    void U32WithEnc(const U32Enc enc, const uint32_t default_value,
                    const uint32_t* JXL_RESTRICT value) {
      ok_ &= U32Coder::Write(enc, *value, writer_);
    }

    void U64(const uint64_t default_value, const uint64_t* JXL_RESTRICT value) {
      ok_ &= U64Coder::Write(*value, writer_);
    }

    void F16(const float default_value, const float* JXL_RESTRICT value) {
      ok_ &= F16Coder::Write(*value, writer_);
    }

    void BeginExtensions(uint64_t* JXL_RESTRICT extensions) {
      VisitorBase<WriteVisitor>::BeginExtensions(extensions);
      if (*extensions == 0) {
        JXL_ASSERT(extension_bits_ == 0);
      } else {
        // NOTE: extension_bits_ can be zero if the extensions do not require
        // any additional fields.
        ok_ &= U64Coder::Write(extension_bits_, writer_);
      }
    }
    // EndExtensions = default.

    Status OK() const { return ok_; }

   private:
    const size_t extension_bits_;
    BitWriter* JXL_RESTRICT writer_;
    bool ok_ = true;
  };
};

}  // namespace jxl

#endif  // JXL_FIELDS_H_
