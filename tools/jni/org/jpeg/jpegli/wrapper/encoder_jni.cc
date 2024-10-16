// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "encoder_jni.h"  // NOLINT: build/include

#include <jni.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>

#include "lib/jpegli/common.h"
#include "lib/jpegli/encode.h"
#include "lib/jpegli/types.h"

namespace org_jpeg_jpegli_wrapper {
namespace {

jint JNI_VERSION = JNI_VERSION_1_6;

jclass JC_WritableByteChannel;
jmethodID JMID_WritableByteChannel_write;

enum ReturnCode {
  OK = 0,
  ERROR_ALLOCATION = -1,
  ERROR_INVALID_PARAMS = -2,
  ERROR_INTERNAL = -3
};

// #define RETURN_ERROR(T) return ERROR_##T
#define RETURN_ERROR(T) \
  return fprintf(stderr, "%s:%d: error " #T "\n", __FILE__, __LINE__), ERROR_##T

void ExitHandler(j_common_ptr cinfo) {
  (*cinfo->err->output_message)(cinfo);
  jmp_buf* env = reinterpret_cast<jmp_buf*>(cinfo->client_data);
  jpegli_destroy(cinfo);
  longjmp(*env, 1);
}

struct DestinationManager {
  // IMPORTANT: this should always be the first member!
  jpeg_destination_mgr pub;

  std::unique_ptr<uint8_t[]> output = nullptr;
  size_t output_size;

  JNIEnv* jenv;
  jobject sink;
  bool has_error = false;

  DestinationManager() {
    pub.init_destination = init_destination;
    pub.empty_output_buffer = empty_output_buffer;
    pub.term_destination = term_destination;
  }

  void Rewind() {
    pub.next_output_byte = output.get();
    pub.free_in_buffer = output_size;
  }

  void WriteAndRewind() {
    size_t to_write = pub.next_output_byte - output.get();
    if (to_write == 0) {
      return;
    }
    jobject byte_buffer = jenv->NewDirectByteBuffer(output.get(), to_write);
    if (byte_buffer == nullptr || jenv->ExceptionCheck()) {
      has_error = true;
      Rewind();
      return;
    }
    jint num_written =
        jenv->CallIntMethod(sink, JMID_WritableByteChannel_write, byte_buffer);
    jenv->DeleteLocalRef(byte_buffer);
    if (num_written != static_cast<jint>(to_write) || jenv->ExceptionCheck()) {
      has_error = true;
    }
    Rewind();
  }

  static void init_destination(j_compress_ptr cinfo) {
    auto* self = reinterpret_cast<DestinationManager*>(cinfo->dest);
    self->Rewind();
  }

  static boolean empty_output_buffer(j_compress_ptr cinfo) {
    auto* self = reinterpret_cast<DestinationManager*>(cinfo->dest);
    if (self->has_error) return FALSE;
    self->WriteAndRewind();
    if (self->has_error) return FALSE;
    return TRUE;
  }

  static void term_destination(j_compress_ptr cinfo) {
    auto* self = reinterpret_cast<DestinationManager*>(cinfo->dest);
    self->WriteAndRewind();
  }
};

typedef std::array<jint, 33> Config;

class Encoder {
 public:
  Encoder(JNIEnv* jenv, jint width, jint height, const Config& config,
          jintArray input, jobject output)
      : jenv_(jenv), width_(width), height_(height), input_(input) {
    healthy_ = (static_cast<jint>(width_) == width) &&
               (static_cast<jint>(height_) == height);

    jint config_present = config[32];
    if (config_present & 1) {
      quality_ = config[0];
      healthy_ &= (static_cast<jint>(quality_) == config[0]);
      healthy_ &= (quality_ >= 1 && quality_ <= 100);
    } else {
      healthy_ = false;  // quality is mandatory
    }

    dest_.jenv = jenv;
    dest_.sink = output;

    size_t justified_width = (width + 31) & ~31;
    size_t output_buffer_min = batch_lines_ * justified_width * 4;
    size_t output_buffer_tmp = 1;
    while (output_buffer_tmp < output_buffer_min) output_buffer_tmp <<= 1;
    output_buffer_size_ = output_buffer_tmp;
  }

  ~Encoder() { jpegli_destroy_compress(&cinfo_); }

  int Run() {
    if (!healthy_) RETURN_ERROR(INVALID_PARAMS);

    std::unique_ptr<JSAMPROW[]> batch_rows{new (std::nothrow)
                                               JSAMPROW[batch_lines_]};
    if (!batch_rows) RETURN_ERROR(ALLOCATION);

    size_t input_stride = width_ * 4;
    std::unique_ptr<uint8_t[]> batch{new (std::nothrow)
                                         uint8_t[input_stride * batch_lines_]};
    if (!batch) RETURN_ERROR(ALLOCATION);

    size_t stride = width_ * 3;
    for (size_t i = 0; i < batch_lines_; ++i) {
      batch_rows[i] = batch.get() + i * stride;
    }

    dest_.output_size = output_buffer_size_;
    dest_.output = std::unique_ptr<uint8_t[]>{new (std::nothrow)
                                                  uint8_t[output_buffer_size_]};
    if (!dest_.output) RETURN_ERROR(ALLOCATION);
    dest_.Rewind();

    // Setup error handling.
    cinfo_.err = jpegli_std_error(&err_);
    cinfo_.client_data = reinterpret_cast<void*>(&env_);
    cinfo_.err->error_exit = &ExitHandler;
    if (setjmp(env_)) {
      RETURN_ERROR(INTERNAL);
    }

    jpegli_create_compress(&cinfo_);
    cinfo_.dest = reinterpret_cast<jpeg_destination_mgr*>(&dest_);

    cinfo_.image_width = width_;
    cinfo_.image_height = height_;
    cinfo_.input_components = 3;
    cinfo_.in_color_space = JCS_RGB;
    jpegli_set_defaults(&cinfo_);
    jpegli_set_quality(&cinfo_, quality_, TRUE);
    cinfo_.comp_info[0].v_samp_factor = v_sampling_[0];
    jpegli_set_progressive_level(&cinfo_, 0);
    cinfo_.optimize_coding = FALSE;
    jpegli_start_compress(&cinfo_, TRUE);

    while (cinfo_.next_scanline < cinfo_.image_height) {
      size_t lines_left = cinfo_.image_height - cinfo_.next_scanline;
      size_t num_lines = std::min(batch_lines_, lines_left);
      // We use batch buffer both for temporary storage of input RGBA and pixel
      // data passed to encoder.
      if (!ReadBatch(cinfo_.next_scanline, num_lines,
                     reinterpret_cast<jint*>(batch.get()), batch.get())) {
        RETURN_ERROR(INTERNAL);
      }
      size_t lines_done = 0;
      while (lines_done < num_lines) {
        lines_done += jpegli_write_scanlines(
            &cinfo_, batch_rows.get() + lines_done, num_lines - lines_done);
        if (dest_.has_error) {
          RETURN_ERROR(INTERNAL);
        }
      }
    }

    jpegli_finish_compress(&cinfo_);
    if (dest_.has_error) {
      RETURN_ERROR(INTERNAL);
    }

    dest_.WriteAndRewind();
    if (dest_.has_error) {
      RETURN_ERROR(INTERNAL);
    }

    return OK;
  }

 private:
  bool ReadBatch(size_t y0, size_t num_lines, jint* tmp_buffer,
                 uint8_t* buffer) {
    size_t num_pixels = num_lines * width_;
    jenv_->GetIntArrayRegion(input_, y0 * width_, num_pixels, tmp_buffer);
    if (jenv_->ExceptionCheck()) return false;
    // TODO(eustas): speedup
    for (size_t i = 0; i < num_pixels; ++i) {
      // TODO(eustas): take care of endianness.
      memcpy(buffer + 3 * i, tmp_buffer + i, 4);
      // Convert BGRA input data to RGB.
      std::swap(buffer[3 * i], buffer[3 * i + 2]);
    }
    return true;
  }

  // Interface
  bool healthy_;
  JNIEnv* jenv_;
  size_t width_;
  size_t height_;
  size_t quality_;
  // TODO(eustas): make configurable
  size_t v_sampling_[3] = {1, 1, 1};
  // TODO(eustas): depends on v_sampling_; make configurable
  size_t batch_lines_ = 32;
  size_t output_buffer_size_;
  jintArray input_;

  // Jpegli encoder
  jpeg_compress_struct cinfo_ = {};
  jpeg_error_mgr err_;
  jmp_buf env_;
  DestinationManager dest_;
};

char* kEncodeName = const_cast<char*>("nativeEncode");
char* kEncodeSig =
    const_cast<char*>("(III[ILjava/nio/channels/WritableByteChannel;)I");

const JNINativeMethod kEncoderMethods[] = {
    {kEncodeName, kEncodeSig,
     reinterpret_cast<void*>(
         Java_org_jpeg_jpegli_wrapper_Encoder_nativeEncode)}};

static const size_t kNumEncoderMethods = 1;

}  // namespace

jint JniRegister(JavaVM* vm) {
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION) != JNI_OK) {
    return JNI_ERR;
  }

  jclass localClassRef = env->FindClass("org/jpeg/jpegli/wrapper/EncoderJni");
  if (localClassRef == nullptr || env->ExceptionCheck()) {
    return JNI_ERR;
  }
  if (env->RegisterNatives(localClassRef, kEncoderMethods, kNumEncoderMethods) <
      0) {
    return JNI_ERR;
  }
  env->DeleteLocalRef(localClassRef);
  return JNI_VERSION;
}

}  // namespace org_jpeg_jpegli_wrapper

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNICALL
Java_org_jpeg_jpegli_wrapper_Encoder_nativeInit(JNIEnv* env, jobject /*jobj*/) {
  using org_jpeg_jpegli_wrapper::ERROR_INTERNAL;
  using org_jpeg_jpegli_wrapper::JC_WritableByteChannel;
  using org_jpeg_jpegli_wrapper::JMID_WritableByteChannel_write;
  using org_jpeg_jpegli_wrapper::OK;

  jclass localClassRef =
      env->FindClass("java/nio/channels/WritableByteChannel");
  if (localClassRef == nullptr || env->ExceptionCheck()) {
    RETURN_ERROR(INTERNAL);
  }
  JC_WritableByteChannel = (jclass)env->NewGlobalRef(localClassRef);
  if (JC_WritableByteChannel == nullptr || env->ExceptionCheck()) {
    RETURN_ERROR(INTERNAL);
  }
  env->DeleteLocalRef(localClassRef);

  JMID_WritableByteChannel_write = env->GetMethodID(
      JC_WritableByteChannel, "write", "(Ljava/nio/ByteBuffer;)I");
  if (JMID_WritableByteChannel_write == nullptr || env->ExceptionCheck()) {
    RETURN_ERROR(INTERNAL);
  }

  return OK;
}

JNIEXPORT jint JNICALL Java_org_jpeg_jpegli_wrapper_Encoder_nativeEncode(
    JNIEnv* env, jobject /*jobj*/, jint width, jint height, jintArray config,
    jintArray input, jobject output) {
  using org_jpeg_jpegli_wrapper::Config;
  using org_jpeg_jpegli_wrapper::Encoder;
  using org_jpeg_jpegli_wrapper::ERROR_ALLOCATION;
  using org_jpeg_jpegli_wrapper::ERROR_INVALID_PARAMS;

  Config config_values;
  env->GetIntArrayRegion(config, 0, 33, config_values.data());
  if (env->ExceptionCheck()) {
    RETURN_ERROR(INVALID_PARAMS);
  }

  std::unique_ptr<Encoder> encoder{new (std::nothrow) Encoder(
      env, width, height, config_values, input, output)};
  if (!encoder) RETURN_ERROR(ALLOCATION);
  return encoder->Run();
}

#ifdef __cplusplus
}
#endif
