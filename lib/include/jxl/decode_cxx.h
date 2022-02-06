/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_decoder
 * @{
 *
 * @file decode_cxx.h
 * @brief C++ header-only helper for @ref decode.h.
 *
 * There's no binary library associated with the header since this is a header
 * only library.
 */
#if !defined(JXL_DECODE_CXX_H_) || defined(CUSTOM_GENERATOR)

#ifndef CUSTOM_GENERATOR
#	ifndef DOC_GENERATOR
#		define JXL_DECODE_CXX_H_
#		include <memory>
#		include "jxl/decode.h"
#		if !(defined(__cplusplus) || defined(c_plusplus))
#			error "This a C++ only header. Use jxl/decode.h from C sources."
#		endif
#		define CLEAR_GENERATOR true
#		include "typebuilder/type_generator.h"
#	else
	ESCAPE(#ifndef JXL_DECODE_CXX_H_)
	ESCAPE(#define JXL_DECODE_CXX_H_)
	ESCAPE(#include <memory>)
	ESCAPE(#include "jxl/decode.h")
	ESCAPE(#if !(defined(__cplusplus) || defined(c_plusplus)))
	ESCAPE(#error "This a C++ only header. Use jxl/decode.h from C sources.")
	ESCAPE(#endif)
#	endif
#endif


/** Struct to call JxlDecoderDestroy from the JxlDecoderPtr unique_ptr.
 */
Struct(JxlDecoderDestroyStruct,
	RawCode(
  		/** Calls @ref JxlDecoderDestroy() on the passed decoder.
		  */
		void operator()(JxlDecoder* decoder) { JxlDecoderDestroy(decoder); }
	)
)

/** std::unique_ptr<> type that calls JxlDecoderDestroy() when releasing the
 * decoder.
 *
 * Use this helper type from C++ sources to ensure the decoder is destroyed and
 * their internal resources released.
 */
TypeDef(JxlDecoderPtr, std::unique_ptr<JxlDecoder, JxlDecoderDestroyStruct>)

/** Creates an instance of JxlDecoder into a JxlDecoderPtr and initializes it.
 *
 * This function returns a unique_ptr that will call JxlDecoderDestroy() when
 * releasing the pointer. See @ref JxlDecoderCreate for details on the
 * instance creation.
 *
 * @param memory_manager custom allocator function. It may be NULL. The memory
 *        manager will be copied internally.
 * @return a @c NULL JxlDecoderPtr if the instance can not be allocated or
 *         initialized
 * @return initialized JxlDecoderPtr instance otherwise.
 */
Static Inline BodyMethod(JxlDecoderPtr, JxlDecoderMake, (const JxlMemoryManager* memory_manager),
	return JxlDecoderPtr(JxlDecoderCreate(memory_manager));
)

#if CLEAR_GENERATOR
#	include "typebuilder/clear_generator.h"
#endif

#ifdef DOC_GENERATOR
	ESCAPE(#endif)
#endif

#endif  /* JXL_DECODE_CXX_H_ */

/** @}*/
