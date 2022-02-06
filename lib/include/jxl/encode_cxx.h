// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/** @addtogroup libjxl_encoder
 * @{
 * 
 *  @file encode_cxx.h
 *  @brief C++ header-only helper for @ref encode.h.
 * 
 *  There's no binary library associated with the header since this is a header
 *  only library.
 */

#if !defined(JXL_ENCODE_CXX_H_) || defined(CUSTOM_GENERATOR)
#ifndef CUSTOM_GENERATOR
#	ifndef DOC_GENERATOR
#		define JXL_ENCODE_CXX_H_
#		include <memory>
#		include "jxl/encode.h"
#		if !(defined(__cplusplus) || defined(c_plusplus))
#			error "This a C++ only header. Use jxl/encode.h from C sources."
#		endif
#		define CLEAR_GENERATOR true
#		include "typebuilder/type_generator.h"
#	else
	ESCAPE(#ifndef JXL_ENCODE_CXX_H_)
	ESCAPE(#define JXL_ENCODE_CXX_H_)
	ESCAPE(#include <memory>)
	ESCAPE(#include "jxl/encode.h")
	ESCAPE(#if !(defined(__cplusplus) || defined(c_plusplus)))
		ESCAPE(#error "This a C++ only header. Use jxl/encode.h from C sources.")
	ESCAPE(#endif)
#	endif
#endif

/** Struct to call JxlEncoderDestroy from the JxlEncoderPtr unique_ptr.
 */
Struct(JxlEncoderDestroyStruct,
	RawCode(
		/** Calls @ref JxlEncoderDestroy() on the passed encoder.
		 */
  		void operator()(JxlEncoder* encoder) { JxlEncoderDestroy(encoder); }
	)
)

/** std::unique_ptr<> type that calls JxlEncoderDestroy() when releasing the
 *  encoder.
 * 
 *  Use this helper type from C++ sources to ensure the encoder is destroyed and
 *  their internal resources released.
 */
TypeDef(JxlEncoderPtr, std::unique_ptr<JxlEncoder, JxlEncoderDestroyStruct>)

/** Creates an instance of JxlEncoder into a JxlEncoderPtr and initializes it.
 * 
 *  This function returns a unique_ptr that will call JxlEncoderDestroy() when
 *  releasing the pointer. See @ref JxlEncoderCreate for details on the
 *  instance creation.
 * 
 *  @param memory_manager custom allocator function. It may be NULL. The memory
 *         manager will be copied internally.
 *  @return a @c NULL JxlEncoderPtr if the instance can not be allocated or
 *          initialized
 *  @return initialized JxlEncoderPtr instance otherwise.
 */
Static Inline BodyMethod(JxlEncoderPtr, JxlEncoderMake, (const JxlMemoryManager* memory_manager),
  return JxlEncoderPtr(JxlEncoderCreate(memory_manager));
)

#if CLEAR_GENERATOR
#	include "typebuilder/clear_generator.h"
#endif

#ifdef DOC_GENERATOR
	ESCAPE(#endif)
#endif

#endif  /* JXL_ENCODE_CXX_H_*/

/** @}*/
