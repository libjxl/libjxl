// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/** @addtogroup libjxl_butteraugli
 * @{
 *
 * @file butteraugli_cxx.h
 * @brief C++ header-only helper for @ref butteraugli.h.
 *
 * There's no binary library associated with the header since this is a header
 * only library.
 */

#if !defined(JXL_BUTTERAUGLI_CXX_H_) || defined(CUSTOM_GENERATOR)

#ifndef CUSTOM_GENERATOR
# 	ifndef DOC_GENERATOR
#		define JXL_BUTTERAUGLI_CXX_H_
#		include <memory>
#		include "jxl/butteraugli.h"
#		if !(defined(__cplusplus) || defined(c_plusplus))
#			error "This a C++ only header. Use jxl/butteraugli.h from C sources."
#		endif
#		define CLEAR_GENERATOR true
#		include "typebuilder/type_generator.h"
#	else
		ESCAPE(#ifndef JXL_BUTTERAUGLI_CXX_H_)
		ESCAPE(#define JXL_BUTTERAUGLI_CXX_H_)
		ESCAPE(#include <memory>)
		ESCAPE(#include "jxl/butteraugli.h")
#	endif
#endif

/** Struct to call JxlButteraugliApiDestroy from the JxlButteraugliApiPtr
 * unique_ptr.
 */
Struct(JxlButteraugliApiDestroyStruct,
	RawCode(
	  	/** Calls @ref JxlButteraugliApiDestroy() on the passed api.
		 */
  		void operator()(JxlButteraugliApi* api) { JxlButteraugliApiDestroy(api); }
	)
)

/** std::unique_ptr<> type that calls JxlButteraugliApiDestroy() when releasing
 * the pointer.
 *
 * Use this helper type from C++ sources to ensure the api is destroyed and
 * their internal resources released.
 */
Type(JxlButteraugliApiPtr, std::unique_ptr<JxlButteraugliApi, JxlButteraugliApiDestroyStruct>)


/** Struct to call JxlButteraugliResultDestroy from the JxlButteraugliResultPtr
 * unique_ptr.
 */
Struct(JxlButteraugliResultDestroyStruct,
  RawCode(
    /** Calls @ref JxlButteraugliResultDestroy() on the passed result object.
	 */
    void operator()(JxlButteraugliResult* result) {
      JxlButteraugliResultDestroy(result);
    }
  )
)

/** std::unique_ptr<> type that calls JxlButteraugliResultDestroy() when
 * releasing the pointer.
 *
 * Use this helper type from C++ sources to ensure the result object is
 * destroyed and their internal resources released.
 */
Type(JxlButteraugliResultPtr, std::unique_ptr<JxlButteraugliResult, JxlButteraugliResultDestroyStruct>)


#if CLEAR_GENERATOR
#	undef CLEAR_GENERATOR
#	include "typebuilder/clear_generator.h"
#endif

#ifdef DOC_GENERATOR
    ESCAPE(#endif)
#endif

#endif  /* JXL_BUTTERAUGLI_CXX_H_ */

/** @}*/
