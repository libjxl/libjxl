/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_threads
 * @{
 *
 * @file resizable_parallel_runner_cxx.h
 * @ingroup libjxl_threads
 * @brief C++ header-only helper for @ref resizable_parallel_runner.h.
 *
 * There's no binary library associated with the header since this is a header
 * only library.
 */

#if !defined(JXL_RESIZABLE_PARALLEL_RUNNER_CXX_H_) || defined(CUSTOM_GENERATOR)

#ifndef CUSTOM_GENERATOR
#	ifndef DOC_GENERATOR
#		include <memory>
#		include "jxl/resizable_parallel_runner.h"
#		if !(defined(__cplusplus) || defined(c_plusplus))
#			error \
    			"This a C++ only header. Use jxl/jxl_resizable_parallel_runner.h from C" \
    			"sources."
#		endif
#		define CLEAR_GENERATOR true
#		include "typebuilder/type_generator.h"
#	else
	ESCAPE(#ifndef JXL_RESIZABLE_PARALLEL_RUNNER_CXX_H_)
	ESCAPE(#if !(defined(__cplusplus) || defined(c_plusplus)))
	ESCAPE(#error \)
	ESCAPE("This a C++ only header. Use jxl/jxl_resizable_parallel_runner.h from C" \)
	ESCAPE("sources.")
	ESCAPE(#endif)
	ESCAPE(#define JXL_RESIZABLE_PARALLEL_RUNNER_CXX_H_)
	ESCAPE(#include <memory>)
	ESCAPE(#include "jxl/resizable_parallel_runner.h")
#	endif
#endif

/** Struct to call JxlResizableParallelRunnerDestroy from the
 * JxlResizableParallelRunnerPtr unique_ptr.
 */
Struct(JxlResizableParallelRunnerDestroyStruct,
	RawCode(
		/** Calls @ref JxlResizableParallelRunnerDestroy() on the passed runner.
		 */
  		void operator()(void* runner) { JxlResizableParallelRunnerDestroy(runner); }
	)
)

/** std::unique_ptr<> type that calls JxlResizableParallelRunnerDestroy() when
 * releasing the runner.
 *
 * Use this helper type from C++ sources to ensure the runner is destroyed and
 * their internal resources released.
 */
TypeDef(JxlResizableParallelRunnerPtr, std::unique_ptr<void, JxlResizableParallelRunnerDestroyStruct>)

/** Creates an instance of JxlResizableParallelRunner into a
 * JxlResizableParallelRunnerPtr and initializes it.
 *
 * This function returns a unique_ptr that will call
 * JxlResizableParallelRunnerDestroy() when releasing the pointer. See @ref
 * JxlResizableParallelRunnerCreate for details on the instance creation.
 *
 * @param memory_manager custom allocator function. It may be NULL. The memory
 *        manager will be copied internally.
 * @return a @c NULL JxlResizableParallelRunnerPtr if the instance can not be
 * allocated or initialized
 * @return initialized JxlResizableParallelRunnerPtr instance otherwise.
 */
Static Inline BodyMethod(JxlResizableParallelRunnerPtr, JxlResizableParallelRunnerMake, (const JxlMemoryManager* memory_manager),
  return JxlResizableParallelRunnerPtr(JxlResizableParallelRunnerCreate(memory_manager));
)

#if CLEAR_GENERATOR
#	undef CLEAR_GENERATOR
#	include "typebuilder/clear_generator.h"
#endif

#ifdef DOC_GENERATOR
	ESCAPE(#endif)
#endif

#endif  /* JXL_RESIZABLE_PARALLEL_RUNNER_CXX_H_ */

/** @}*/
