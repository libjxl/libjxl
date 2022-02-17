/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */


/** @addtogroup libjxl_threads
 *  @{
 * 
 *  @file thread_parallel_runner_cxx.h
 *  @brief C++ header-only helper for @ref thread_parallel_runner.h.
 * 
 *  There's no binary library associated with the header since this is a header
 *  only library.
 */

#if !defined(JXL_THREAD_PARALLEL_RUNNER_CXX_H_) || defined(CUSTOM_GENERATOR)

#ifndef CUSTOM_GENERATOR
#	ifndef DOC_GENERATOR
#		define JXL_THREAD_PARALLEL_RUNNER_CXX_H_
#		include <memory>
#		include "jxl/thread_parallel_runner.h"
#		if !(defined(__cplusplus) || defined(c_plusplus))
#		error \
    		"This a C++ only header. Use jxl/jxl_thread_parallel_runner.h from C" \
    		"sources."
#		endif
#		define CLEAR_GENERATOR true
#		include "typebuilder/type_generator.h"
#	else
	ESCAPE(#ifndef JXL_THREAD_PARALLEL_RUNNER_CXX_H_)
	ESCAPE(#define JXL_THREAD_PARALLEL_RUNNER_CXX_H_)
	ESCAPE(#include <memory>)
	ESCAPE(#include "jxl/thread_parallel_runner.h")
	ESCAPE(#if !(defined(__cplusplus) || defined(c_plusplus)))
	ESCAPE(#error \)
    ESCAPE("This a C++ only header. Use jxl/jxl_thread_parallel_runner.h from C" \)
    ESCAPE("sources.")
	ESCAPE(#endif)
#	endif
#endif



/** Struct to call JxlThreadParallelRunnerDestroy from the
 *  JxlThreadParallelRunnerPtr unique_ptr.
 */
Struct(JxlThreadParallelRunnerDestroyStruct,
	RawCode(
  /** Calls @ref JxlThreadParallelRunnerDestroy() on the passed runner.
   */
  void operator()(void* runner) { JxlThreadParallelRunnerDestroy(runner); }

	)
)

/** std::unique_ptr<> type that calls JxlThreadParallelRunnerDestroy() when
 *  releasing the runner.
 * 
 *  Use this helper type from C++ sources to ensure the runner is destroyed and
 *  their internal resources released.
 */
TypeDef(JxlThreadParallelRunnerPtr, std::unique_ptr<void, JxlThreadParallelRunnerDestroyStruct>)

/** Creates an instance of JxlThreadParallelRunner into a
 *  JxlThreadParallelRunnerPtr and initializes it.
 * 
 *  This function returns a unique_ptr that will call
 *  JxlThreadParallelRunnerDestroy() when releasing the pointer. See @ref
 *  JxlThreadParallelRunnerCreate for details on the instance creation.
 * 
 *  @param memory_manager custom allocator function. It may be NULL. The memory
 *         manager will be copied internally.
 *  @param num_worker_threads the number of worker threads to create.
 *  @return a @c NULL JxlThreadParallelRunnerPtr if the instance can not be
 *  allocated or initialized
 *  @return initialized JxlThreadParallelRunnerPtr instance otherwise.
 */
Static Inline BodyMethod(JxlThreadParallelRunnerPtr, JxlThreadParallelRunnerMake, (const JxlMemoryManager* memory_manager, size_t num_worker_threads),
  return JxlThreadParallelRunnerPtr(JxlThreadParallelRunnerCreate(memory_manager, num_worker_threads));
)

#if CLEAR_GENERATOR
#	undef CLEAR_GENERATOR
#	include "typebuilder/clear_generator.h"
#endif

#ifdef DOC_GENERATOR
	ESCAPE(#endif)
#endif

#endif  /* JXL_THREAD_PARALLEL_RUNNER_CXX_H_ */

/** @}*/
