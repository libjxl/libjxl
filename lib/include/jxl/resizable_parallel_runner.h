/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_threads
 * @{
 * @file resizable_parallel_runner.h
 * @brief implementation using std::thread of a resizeable ::JxlParallelRunner.
 */

/** Implementation of JxlParallelRunner than can be used to enable
 * multithreading when using the JPEG XL library. This uses std::thread
 * internally and related synchronization functions. The number of threads
 * created can be changed after creation of the thread pool; the threads
 * (including the main thread) are re-used for every
 * ResizableParallelRunner::Runner call. Only one concurrent
 * JxlResizableParallelRunner call per instance is allowed at a time.
 *
 * This is a scalable, lower-overhead thread pool runner, especially suitable
 * for data-parallel computations in the fork-join model, where clients need to
 * know when all tasks have completed.
 *
 * Compared to the implementation in @ref thread_parallel_runner.h, this
 * implementation is tuned for execution on lower-powered systems, including
 * for example ARM CPUs with big.LITTLE computation models.
 */

#if !defined(JXL_RESIZABLE_PARALLEL_RUNNER_H_) || defined(CUSTOM_GENERATOR)

#ifndef CUSTOM_GENERATOR
#   ifndef DOC_GENERATOR
#       define JXL_RESIZABLE_PARALLEL_RUNNER_H_
#       include <stddef.h>
#       include <stdint.h>
#       include <stdio.h>
#       include <stdlib.h>
#       include "jxl/jxl_threads_export.h"
#       include "jxl/memory_manager.h"
#       include "jxl/parallel_runner.h"
#       define CLEAR_GENERATOR true
#       include "typebuilder/type_generator.h"
#   else
    ESCAPE(#ifndef JXL_RESIZABLE_PARALLEL_RUNNER_H_)
    ESCAPE(#define JXL_RESIZABLE_PARALLEL_RUNNER_H_)
    ESCAPE(#include <stddef.h>)
    ESCAPE(#include <stdint.h>)
    ESCAPE(#include <stdio.h>)
    ESCAPE(#include <stdlib.h>)
    ESCAPE(#include "jxl/jxl_threads_export.h")
    ESCAPE(#include "jxl/memory_manager.h")
    ESCAPE(#include "jxl/parallel_runner.h")
#   endif
#endif

EXTERN_C(

    /** Parallel runner internally using std::thread. Use as JxlParallelRunner.
     */
    Threads_Export Method(JxlParallelRetCode, JxlResizableParallelRunner,
        (
            void* runner_opaque,
            void* jpegxl_opaque,
            JxlParallelRunInit init,
            JxlParallelRunFunction func,
            uint32_t start_range,
            uint32_t end_range
        )
    )

    /** Creates the runner for JxlResizableParallelRunner. Use as the opaque
     * runner. The runner will execute tasks on the calling thread until
     * @ref JxlResizableParallelRunnerSetThreads is called.
     */
    Threads_Export Method(void*, JxlResizableParallelRunnerCreate,
        (const JxlMemoryManager* memory_manager)
    )

    /** Changes the number of threads for JxlResizableParallelRunner.
     */
    Threads_Export Method(void, JxlResizableParallelRunnerSetThreads,
        (void* runner_opaque, size_t num_threads)
    )

    /** Suggests a number of threads to use for an image of given size.
     */
    Threads_Export Method(uint32_t, JxlResizableParallelRunnerSuggestThreads,
        (uint64_t xsize, uint64_t ysize)
    )

    /** Destroys the runner created by JxlResizableParallelRunnerCreate.
     */
    Threads_Export Method(void, JxlResizableParallelRunnerDestroy, (void* runner_opaque))
)

#if CLEAR_GENERATOR
#   undef CLEAR_GENERATOR
#   include "typebuilder/clear_generator.h"
#endif

#ifdef DOC_GENERATOR
    ESCAPE(#endif)
#endif

#endif /* JXL_RESIZABLE_PARALLEL_RUNNER_H_ */

/** @}*/
