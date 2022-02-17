/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_threads
 * @{
 * @file thread_parallel_runner.h
 * @brief implementation using std::thread of a ::JxlParallelRunner.
 */

/** Implementation of JxlParallelRunner than can be used to enable
 * multithreading when using the JPEG XL library. This uses std::thread
 * internally and related synchronization functions. The number of threads
 * created is fixed at construction time and the threads are re-used for every
 * ThreadParallelRunner::Runner call. Only one concurrent
 * JxlThreadParallelRunner call per instance is allowed at a time.
 *
 * This is a scalable, lower-overhead thread pool runner, especially suitable
 * for data-parallel computations in the fork-join model, where clients need to
 * know when all tasks have completed.
 *
 * This thread pool can efficiently load-balance millions of tasks using an
 * atomic counter, thus avoiding per-task virtual or system calls. With 48
 * hyperthreads and 1M tasks that add to an atomic counter, overall runtime is
 * 10-20x higher when using std::async, and ~200x for a queue-based thread
 */


#if !defined(JXL_THREAD_PARALLEL_RUNNER_H_) || defined(CUSTOM_GENERATOR)

#ifndef CUSTOM_GENERATOR
#   ifndef DOC_GENERATOR
#       define JXL_THREAD_PARALLEL_RUNNER_H_
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
    ESCAPE(#ifndef JXL_THREAD_PARALLEL_RUNNER_H_)
    ESCAPE(#define JXL_THREAD_PARALLEL_RUNNER_H_)
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
    Threads_Export Method(JxlParallelRetCode, JxlThreadParallelRunner,
        (
            void* runner_opaque,
            void* jpegxl_opaque,
            JxlParallelRunInit init,
            JxlParallelRunFunction func,
            uint32_t start_range,
            uint32_t end_range
        )
    )

    /** Creates the runner for JxlThreadParallelRunner. Use as the opaque
     * runner.
     */
    Threads_Export Method(void*, JxlThreadParallelRunnerCreate,
        (const JxlMemoryManager* memory_manager, size_t num_worker_threads)
    )

    /** Destroys the runner created by JxlThreadParallelRunnerCreate.
     */
    Threads_Export Method(void, JxlThreadParallelRunnerDestroy, (void* runner_opaque))

    /** Returns a default num_worker_threads value for
     * JxlThreadParallelRunnerCreate.
     */
    Threads_Export Method(size_t, JxlThreadParallelRunnerDefaultNumWorkerThreads, ())
)

#if CLEAR_GENERATOR
#   undef CLEAR_GENERATOR
#   include "typebuilder/clear_generator.h"
#endif

#ifdef DOC_GENERATOR
    ESCAPE(#endif)
#endif

#endif /* JXL_THREAD_PARALLEL_RUNNER_H_ */

/** @}*/
