/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_butteraugli
 * @{
 * @file butteraugli.h
 * @brief Butteraugli API for JPEG XL.
 */

#if !defined(JXL_BUTTERAUGLI_H_) || defined(CUSTOM_GENERATOR)
#ifndef CUSTOM_GENERATOR
#   ifndef DOC_GENERATOR
#	    define JXL_BUTTERAUGLI_H_
#       include "jxl/jxl_export.h"
#       include "jxl/memory_manager.h"
#       include "jxl/parallel_runner.h"
#       include "jxl/types.h"
#       define CLEAR_GENERATOR true
#       include "typebuilder/type_generator.h"
#   else
		ESCAPE(#ifndef JXL_BUTTERAUGLI_H_)
        ESCAPE(#define JXL_BUTTERAUGLI_H_)
        ESCAPE(#include "jxl/jxl_export.h")
        ESCAPE(#include "jxl/memory_manager.h")
        ESCAPE(#include "jxl/parallel_runner.h")
        ESCAPE(#include "jxl/types.h")
#   endif
#endif

EXTERN_C(
    /**
     * Opaque structure that holds a butteraugli API.
     *
     * Allocated and initialized with JxlButteraugliApiCreate().
     * Cleaned up and deallocated with JxlButteraugliApiDestroy().
     */
    Type(JxlButteraugliApi, struct JxlButteraugliApiStruct)

    /**
     * Opaque structure that holds intermediary butteraugli results.
     *
     * Allocated and initialized with JxlButteraugliCompute().
     * Cleaned up and deallocated with JxlButteraugliResultDestroy().
     */
    Type(JxlButteraugliResult, struct JxlButteraugliResultStruct)

    /**
     * Deinitializes and frees JxlButteraugliResult instance.
     *
     * @param result instance to be cleaned up and deallocated.
     */
    Export Method(void, JxlButteraugliResultDestroy, (JxlButteraugliResult* result))

    /**
     * Creates an instance of JxlButteraugliApi and initializes it.
     *
     * @p memory_manager will be used for all the library dynamic allocations made
     * from this instance. The parameter may be NULL, in which case the default
     * allocator will be used. See jxl/memory_manager.h for details.
     *
     * @param memory_manager custom allocator function. It may be NULL. The memory
     *        manager will be copied internally.
     * @return @c NULL if the instance can not be allocated or initialized
     * @return pointer to initialized JxlEncoder otherwise
     */
    Export Method(JxlButteraugliApi*, JxlButteraugliApiCreate, (const JxlMemoryManager* memory_manager))

    /**
     * Set the parallel runner for multithreading.
     *
     * @param api api instance.
     * @param parallel_runner function pointer to runner for multithreading. A
     * multithreaded runner should be set to reach fast performance.
     * @param parallel_runner_opaque opaque pointer for parallel_runner.
     */
    Export Method(void, JxlButteraugliApiSetParallelRunner, (JxlButteraugliApi* api, JxlParallelRunner parallel_runner, void* parallel_runner_opaque))

    /**
     * Set the hf_asymmetry option for butteraugli.
     *
     * @param api api instance.
     * @param v new hf_asymmetry value.
     */
    Export Method(void, JxlButteraugliApiSetHFAsymmetry, (JxlButteraugliApi* api, float v))

    /**
     * Set the intensity_target option for butteraugli.
     *
     * @param api api instance.
     * @param v new intensity_target value.
     */
    Export Method(void, JxlButteraugliApiSetIntensityTarget, (JxlButteraugliApi* api, float v))

    /**
     * Deinitializes and frees JxlButteraugliApi instance.
     *
     * @param api instance to be cleaned up and deallocated.
     */
    Export Method(void, JxlButteraugliApiDestroy, (JxlButteraugliApi* api))

    /**
     * Computes intermediary butteraugli result between an original image and a
     * distortion.
     *
     * @param api api instance for this computation.
     * @param xsize width of the compared images.
     * @param ysize height of the compared images.
     * @param pixel_format_orig pixel format for original image.
     * @param buffer_orig pixel data for original image.
     * @param size_orig size of buffer_orig in bytes.
     * @param pixel_format_dist pixel format for distortion.
     * @param buffer_dist pixel data for distortion.
     * @param size_dist size of buffer_dist in bytes.
     * @return @c NULL if the results can not be computed or initialized.
     * @return pointer to initialized and computed intermediary result.
     */
    Export Method(JxlButteraugliResult*, JxlButteraugliCompute,
        (
            const JxlButteraugliApi* api,
            uint32_t xsize,
            uint32_t ysize,
            const JxlPixelFormat* pixel_format_orig,
            const void* buffer_orig,
            size_t size_orig,
            const JxlPixelFormat* pixel_format_dist,
            const void* buffer_dist,
            size_t size_dist
        )
    )

    /**
     * Computes butteraugli max distance based on an intermediary butteraugli
     * result.
     *
     * @param result intermediary result instance.
     * @return max distance.
     */
    Export Method(float, JxlButteraugliResultGetMaxDistance, (const JxlButteraugliResult* result))

    /**
     * Computes a butteraugli distance based on an intermediary butteraugli result.
     *
     * @param result intermediary result instance.
     * @param pnorm pnorm to calculate.
     * @return distance using the given pnorm.
     */
    Export Method(float, JxlButteraugliResultGetDistance, (const JxlButteraugliResult* result, float pnorm))

    /**
     * Get a pointer to the distmap in the result.
     *
     * @param result intermediary result instance.
     * @param buffer will be set to the distmap. The distance value for (x,y) will
     * be available at buffer + y * row_stride + x.
     * @param row_stride will be set to the row stride of the distmap.
     */
    Export Method(void, JxlButteraugliResultGetDistmap,
        (
            const JxlButteraugliResult* result,
            const float** buffer,
            uint32_t* row_stride
        )
    )
)

#if CLEAR_GENERATOR
#   undef CLEAR_GENERATOR
#   include "typebuilder/clear_generator.h"
#endif

#ifdef DOC_GENERATOR
    ESCAPE(#endif)
#endif

#endif /* JXL_BUTTERAUGLI_H_ */

/** @}*/
