/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_common
 * @{
 * @file memory_manager.h
 * @brief Abstraction functions used by JPEG XL to allocate memory.
 */

#if !defined(JXL_MEMORY_MANAGER_H_) || defined(CUSTOM_GENERATOR)

#ifndef CUSTOM_GENERATOR
#	ifndef DOC_GENERATOR
#		define JXL_MEMORY_MANAGER_H_
#		include <stddef.h>
#		define CLEAR_GENERATOR true
#		include "typebuilder/type_generator.h"
#	else
	ESCAPE(#ifndef JXL_MEMORY_MANAGER_H_)
	ESCAPE(#define JXL_MEMORY_MANAGER_H_)
	ESCAPE(#include <stddef.h>)
#	endif
#endif

EXTERN_C(
	/**
	 * Allocating function for a memory region of a given size.
	 *
	 * Allocates a contiguous memory region of size @p size bytes. The returned
	 * memory may not be aligned to a specific size or initialized at all.
	 *
	 * @param opaque custom memory manager handle provided by the caller.
	 * @param size in bytes of the requested memory region.
	 * @return @c NULL if the memory can not be allocated,
	 * @return pointer to the memory otherwise.
	 */
	Delegate(void*, jpegxl_alloc_func, (void* opaque, size_t size))

	/**
	 * Deallocating function pointer type.
	 *
	 * This function @b MUST do nothing if @p address is @c NULL.
	 *
	 * @param opaque custom memory manager handle provided by the caller.
	 * @param address memory region pointer returned by ::jpegxl_alloc_func, or @c
	 * NULL.
	 */
	Delegate(void, jpegxl_free_func, (void* opaque, void* address))

	/**
	 * Memory Manager struct.
	 * These functions, when provided by the caller, will be used to handle memory
	 * allocations.
	 */
	StructDef2(JxlMemoryManager, JxlMemoryManagerStruct,
		/** The opaque pointer that will be passed as the first parameter to all the
		 * functions in this struct. */
		Member(void*, opaque)

		/** Memory allocation function. This can be NULL if and only if also the
		 * free() member in this class is NULL. All dynamic memory will be allocated
		 * and freed with these functions if they are not NULL. */
		Member(jpegxl_alloc_func, alloc)
		
		/** Free function matching the alloc() member. */
		Member(jpegxl_free_func, free)

		/* TODO(deymo): Add cache-aligned alloc/free functions here. */
	)
)

#if CLEAR_GENERATOR
#	undef CLEAR_GENERATOR
#	include "typebuilder/clear_generator.h"
#endif

#ifdef DOC_GENERATOR
	ESCAPE(#endif)
#endif

#endif /* JXL_MEMORY_MANAGER_H_ */

/** @}*/
