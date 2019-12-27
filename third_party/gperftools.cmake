# Copyright (c) the JPEG XL Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(gperftools STATIC
  gperftools/src/addressmap-inl.h
  gperftools/src/base/atomicops-internals-arm-generic.h
  gperftools/src/base/atomicops-internals-arm-v6plus.h
  gperftools/src/base/atomicops-internals-gcc.h
  gperftools/src/base/atomicops-internals-linuxppc.h
  gperftools/src/base/atomicops-internals-macosx.h
  gperftools/src/base/atomicops-internals-mips.h
  gperftools/src/base/atomicops-internals-windows.h
  gperftools/src/base/atomicops-internals-x86.h
  gperftools/src/base/atomicops.h
  gperftools/src/base/basictypes.h
  gperftools/src/base/commandlineflags.h
  gperftools/src/base/dynamic_annotations.c
  gperftools/src/base/dynamic_annotations.h
  gperftools/src/base/elfcore.h
  gperftools/src/base/googleinit.h
  gperftools/src/base/linux_syscall_support.h
  gperftools/src/base/linuxthreads.cc
  gperftools/src/base/linuxthreads.h
  gperftools/src/base/logging.cc
  gperftools/src/base/logging.h
  gperftools/src/base/low_level_alloc.cc
  gperftools/src/base/spinlock.cc
  gperftools/src/base/spinlock.h
  gperftools/src/base/spinlock_internal.cc
  gperftools/src/base/spinlock_internal.h
  gperftools/src/base/spinlock_linux-inl.h
  gperftools/src/base/spinlock_posix-inl.h
  gperftools/src/base/spinlock_win32-inl.h
  gperftools/src/base/stl_allocator.h
  gperftools/src/base/sysinfo.cc
  gperftools/src/base/sysinfo.h
  gperftools/src/base/thread_annotations.h
  gperftools/src/base/thread_lister.c
  gperftools/src/base/thread_lister.h
  gperftools/src/central_freelist.cc
  gperftools/src/central_freelist.h
  gperftools/src/common.cc
  gperftools/src/common.h
  gperftools/src/emergency_malloc.cc
  gperftools/src/emergency_malloc.h
  gperftools/src/emergency_malloc_for_stacktrace.cc
  gperftools/src/fake_stacktrace_scope.cc
  gperftools/src/gperftools/heap-checker.h
  gperftools/src/gperftools/heap-profiler.h
  gperftools/src/gperftools/malloc_extension.h
  gperftools/src/gperftools/malloc_extension_c.h
  gperftools/src/gperftools/malloc_hook.h
  gperftools/src/gperftools/malloc_hook_c.h
  gperftools/src/gperftools/nallocx.h
  gperftools/src/stacktrace.cc
  gperftools/src/gperftools/stacktrace.h
  gperftools/src/heap-checker-bcad.cc
  gperftools/src/heap-checker.cc
  gperftools/src/heap-profile-stats.h
  gperftools/src/heap-profile-table.cc
  gperftools/src/heap-profile-table.h
  gperftools/src/heap-profiler.cc
  gperftools/src/internal_logging.cc
  gperftools/src/internal_logging.h
  gperftools/src/libc_override.h
  gperftools/src/libc_override_gcc_and_weak.h
  gperftools/src/libc_override_glibc.h
  gperftools/src/libc_override_osx.h
  gperftools/src/libc_override_redefine.h
  gperftools/src/linked_list.h
  gperftools/src/malloc_extension.cc
  gperftools/src/malloc_hook-inl.h
  gperftools/src/malloc_hook.cc
  gperftools/src/malloc_hook_mmap_freebsd.h
  gperftools/src/malloc_hook_mmap_linux.h
  gperftools/src/maybe_emergency_malloc.h
  gperftools/src/maybe_threads.cc
  gperftools/src/maybe_threads.h
  gperftools/src/memfs_malloc.cc
  gperftools/src/memory_region_map.cc
  gperftools/src/packed-cache-inl.h
  gperftools/src/page_heap.cc
  gperftools/src/page_heap.h
  gperftools/src/page_heap_allocator.h
  gperftools/src/pagemap.h
  gperftools/src/raw_printer.cc
  gperftools/src/raw_printer.h
  gperftools/src/sampler.cc
  gperftools/src/sampler.h
  gperftools/src/span.cc
  gperftools/src/span.h
  gperftools/src/stack_trace_table.cc
  gperftools/src/stack_trace_table.h
  gperftools/src/static_vars.cc
  gperftools/src/static_vars.h
  gperftools/src/symbolize.cc
  gperftools/src/symbolize.h
  gperftools/src/system-alloc.cc
  gperftools/src/system-alloc.h
  gperftools/src/tcmalloc.cc
  gperftools/src/tcmalloc_guard.h
  gperftools/src/third_party/valgrind.h
  gperftools/src/thread_cache.cc
  gperftools/src/thread_cache.hgperftools/
)
target_include_directories(gperftools
  PRIVATE "${CMAKE_CURRENT_LIST_DIR}/gperftools/src"
  PRIVATE "${CMAKE_CURRENT_LIST_DIR}/gperftools_config")

# Force every translation unit to include the fixup header.
target_compile_options(gperftools
    PRIVATE -include "${CMAKE_CURRENT_LIST_DIR}/gperftools_fixes.h" -Wall -Werror)

set_source_files_properties(gperftools/src/tcmalloc.cc PROPERTIES
    COMPILE_FLAGS "-include ${CMAKE_CURRENT_LIST_DIR}/gperftools_fixes.h -Wno-unused-function")

set_property(TARGET gperftools PROPERTY POSITION_INDEPENDENT_CODE ON)
