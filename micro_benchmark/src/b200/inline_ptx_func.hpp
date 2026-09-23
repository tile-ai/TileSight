#include <cuda.h>

inline uint32_t __device__
cast_smem_ptr_to_uint(void const *const ptr)
{
    // We prefer to use the new CVTA intrinsics if they are available, otherwise we will fall back to
    // the previous internal intrinsics if they are available.
    //
    // This NVVM intrinsic converts an address in shared memory to a plain
    // unsigned integer. This is necessary to pass to shared memory instructions
    // in inline PTX.
    //
    // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only available in 10.2].
    //
    //__device__ size_t __cvta_generic_to_shared(void* ptr);

    /// CUTE helper to get SMEM pointer
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    //   uint32_t smem_ptr;

    //   asm(
    //   "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
    //     : "=r"(smem_ptr) : "l"(ptr));

    //   return smem_ptr;
}

inline void __device__ tmem_allocate(uint32_t *dst_ptr, int num_columns)
{
    uint32_t dst_intptr = cast_smem_ptr_to_uint(dst_ptr);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(dst_intptr), "r"(num_columns));
}

inline void __device__ tmem_free(uint32_t tmem_ptr, int num_columns)
{
    asm volatile(
        "{\n\t"
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1; \n\t"
        "}"
        :
        : "r"(tmem_ptr), "r"(num_columns));
}

// 32 data path lanes, 32-bit pattern, repeated N times
template <int N>
inline void __device__ tmem_st_32dp32bNx(uint32_t const &dst_addr, uint32_t const *src_ptr)
{
    static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of 2");

    if constexpr (N == 1)
    {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32"
                     "[%0],"
                     "{%1};\n"
                     :
                     : "r"(dst_addr), "r"(src_ptr[0]));
    }
    else if constexpr (N == 2)
    {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32"
                     "[%0],"
                     "{%1, %2};\n"
                     :
                     : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]));
    }
    else if constexpr (N == 4)
    {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32"
                     "[%0],"
                     "{%1, %2, %3, %4};\n"
                     :
                     : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]));
    }
    else if constexpr (N == 8)
    {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32"
                     "[%0],"
                     "{%1, %2, %3, %4, %5, %6, %7, %8};\n"
                     :
                     : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]));
    }
    else if constexpr (N == 16)
    {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32"
                     "[%0],"
                     "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                     :
                     : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]), "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]), "r"(src_ptr[14]), "r"(src_ptr[15]));
    }
    else if constexpr (N == 32)
    {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32"
                     "[%0],"
                     "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};\n"
                     :
                     : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]), "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]), "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]), "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]), "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]), "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]), "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]), "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]));
    }
    else if constexpr (N == 64)
    {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x64.b32"
                     "[%0],"
                     "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};\n"
                     :
                     : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]), "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]), "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]), "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]), "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]), "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]), "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]), "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]), "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]), "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]), "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]), "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]), "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]), "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]), "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]), "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]), "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]), "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]), "r"(src_ptr[62]), "r"(src_ptr[63]));
    }
    else if constexpr (N == 128)
    {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x128.b32"
                     "[%0],"
                     "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127, %128};\n"
                     :
                     : "r"(dst_addr), "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]), "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(src_ptr[8]), "r"(src_ptr[9]), "r"(src_ptr[10]), "r"(src_ptr[11]), "r"(src_ptr[12]), "r"(src_ptr[13]), "r"(src_ptr[14]), "r"(src_ptr[15]), "r"(src_ptr[16]), "r"(src_ptr[17]), "r"(src_ptr[18]), "r"(src_ptr[19]), "r"(src_ptr[20]), "r"(src_ptr[21]), "r"(src_ptr[22]), "r"(src_ptr[23]), "r"(src_ptr[24]), "r"(src_ptr[25]), "r"(src_ptr[26]), "r"(src_ptr[27]), "r"(src_ptr[28]), "r"(src_ptr[29]), "r"(src_ptr[30]), "r"(src_ptr[31]), "r"(src_ptr[32]), "r"(src_ptr[33]), "r"(src_ptr[34]), "r"(src_ptr[35]), "r"(src_ptr[36]), "r"(src_ptr[37]), "r"(src_ptr[38]), "r"(src_ptr[39]), "r"(src_ptr[40]), "r"(src_ptr[41]), "r"(src_ptr[42]), "r"(src_ptr[43]), "r"(src_ptr[44]), "r"(src_ptr[45]), "r"(src_ptr[46]), "r"(src_ptr[47]), "r"(src_ptr[48]), "r"(src_ptr[49]), "r"(src_ptr[50]), "r"(src_ptr[51]), "r"(src_ptr[52]), "r"(src_ptr[53]), "r"(src_ptr[54]), "r"(src_ptr[55]), "r"(src_ptr[56]), "r"(src_ptr[57]), "r"(src_ptr[58]), "r"(src_ptr[59]), "r"(src_ptr[60]), "r"(src_ptr[61]), "r"(src_ptr[62]), "r"(src_ptr[63]), "r"(src_ptr[64]), "r"(src_ptr[65]), "r"(src_ptr[66]), "r"(src_ptr[67]), "r"(src_ptr[68]), "r"(src_ptr[69]), "r"(src_ptr[70]), "r"(src_ptr[71]), "r"(src_ptr[72]), "r"(src_ptr[73]), "r"(src_ptr[74]), "r"(src_ptr[75]), "r"(src_ptr[76]), "r"(src_ptr[77]), "r"(src_ptr[78]), "r"(src_ptr[79]), "r"(src_ptr[80]), "r"(src_ptr[81]), "r"(src_ptr[82]), "r"(src_ptr[83]), "r"(src_ptr[84]), "r"(src_ptr[85]), "r"(src_ptr[86]), "r"(src_ptr[87]), "r"(src_ptr[88]), "r"(src_ptr[89]), "r"(src_ptr[90]), "r"(src_ptr[91]), "r"(src_ptr[92]), "r"(src_ptr[93]), "r"(src_ptr[94]), "r"(src_ptr[95]), "r"(src_ptr[96]), "r"(src_ptr[97]), "r"(src_ptr[98]), "r"(src_ptr[99]), "r"(src_ptr[100]), "r"(src_ptr[101]), "r"(src_ptr[102]), "r"(src_ptr[103]), "r"(src_ptr[104]), "r"(src_ptr[105]), "r"(src_ptr[106]), "r"(src_ptr[107]), "r"(src_ptr[108]), "r"(src_ptr[109]), "r"(src_ptr[110]), "r"(src_ptr[111]), "r"(src_ptr[112]), "r"(src_ptr[113]), "r"(src_ptr[114]), "r"(src_ptr[115]), "r"(src_ptr[116]), "r"(src_ptr[117]), "r"(src_ptr[118]), "r"(src_ptr[119]), "r"(src_ptr[120]), "r"(src_ptr[121]), "r"(src_ptr[122]), "r"(src_ptr[123]), "r"(src_ptr[124]), "r"(src_ptr[125]), "r"(src_ptr[126]), "r"(src_ptr[127]));
    }
}

// 32 data path lanes, 32-bit pattern, repeated N times
template <int N>
inline void __device__ tmem_ld_32dp32bNx(uint32_t const &src_addr, uint32_t *dst_ptr)
{
    static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of 2");

    if constexpr (N == 1)
    {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32"
                     "{%0},"
                     "[%1];\n"
                     : "=r"(dst_ptr[0])
                     : "r"(src_addr));
    }
    else if constexpr (N == 2)
    {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32"
                     "{%0, %1},"
                     "[%2];\n"
                     : "=r"(dst_ptr[0]), "=r"(dst_ptr[1])
                     : "r"(src_addr));
    }
    else if constexpr (N == 4)
    {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32"
                     "{%0, %1, %2, %3},"
                     "[%4];\n"
                     : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]), "=r"(dst_ptr[3])
                     : "r"(src_addr));
    }
    else if constexpr (N == 8)
    {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32"
                     "{%0, %1, %2, %3, %4, %5, %6, %7},"
                     "[%8];\n"
                     : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]), "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]), "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
                     : "r"(src_addr));
    }
    else if constexpr (N == 16)
    {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32"
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},"
                     "[%16];\n"
                     : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]), "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]), "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]), "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]), "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]), "=r"(dst_ptr[15])
                     : "r"(src_addr));
    }
    else if constexpr (N == 32)
    {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32"
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31},"
                     "[%32];\n"
                     : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]), "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]), "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]), "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]), "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]), "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]), "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]), "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]), "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]), "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]), "=r"(dst_ptr[30]), "=r"(dst_ptr[31])
                     : "r"(src_addr));
    }
    else if constexpr (N == 64)
    {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32"
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},"
                     "[%64];\n"
                     : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]), "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]), "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]), "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]), "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]), "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]), "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]), "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]), "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]), "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]), "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]), "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]), "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]), "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]), "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]), "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]), "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]), "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]), "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]), "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]), "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]), "=r"(dst_ptr[63])
                     : "r"(src_addr));
    }
    else if constexpr (N == 128)
    {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x128.b32"
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127},"
                     "[%128];\n"
                     : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]), "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]), "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]), "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]), "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]), "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]), "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]), "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]), "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]), "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]), "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]), "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]), "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]), "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]), "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]), "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]), "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]), "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]), "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]), "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]), "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]), "=r"(dst_ptr[63]), "=r"(dst_ptr[64]), "=r"(dst_ptr[65]), "=r"(dst_ptr[66]), "=r"(dst_ptr[67]), "=r"(dst_ptr[68]), "=r"(dst_ptr[69]), "=r"(dst_ptr[70]), "=r"(dst_ptr[71]), "=r"(dst_ptr[72]), "=r"(dst_ptr[73]), "=r"(dst_ptr[74]), "=r"(dst_ptr[75]), "=r"(dst_ptr[76]), "=r"(dst_ptr[77]), "=r"(dst_ptr[78]), "=r"(dst_ptr[79]), "=r"(dst_ptr[80]), "=r"(dst_ptr[81]), "=r"(dst_ptr[82]), "=r"(dst_ptr[83]), "=r"(dst_ptr[84]), "=r"(dst_ptr[85]), "=r"(dst_ptr[86]), "=r"(dst_ptr[87]), "=r"(dst_ptr[88]), "=r"(dst_ptr[89]), "=r"(dst_ptr[90]), "=r"(dst_ptr[91]), "=r"(dst_ptr[92]), "=r"(dst_ptr[93]), "=r"(dst_ptr[94]), "=r"(dst_ptr[95]), "=r"(dst_ptr[96]), "=r"(dst_ptr[97]), "=r"(dst_ptr[98]), "=r"(dst_ptr[99]), "=r"(dst_ptr[100]), "=r"(dst_ptr[101]), "=r"(dst_ptr[102]), "=r"(dst_ptr[103]), "=r"(dst_ptr[104]), "=r"(dst_ptr[105]), "=r"(dst_ptr[106]), "=r"(dst_ptr[107]), "=r"(dst_ptr[108]), "=r"(dst_ptr[109]), "=r"(dst_ptr[110]), "=r"(dst_ptr[111]), "=r"(dst_ptr[112]), "=r"(dst_ptr[113]), "=r"(dst_ptr[114]), "=r"(dst_ptr[115]), "=r"(dst_ptr[116]), "=r"(dst_ptr[117]), "=r"(dst_ptr[118]), "=r"(dst_ptr[119]), "=r"(dst_ptr[120]), "=r"(dst_ptr[121]), "=r"(dst_ptr[122]), "=r"(dst_ptr[123]), "=r"(dst_ptr[124]), "=r"(dst_ptr[125]), "=r"(dst_ptr[126]), "=r"(dst_ptr[127])
                     : "r"(src_addr));
    }
}

inline void __device__ fence_view_async_tmem_load()
{
    asm volatile("tcgen05.wait::ld.sync.aligned; " ::);
}

inline void __device__ fence_view_async_tmem_store()
{
    asm volatile("tcgen05.wait::st.sync.aligned; " ::);
}

template <int M, int N>
inline void __device__ amma_fp16bf16_ss(uint64_t const desc_a,
                                        uint64_t const desc_b,
                                        uint32_t const tmem_c,
                                        uint32_t const idesc,
                                        uint32_t const addC = 1)
{
    static_assert(M == 64 || M == 128, "SM100_MMA_F16BF16 M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
    static_assert((M == 64 && (N % 8 == 0) && (8 <= N) && (N <= 256)) ||
                      (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
                  "SM100_MMA_F16BF16 N-mode size should be a multiple of 8 between 8 and 256 for M=64,\
                 or a multiple of 16 between 16 and 256 for M=128.");

    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(addC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}

union InstrDescriptor
{
    uint32_t desc_;

    struct
    {
        // Bitfield implementation avoids the need for shifts in assignment
        uint16_t sparse_id2_ : 2, // bit [ 0, 2) : Sparse meta data id2
            sparse_flag_ : 1,     // bit [ 2, 3) : 0 = dense. 1 = sparse. 1 value valid only for F32F16/S8/MXF8F6F4
            saturate_ : 1,        // bit [ 3, 4) : 0 = no saturate. 1 = saturate. 1 value valid only for S8
            c_format_ : 2,        // bit [ 4, 6) : 0 = F16. 1 = F32, 2 = S32
            : 1,                  //
            a_format_ : 3,        // bit [ 7,10) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. Boolean MMA: 0 Boolean
            b_format_ : 3,        // bit [10,13) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1 signed 8 bit. Boolean MMA: 0 Boolean
            a_negate_ : 1,        // bit [13,14) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
            b_negate_ : 1,        // bit [14,15) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format and MXF8F6F4Format
            a_major_ : 1;         // bit [15,16) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
        uint16_t b_major_ : 1,    // bit [16,17) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
            n_dim_ : 6,           // bit [17,23) : 3 LSBs not included. Valid values range from 1 (N=8) to 32 (N=256).  All values are not valid for all instruction formats
            : 1,                  //
            m_dim_ : 5,           // bit [24,29) : 4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16 (M=256)
            : 1,                  //
            max_shift_ : 2;       // bit [30,32) : Maximum shift for WS instruction. Encoded as follows: 0 = no shift, 1 = maximum shift of 8, 2 = maximum shift of 16, 3 = maximum shift of 32.
    };

    // Decay to a uint32_t
    inline __device__ constexpr explicit
    operator uint32_t() const noexcept { return desc_; }
};

template <int M, int N>
inline __device__ constexpr InstrDescriptor
make_instr_desc()
{
    InstrDescriptor desc_i = {};

    desc_i.a_format_ = 0; // F16
    desc_i.b_format_ = 0; // F16
    desc_i.c_format_ = 1; // F32

    desc_i.m_dim_ = (M >> 4);
    desc_i.n_dim_ = (N >> 3);

    desc_i.a_major_ = 0; // K-major
    desc_i.b_major_ = 0; // K-major

    desc_i.a_negate_ = 0; // no negate
    desc_i.b_negate_ = 0; // no negate
    desc_i.saturate_ = 0; // no saturate

    desc_i.sparse_flag_ = 0; // dense
    desc_i.sparse_id2_ = 0;

    desc_i.max_shift_ = 0; // NoShift

    return desc_i;
}

union SmemDescriptor
{
    uint64_t desc_ = 0;
    // Bitfield implementation avoids the need for shifts in assignment
    struct
    {
        // start_address, bit [0,14), 4LSB not included
        uint16_t start_address_ : 14, : 2; // 14 bits [0,14), 2 bits unused
        // leading dimension byte offset, bit [16,30), 4LSB not included
        uint16_t leading_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
        // stride dimension byte offset, bit [32,46), 4LSB not included
        uint16_t stride_byte_offset_ : 14, version_ : 2; // 14 bits [0,14), 2 bits [14,16)
        // base_offset, bit [49,52). leading_byte_offset_mode, bit [52,53).
        uint8_t : 1, base_offset_ : 3, lbo_mode_ : 1, : 3; // 1 bit unused, 3 bits [1,4), 1 bit [4,5), 3 bits unused
        // layout type, bit [61,64), SWIZZLE_NONE matrix descriptor = 0, SWIZZLE_128B matrix descriptor = 2, SWIZZLE_64B descriptor = 4, SWIZZLE_32B descriptor = 6, SWIZZLE_128B_BASE32B = 1, N/A = 3, N/A = 5, N/A = 7
        uint8_t : 5, layout_type_ : 3; // 6 bits unused, 3 bits [5,8)
    };
    // Seperate the field, as we may only update one part of desc
    struct
    {
        uint32_t lo;
        uint32_t hi;
    };

    // Decay to a uint64_t
    inline __device__ constexpr
    operator uint64_t() const noexcept { return desc_; }
};

enum class LayoutType : uint8_t
{
    SWIZZLE_NONE = 0,
    SWIZZLE_128B_BASE32B = 1,
    SWIZZLE_128B = 2,
    SWIZZLE_64B = 4,
    SWIZZLE_32B = 6
};

// hardcode to K major and fp16
template <int MN, int K, typename T>
inline __device__ constexpr SmemDescriptor
make_smem_desc(T *smem_ptr)
{
    constexpr int leading_bytes = K * sizeof(T);
    LayoutType sw_layout;
    if constexpr ((leading_bytes % 128) == 0)
    {
        sw_layout = LayoutType::SWIZZLE_128B;
    }
    else if constexpr ((leading_bytes % 64) == 0)
    {
        sw_layout = LayoutType::SWIZZLE_64B;
    }
    else if constexpr ((leading_bytes % 32) == 0)
    {
        sw_layout = LayoutType::SWIZZLE_32B;
    }
    else
    {
        static_assert(K == 0, "no support ");
    }

    SmemDescriptor desc;
    desc.version_ = 1;  // Set the version for blackwell
    desc.lbo_mode_ = 0; // set to legacy mode by default
    desc.layout_type_ = uint8_t(sw_layout);

    // Start address (4LSB not included)
    uint32_t start_address = cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(start_address >> 4);

    constexpr uint8_t base_offset = 0;
    desc.base_offset_ = base_offset;

    desc.stride_byte_offset_ = (8 * leading_bytes) >> 4;
    desc.leading_byte_offset_ = 1; // hardcode to 1 as we don't support SWIZZLE_NONE

    return desc;
}

inline __device__ void amma_commit(uint64_t const *smem_ptr)
{
    uint32_t bar_intptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :
                 : "r"(bar_intptr));
}
