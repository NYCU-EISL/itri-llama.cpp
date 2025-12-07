#pragma once

#include "ggml-alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_qilai_buffer_type(void);

struct matrix_mul_q4_0_q8_0_params {
    void * src0; // Q4_0
    void * src1; // Q8_0
    float *dst;  // F32

    size_t ne00, ne01, ne02, ne03;
    size_t ne10, ne11, ne12, ne13;
    size_t ne0, ne1, ne2, ne3;

    size_t nb00, nb01, nb02, nb03;
    size_t nb10, nb11, nb12, nb13;
    size_t nb0, nb1, nb2, nb3;
};

#ifdef __cplusplus
}
#endif
