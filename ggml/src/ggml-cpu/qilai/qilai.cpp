#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "qilai.h"

#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "traits.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>  // for GGML_ASSERT
#include <stdexcept>
#include <thread>

// clang-format off

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

namespace ggml::cpu::qilai {

class tensor_traits_base : public ggml::cpu::tensor_traits {
  public:
    virtual int repack(struct ggml_tensor * t, const void * data, size_t data_size) = 0;
};

class tensor_traits_common : public tensor_traits_base {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                size = ggml_nelements(op->src[0]) * sizeof(float);
                return true;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->type == GGML_TYPE_Q4_0) {
                    forward_mul_mat(params, op);
                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = params->ith;
        int nth = params->nth;
        /*          
         *          O   = F * W^T
         *          O^T = ( F * W^T )^T
         *              = W * F^T
         *          W       F       O
         * Layout: (n*k) x (k*m) = (n*m)
         * ne00 - k
         * ne01 - n
         * ne10 - k
         * ne11 - m
         * ne0  - n
         * ne1  - m
        */

        const ggml_type     type = src0->type;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_size(type));
        GGML_ASSERT(nb10 == ggml_type_size(src1->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == sizeof(float));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        const size_t                  batch_feature = ne12 * ne13;
        [[maybe_unused]] const size_t batch_weight  = ne02 * ne03;
        const size_t                  gemm_m        = ne11;
        const size_t                  gemm_k        = ne10;
        const size_t                  gemm_n        = ne01;

        GGML_ASSERT(batch_weight == 1);

        const int64_t r2 = ne12/ne02;
        const int64_t r3 = ne13/ne03;

        const int64_t ne_plane      = ne01*ne00;
        const size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*sizeof(float);

        // if (ith != 0) {
        //     return;
        // }

        void * wdata = params->wdata;

        // convert src0 to float
        if (type != GGML_TYPE_F32) {
            const auto * type_traits = ggml_get_type_traits(type);
            ggml_to_float_t const to_float = type_traits->to_float;
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    const void  *       x      = (char *)  src0->data + i02*nb02          + i03*nb03;
                          float * const wplane = (float *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                    const int64_t start =       ith*ne01/nth;
                    const int64_t end   = (ith + 1)*ne01/nth;
                    if (start < end) {
                        for (int64_t i01 = start; i01 < end; i01++) {
                            to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        for (int64_t i13 = 0; i13 < ne13; i13++) {
            for (int64_t i12 = 0; i12 < ne12; i12++) {
                const int64_t i03 = i13/r3;
                const int64_t i02 = i12/r2;

                const float * x = (float *) ((char *) src0->data + i02*nb02 + i03*nb03);
                const float * y = (float *) ((char *) src1->data + i12*nb12 + i13*nb13);
                      float * d = (float *) ((char *)  dst->data + i12*nb2  + i13*nb3);

                if (type != GGML_TYPE_F32) {
                    x = (float *) wdata + i02*ne_plane + i03*ne02*ne_plane;
                }
                
                const int64_t start =       ith*ne11/nth;
                const int64_t end   = (ith + 1)*ne11/nth;
                if (start < end) {
                    for (int64_t i11 = start; i11 < end; i11++) {
                        for (int64_t i01 = 0; i01 < ne01; i01++) {
                            float sum = 0.0f;
                            for (int64_t i00 = 0; i00 < ne00; i00++) {
                                const float xv = x[i01*ne00 + i00];
                                const float yv = y[i11*ne10 + i00];
                                sum += xv * yv;
                            }
                            d[i11*ne0 + i01] = sum;
                        }
                    }
                }
                
            }
        }
    }

    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        memcpy(t->data, data, data_size);
        return 0;
    }
};
static const tensor_traits_common             qilai_impl;

}  // namespace ggml::cpu::qilai

static const ggml::cpu::tensor_traits * ggml_qilai_get_optimal_repack_type(const struct ggml_tensor * cur) {
    // if (cur->type == GGML_TYPE_Q4_0) {
    //     if (cur->ne[1] % 16 == 0) {
    //         return &ggml::cpu::riscv64_spacemit::q4_0_16x8_q8_0;
    //     }
    // } else if (cur->type == GGML_TYPE_Q4_1) {
    //     if (cur->ne[1] % 16 == 0) {
    //         return &ggml::cpu::riscv64_spacemit::q4_1_16x8_q8_0;
    //     }
    // } else if (cur->type == GGML_TYPE_Q4_K) {
    //     if (cur->ne[1] % 16 == 0) {
    //         return &ggml::cpu::riscv64_spacemit::q4_k_16x8_q8_0;
    //     }
    // } else if (cur->type == GGML_TYPE_F32) {
    //     return &ggml::cpu::riscv64_spacemit::rvv_impl;
    // }

    return &ggml::cpu::qilai::qilai_impl;
}

static enum ggml_status ggml_backend_qilai_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         struct ggml_tensor *  tensor) {
    tensor->extra =
        (void *) const_cast<ggml::cpu::tensor_traits *>(ggml_qilai_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_qilai_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            struct ggml_tensor *  tensor,
                                                            const void *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::qilai::tensor_traits_base *) tensor->extra;
    if (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_qilai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_QILAI";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_qilai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_qilai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_qilai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cpu_qilai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cpu_qilai_nbytes(ggml_backend_buffer_type_t buft,
                                                       const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }

    size_t       nbytes;
    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
        }
    } else {
        nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;
        if (tensor->type == GGML_TYPE_Q4_K) {
            GGML_ASSERT(nbytes % sizeof(block_q4_K) == 0);
            nbytes = (nbytes / sizeof(block_q4_K)) * sizeof(block_q4_1) * 8;
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                nbytes += (tensor->ne[i] - 1) * (tensor->nb[i] / sizeof(block_q4_K)) * sizeof(block_q4_1) * 8;
            }
        } else {
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
            }
        }
    }

    GGML_UNUSED(buft);
    return nbytes;
}

namespace ggml::cpu::qilai {

class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_qilai_buffer_type()) {
                    if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        return false;
                    }
                    if (op->src[1]->type == GGML_TYPE_F32 &&
                        op->src[0]->type == GGML_TYPE_Q4_0) {
                        return true;
                    }
                }
                break;
            // case GGML_OP_NORM:
            // case GGML_OP_RMS_NORM:
            //     if (op->src[0]->type == GGML_TYPE_F32) {
            //         return true;
            //     }
            //     break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_qilai_buffer_type()) {
                    return (ggml::cpu::tensor_traits *) (&ggml::cpu::qilai::qilai_impl);
                }
        //         if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
        //             return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        //         }
        //         break;
        //     case GGML_OP_NORM:
        //     case GGML_OP_RMS_NORM:
        //         return (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            default:
                // GGML_ABORT("fatal error");
                break;
        }

        return nullptr;
    }
};

}  // namespace ggml::cpu::qilai

ggml_backend_buffer_type_t ggml_backend_cpu_qilai_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_qilai = {
  /* .iface    = */
        {
         /* .get_name         = */ ggml_backend_cpu_qilai_buffer_type_get_name,
         /* .alloc_buffer     = */ ggml_backend_cpu_qilai_buffer_type_alloc_buffer,
         /* .get_alignment    = */ ggml_backend_cpu_qilai_buffer_type_get_alignment,
         /* .get_max_size     = */ nullptr,
         /* .get_alloc_size   = */ ggml_backend_cpu_qilai_nbytes,
         /* .is_host          = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        new ggml::cpu::qilai::extra_buffer_type(),
    };

    return &ggml_backend_cpu_buffer_type_qilai;
}