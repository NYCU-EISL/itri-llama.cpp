#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "qilai.h"

#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "quants.h"
#include "traits.h"
#include "vec.h"

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

    static void ggml_compute_forward_mul_mat_one_chunk(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const enum ggml_type type,
        const int64_t num_rows_per_vec_dot,
        const int64_t ir0_start,
        const int64_t ir0_end,
        const int64_t ir1_start,
        const int64_t ir1_end) {

        const struct ggml_tensor * src0 = dst->src[0];
        const struct ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        const bool src1_cont = ggml_is_contiguous(src1);

        ggml_vec_dot_t const vec_dot      = ggml_vec_dot_q4_0_q8_0;
        enum ggml_type const vec_dot_type = GGML_TYPE_Q8_0;

        // broadcast factors
        const int64_t r2 = ne12 / ne02;
        const int64_t r3 = ne13 / ne03;

        //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

        // threads with no work simply yield (not sure if it helps)
        if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
            return;
        }

        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        assert(ne12 % ne02 == 0);
        assert(ne13 % ne03 == 0);

        // block-tiling attempt
        const int64_t blck_0 = 16;
        const int64_t blck_1 = 16;

        const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

        // attempt to reduce false-sharing (does not seem to make a difference)
        // 16 * 2, accounting for mmla kernels
        float tmp[32];

        for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
            for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
                for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                    const int64_t i13 = (ir1 / (ne12 * ne1));
                    const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                    const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                    // broadcast src0 into src1
                    const int64_t i03 = i13 / r3;
                    const int64_t i02 = i12 / r2;

                    const int64_t i1 = i11;
                    const int64_t i2 = i12;
                    const int64_t i3 = i13;

                    const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                    // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                    //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                    //       the original src1 data pointer, so we should index using the indices directly
                    // TODO: this is a bit of a hack, we should probably have a better way to handle this
                    const char * src1_col = (const char*)wdata +
                        (src1_cont || src1->type != vec_dot_type
                            ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                            : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                    float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                    //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                    //}

                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                        vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                    }

                    for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                        memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                    }
                }
            }
        }
    }

    void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS
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

        const int ith = params->ith;
        const int nth = params->nth;

        enum ggml_type           const vec_dot_type         = GGML_TYPE_Q8_0;
        ggml_from_float_t        const from_float           = ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref;
        int64_t                  const vec_dot_num_rows     = 1;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_size(src0->type));
        GGML_ASSERT(nb10 == ggml_type_size(src1->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == sizeof(float));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        if (src1->type != vec_dot_type) {
            void * wdata = params->wdata;

            const size_t nbw0 = ggml_type_size(vec_dot_type);
            const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
            const size_t nbw2 = nbw1*ne11;
            const size_t nbw3 = nbw2*ne12;

            assert(params->wsize >= ne13*nbw3);
            GGML_ASSERT(src1->type == GGML_TYPE_F32);
            for (int64_t i13 = 0; i13 < ne13; ++i13) {
                for (int64_t i12 = 0; i12 < ne12; ++i12) {
                    for (int64_t i11 = 0; i11 < ne11; ++i11) {
                        size_t bs = ggml_blck_size(vec_dot_type);
                        int64_t ne10_block_start = (ith * ne10/bs) / nth;
                        int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                        from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                                   (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                                   (ne10_block_end - ne10_block_start) * bs);
                    }
                }
            }
        }
        

        if (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
        const int64_t nr0 = ne0;

        // This is the size of the rest of the dimensions of the result
        const int64_t nr1 = ne1 * ne2 * ne3;

        // Now select a reasonable chunk size.
        int chunk_size = 16;

        // We need to step up the size if it's small
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // distribute the work across the inner or outer loop based on which one is larger
        // The number of chunks in the 0/1 dim.
        // CEIL(nr0/chunk_size)
        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
        //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
        //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
        if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
            // distribute the thread work across the inner or outer loop based on which one is larger
            nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
            nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
        }

        // The number of elements in each chunk
        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        int current_chunk = ith;

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
            int64_t num_rows_per_vec_dot = vec_dot_num_rows;

            // these checks are needed to avoid crossing dim1 boundaries
            // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
            if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
                num_rows_per_vec_dot = 1;
            }
            ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
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