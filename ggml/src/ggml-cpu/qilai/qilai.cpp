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
                return true;
            // case GGML_OP_NORM:
            // case GGML_OP_RMS_NORM:
            //     size = 0;
            //     return true;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                // printf("===================================");
                return true;
            // case GGML_OP_NORM:
            //     forward_norm_f32(params, op);
            //     return true;
            // case GGML_OP_RMS_NORM:
            //     forward_rms_norm_f32(params, op);
            //     return true;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        // memcpy(t->data, data, data_size);
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

    return nullptr;
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
                return true;
            //     if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
            //         op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
            //         ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
            //         if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
            //             return false;
            //         }
            //         if (op->src[1]->type == GGML_TYPE_F32) {
            //             return true;
            //         }
            //     }
            //     break;
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
                return (ggml::cpu::tensor_traits *) (&ggml::cpu::qilai::qilai_impl);
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