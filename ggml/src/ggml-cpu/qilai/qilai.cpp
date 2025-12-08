#include "ggml-impl.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <ctime>
#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "nx27v.h"
#include "qilai.h"
#include "quants.h"
#include "traits.h"
#include "vec.h"

#include <time.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>  // for GGML_ASSERT
#include <stdexcept>
#include <thread>
#include <map>
#include <tuple>

// clang-format off

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

constexpr size_t div_round_up(size_t a, size_t b) {
  return (a + b - 1) / b;
}

namespace ggml::cpu::qilai {

  class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                size = ggml_row_size(GGML_TYPE_Q8_0, ggml_nelements(op->src[1])) * 4;
                size = ((size + QK4_0 - 1) / QK4_0) * (QK4_0 * sizeof(float) + sizeof(float));
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
                    // forward_mul_mat(params, op);
                    forward_mul_mat_q4(params, op);
                    // GGML_LOG_INFO("[Thread %d] GGML_OP_MUL_MAT src0: [%d, %d, %d, %d] src1: [%d, %d, %d, %d].\n", 
                    //     params->ith, op->src[0]->ne[0], op->src[0]->ne[1], op->src[0]->ne[2], op->src[0]->ne[3],
                    //                  op->src[1]->ne[0], op->src[1]->ne[1], op->src[1]->ne[2], op->src[1]->ne[3]);
                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    static void forward_mul_mat_q4(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor * dst = op;

        GGML_ASSERT(src0->type == GGML_TYPE_Q4_0); // Only accepts Q4_0 source

        GGML_TENSOR_BINARY_OP_LOCALS;

        // Thread info
        const int ith = params->ith;
        const int nth = params->nth;

        const size_t batch_src0 = ne02 * ne03;
        const size_t batch_src1 = ne12 * ne13;
        const size_t m = ne11;
        const size_t k = ne10; // ne10 == ne00
        const size_t n = ne01;

        GGML_ASSERT(batch_src0 == 1); // Only support single batch for src0

        // Calc the size of qunated src1
        const size_t block_count_k = div_round_up(k, QK4_0);
        const size_t src1_q8_0_size_per_batch = m * block_count_k * sizeof(block_q8_0);
        const size_t src1_q8_0_row_stride = block_count_k * sizeof(block_q8_0);
        const size_t src1_q8_0_stride = 
            div_round_up(src1_q8_0_size_per_batch, alignof(uint64_t)) * alignof(uint64_t); // Round to nearest uint64_t
        const size_t src1_q8_0_size = batch_src1 * src1_q8_0_stride;

        // Check params->wdata is large enough to store quanted src1
        const size_t desired_wsize = src1_q8_0_size + alignof(uint64_t) - 1;
        if (ith == 0 && params->wsize < desired_wsize) {
            throw std::runtime_error("wsize less than desired_wsize");
        }

        std::vector<matrix_mul_q4_0_q8_0_params> matmul_params_vec(batch_src1);
        for (size_t batch_idx = 0; batch_idx < batch_src1; ++batch_idx) {
            matrix_mul_q4_0_q8_0_params matmul_params;
            matmul_params.src0 = (void *)src0->data;
            matmul_params.src1 = (void *)(params->wdata + batch_idx * src1_q8_0_stride);
            matmul_params.dst = (float *)dst->data + batch_idx * ne0 * ne1;

            matmul_params.ne00 = ne00; matmul_params.ne01 = ne01;
            matmul_params.ne02 = ne02; matmul_params.ne03 = ne03;

            matmul_params.ne10 = ne10; matmul_params.ne11 = ne11;
            matmul_params.ne12 = ne12; matmul_params.ne13 = ne13;

            matmul_params.ne0 = ne0; matmul_params.ne1 = ne1;
            matmul_params.ne2 = ne2; matmul_params.ne3 = ne3;

            matmul_params.nb00 = nb00; matmul_params.nb01 = nb01;
            matmul_params.nb02 = nb02; matmul_params.nb03 = nb03;

            // Store quanted
            matmul_params.nb10 = sizeof(block_q8_0);
            matmul_params.nb11 = src1_q8_0_row_stride;
            matmul_params.nb12 = src1_q8_0_size_per_batch; 
            matmul_params.nb13 = 1;

            matmul_params.nb0 = nb0; matmul_params.nb1 = nb1;
            matmul_params.nb2 = nb2; matmul_params.nb3 = nb3;

            matmul_params_vec[batch_idx] = matmul_params;
        }
        
        // Set wdata pointer to align to uint64_t
        void * wdata = reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(params->wdata) + alignof(uint64_t) - 1) & 
                            (~(alignof(uint64_t) - 1)));

        // Quant src1 to q8_0
        if (src1->type == GGML_TYPE_F32) {

            GGML_ASSERT(nb10 == sizeof(float) && 
                        nb11 == ne10 * sizeof(float) && 
                        nb12 == ne11 * nb11); // src1 is contiguous float matrix

            // Divide the work among threads
            constexpr int num_rows_per_task = 4;
            const int task_count_per_batch = div_round_up(m, num_rows_per_task);
            const int task_count = batch_src1 * task_count_per_batch;
            const int task_count_per_thread = (task_count + nth - 1) / nth;

            // This thread's task range
            const int task_begin = ith * task_count_per_thread;
            const int task_end = std::min((ith + 1) * task_count_per_thread, task_count);

            for (int task_idx = task_begin; task_idx < task_end; ++task_idx) {
                const int batch_idx = task_idx / task_count_per_batch;
                const int task_idx_in_batch = task_idx % task_count_per_batch;
                const int m_idx = task_idx_in_batch * num_rows_per_task;

                const int rows_to_be_handled = std::min(num_rows_per_task, (int)(m - m_idx));

                const size_t src1_ptr_offset = batch_idx * nb12 + m_idx * nb11;
                const size_t quant_src1_ptr_offset = batch_idx * src1_q8_0_stride + m_idx * src1_q8_0_row_stride;

                float * src1_ptr = (float *)src1->data + src1_ptr_offset / sizeof(float);
                block_q8_0 * quant_src1_ptr = (block_q8_0 *)((char *)wdata + quant_src1_ptr_offset);

                quantize_row_q8_0(src1_ptr, quant_src1_ptr, k * rows_to_be_handled);
            }
        }
        
        ggml_barrier(params->threadpool);

        // Compute mul_mat using q4_0 src0 and q8_0 src1
        // if (ith == 0)
        // GGML_LOG_INFO("[Thread %d] GGML_OP_MUL_MAT with Q4_0 x Q8_0: m = %zu, n = %zu, k = %zu, batch_src1 = %zu\n", 
        //     ith, m, n, k, batch_src1);

        // if (ith == 0)
        //     GGML_LOG_INFO("[Thread %d] dst: [%d, %d, %d, %d].\n", 
        //         ith, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);

        // Divide the work among threads by tiling dst matrix
        const int tile_m_size = 64;
        const int tile_n_size = 64; // Use wider tile for vector-matrix mul

        const int tile_m_count_per_batch = div_round_up(m, tile_m_size);
        const int tile_n_count_per_batch = div_round_up(n, tile_n_size);
        const int tile_count_per_batch = tile_m_count_per_batch * tile_n_count_per_batch;
        const int tile_count = batch_src1 * tile_count_per_batch;

        // if (ith == 0)
        // GGML_LOG_INFO("[Thread %d] total tiles: %d (tile_m_count_per_batch=%d, tile_n_count_per_batch=%d)\n", 
        //     ith, tile_count, tile_m_count_per_batch, tile_n_count_per_batch);

        // Each thread handles multiple tiles
        {
            const int tiles_per_thread = div_round_up(tile_count, nth);
            const int tile_begin = ith * tiles_per_thread;
            const int tile_end = std::min((ith + 1) * tiles_per_thread, tile_count);

            // GGML_LOG_INFO("[Thread %d] handling tiles %d to %d\n", ith, tile_begin, tile_end);

            for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
                const int batch_idx = tile_idx / tile_count_per_batch;
                const int tile_idx_in_batch = tile_idx % tile_count_per_batch;
                const int tile_m_idx = tile_idx_in_batch / tile_n_count_per_batch;
                const int tile_n_idx = tile_idx_in_batch % tile_n_count_per_batch;

                // GGML_LOG_INFO("[Thread %d] processing tile %d (batch %d, tile_m_idx %d, tile_n_idx %d)\n", 
                //     ith, tile_idx, batch_idx, tile_m_idx, tile_n_idx);

                // Row and col start index of the tile
                const size_t m_start = tile_m_idx * tile_m_size;
                const size_t m_count = std::min((size_t)tile_m_size, m - m_start);

                const size_t n_start = tile_n_idx * tile_n_size;
                const size_t n_count = std::min((size_t)tile_n_size, n - n_start);

                // GGML_LOG_INFO("[Thread %d] tile %d: m_start=%zu, m_count=%zu, n_start=%zu, n_count=%zu\n", 
                //     ith, tile_idx, m_start, m_count, n_start, n_count);

                // Prepare pointers
                matrix_mul_q4_0_q8_0_params & matmul_params = matmul_params_vec[batch_idx];

                const size_t src0_ptr_offset = n_start * nb01;
                const size_t src1_ptr_offset = m_start * src1_q8_0_row_stride + batch_idx * src1_q8_0_stride;
                const size_t dst_ptr_offset =  n_start * nb0 + m_start * nb1 + batch_idx * nb2;

                // GGML_LOG_INFO("[Thread %d] tile %d: src0_ptr_offset=%zu, src1_ptr_offset=%zu, dst_ptr_offset=%zu\n", 
                //     ith, tile_idx, src0_ptr_offset, src1_ptr_offset, dst_ptr_offset);

                void * src0_ptr = src0->data + src0_ptr_offset;
                void * src1_ptr = wdata + src1_ptr_offset;
                float * dst_ptr = (float *)dst->data + dst_ptr_offset / sizeof(float);

                // Compute the tile
                if (ith < 1) {
                    // Use NX27V accelerator for first few threads
	                // GGML_LOG_INFO("[QILAI] T%d: Src0:[%d, %d] Src1:[%d, %d]\n", ith, ne00, ne01, ne10, ne11);
                    nx27v_mat_mul_tile_q4_0_q8_0(matmul_params, m_start, m_count, n_start, n_count, ith);
                } else {
                    // Use CPU implementation for other threads
                    static thread_local unsigned long long counter = 0, n_counter = 0;
                    static thread_local double calc_time = 0.0;

                    timespec start_time, end_time;
                    clock_gettime(CLOCK_MONOTONIC, &start_time);

                    for (size_t im = 0; im < m_count; ++im) {
                        for (size_t in = 0; in < n_count; ++in) {
                            void * src0_elem_ptr = src0_ptr + (in * nb01);
                            void * src1_elem_ptr = src1_ptr + (im * src1_q8_0_row_stride);
                            float * dst_elem_ptr = dst_ptr + (in * nb0 + im * nb1) / sizeof(float);

                            // GGML_LOG_INFO("[Thread %d] tile %d: computing dst[%zu, %zu], src0_ptr_offset=%zu, src1_ptr_offset=%zu, dst_ptr_offset=%zu\n", 
                            //     ith, tile_idx, m_start + im, n_start + in, 
                            //     ((size_t)src0_elem_ptr - (size_t)src0_ptr), 
                            //     ((size_t)src1_elem_ptr - (size_t)src1_ptr), 
                            //     ((size_t)dst_elem_ptr - (size_t)dst_ptr));

                            ggml_vec_dot_q4_0_q8_0(k, dst_elem_ptr, 0, src0_elem_ptr, 0, src1_elem_ptr, 0, 1);
                        }
                    }
		    
                    clock_gettime(CLOCK_MONOTONIC, &end_time);		    
                    calc_time += (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
                    n_counter += matmul_params.ne00 * m_count * n_count;
                    if (++counter % 1000 == 0) {
                    GGML_LOG_INFO("[QILAI T%d] Avg. ms of %llu matmul tiles: calc=%.6f, calc_per_n=%.6f\n", ith, counter, calc_time * 1e3 / counter, calc_time * 1e3 / n_counter);
                    }
                }
            }
        }

        // Profiling
        struct ne_pair {
            int ne0[4], ne1[4];

            ne_pair() {}
            ne_pair(int a, int b, int c, int d, int e, int f, int g, int h) {
                ne0[0] = a;
                ne0[1] = b;
                ne0[2] = c;
                ne0[3] = d;
                ne1[0] = e;
                ne1[1] = f;
                ne1[2] = g;
                ne1[3] = h;
            }

            bool operator<(const ne_pair & other) const noexcept {
                const int lhs[8] = {ne0[0], ne0[1], ne0[2], ne0[3], ne1[0], ne1[1], ne1[2], ne1[3]};
                const int rhs[8] = {other.ne0[0], other.ne0[1], other.ne0[2], other.ne0[3],
                                    other.ne1[0], other.ne1[1], other.ne1[2], other.ne1[3]};
                for (int i = 0; i < 8; ++i) {
                    if (lhs[i] != rhs[i]) return lhs[i] < rhs[i];
                }
                return false;
            }
        };

        static thread_local long long counter = 0;
        static thread_local std::map<ne_pair, int> ne_stat;

        ne_stat[ne_pair(ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13)]++;

        if (++counter % 100 == 0) {
            GGML_LOG_INFO("[NX27V T%d] finished %lld matmul\n", ith, counter);

            std::map<int, std::vector<ne_pair>> ko_board;
            for (auto &[ne, cnt]: ne_stat) {
                ko_board[-cnt].push_back(ne);
            }

            GGML_LOG_INFO("[QILAI T%d] Top 10 dim:\n", ith);
            int print_num = 0;
            bool flag = false;
            for (auto &[cnt, ne_vec]: ko_board) {
                for (auto &ne: ne_vec) {
                    if (print_num++ < 10) {
                        GGML_LOG_INFO("            ne0[");
                        
                        for (auto i: ne.ne0)
                            GGML_LOG_INFO(" %d", i);
                        GGML_LOG_INFO("] ne1[");
                        for (auto i: ne.ne1)
                            GGML_LOG_INFO(" %d", i);
                        GGML_LOG_INFO("]: %d\n", -cnt);
                    } else {
                        flag=true;
                        break;
                    }
                }
                if (flag) break;
            }

        }

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

        // printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

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

        // GGML_LOG_INFO("SRC0 NE:");
        // for (auto &i: src0->ne) {
        //     GGML_LOG_INFO(" %lld", i);
        // }
        // GGML_LOG_INFO("\n");
        // GGML_LOG_INFO("SRC1 NE:");
        // for (auto &i: src1->ne) {
        //     GGML_LOG_INFO(" %lld", i);
        // }
        // GGML_LOG_INFO("\n");

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

                    // GGML_LOG_INFO("[NX27V] Thread %d: Prepared src0 at offset 0x%lx, src1 at offset 0x%lx\n", params->ith, src0_addr_offset, src1_addr_offset);

                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                        // vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                        
                        if (params->ith < 0) {
                            nx27v_vec_dot_q4_0_q8_0(
                                ne00,
                                &tmp[ir0 - iir0],
                                (num_rows_per_vec_dot > 1 ? 16 : 0),
                                src0_row + ir0 * nb01,
                                (num_rows_per_vec_dot > 1 ? nb01 : 0),
                                src1_col,
                                (num_rows_per_vec_dot > 1 ? src1_col_stride : 0),
                                num_rows_per_vec_dot,
                                params->ith);

                            // float ref_ans = 0.0f;
                            // vec_dot(ne00, &ref_ans, (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);

                            // if (std::abs(tmp[ir0 - iir0] - ref_ans) > 1e-3) {
                            //     GGML_LOG_INFO("[NX27V] Thread %d: MISMATCH at ir0=%lld, ir1=%lld: accelerator=%f, reference=%f\n", params->ith, ir0, ir1, tmp[ir0 - iir0], ref_ans);
                            // }
                        } else {
                            vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                        }


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

        // Transform src1 to the same computation type if needed
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
};
static const tensor_traits             qilai_impl;

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

    auto tensor_traits = (ggml::cpu::qilai::tensor_traits *) tensor->extra;
    if (tensor_traits) {
        // auto OK = tensor_traits->repack(tensor, data, size);
        memcpy(tensor->data, data, size);
        int OK = 0;
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

    static bool initialized = false;
    if (!initialized) {
        nx27v_init();
        initialized = true;
    }

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
