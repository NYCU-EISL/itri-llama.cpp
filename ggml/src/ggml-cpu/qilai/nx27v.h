#pragma once

#include <cstddef>
#include "qilai.h"

void nx27v_init();
void nx27v_vec_dot_q4_0_q8_0(int n, float * s, size_t bs, 
                             const void * vx, size_t bx, 
                             const void * vy, size_t by, int nrc, int ith_thread);

void nx27v_mat_mul_tile_q4_0_q8_0(matrix_mul_q4_0_q8_0_params &params, 
    size_t m_start, size_t m_count, size_t n_start, size_t n_count, 
    int ith_thread);