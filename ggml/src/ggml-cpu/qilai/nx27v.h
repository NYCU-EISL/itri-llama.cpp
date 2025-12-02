#pragma once

#include <cstddef>

void nx27v_init();
void nx27v_vec_dot_q4_0_q8_0(int n, float * s, size_t bs, 
                             const void * vx, size_t bx, 
                             const void * vy, size_t by, int nrc, int ith_thread);