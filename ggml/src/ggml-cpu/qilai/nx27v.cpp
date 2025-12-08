#include <assert.h>
#include <bits/time.h>
#include <cstdint>
#include <stddef.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <linux/rpmsg.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <time.h>

#include "ggml-impl.h"
#include "quants.h"

#include "nx27v.h"
#include "my_msg.h"
#include "qilai.h"

static bool rpmsg_init = false;
static void *share_addr = nullptr;
static int datafd[RPMSG_EPT_NUM];

constexpr size_t div_round_up(size_t a, size_t b) {
  return (a + b - 1) / b;
}

static void setup_rpmsg_endpoint(const char *ept_name, int dst) {
    struct rpmsg_endpoint_info eptinfo;

    int ctrlfd = open("/dev/rpmsg_ctrl0", O_RDWR);
    assert(ctrlfd >= 0 && "Failed to open rpmsg_ctrl0");

    memset(&eptinfo, 0, sizeof(eptinfo));
    strncpy(eptinfo.name, ept_name, sizeof(eptinfo.name) - 1);

    eptinfo.src = RPMSG_ADDR_ANY;
    eptinfo.dst = dst;

    int ret = ioctl(ctrlfd, RPMSG_CREATE_EPT_IOCTL, &eptinfo);
    assert(ret == 0 && "Failed to create rpmsg endpoint");

    close(ctrlfd);
}

static void setup_shared_memory() {
    int memfd = open("/dev/mem", O_RDWR | O_SYNC);
    assert(memfd >= 0 && "Failed to open /dev/mem");

    share_addr = mmap(NULL, SHARE_MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, SHARE_MEM_ADDR);
    assert(share_addr != MAP_FAILED && "Failed to mmap shared memory");

    close(memfd);
}

void nx27v_init() {
    GGML_LOG_INFO("[NX27V] Initializing...\n");
    if (rpmsg_init) {
	GGML_LOG_INFO("[NX27V] Already initialized.\n");
	return;
    }

    if (access("/dev/rpmsg_ctrl0", F_OK) != 0) {
	fprintf(stderr, "RPMsg driver not loaded. Please load the rpmsg_char driver.\n");
	exit(EXIT_FAILURE);
    }
    GGML_LOG_INFO("[NX27V] /dev/rpmsg_ctrl0 found.\n");

    if (access("/dev/rpmsg0", F_OK) != 0) {
	for (int i = 0; i < RPMSG_EPT_NUM; ++i) {
	    char ept_name[32];
	    snprintf(ept_name, sizeof(ept_name), "%s_%d", RPMSG_EPT_NAME, i);
	    setup_rpmsg_endpoint(ept_name, RPMSG_EPT_ADDR[i]);
	}
	GGML_LOG_INFO("[NX27V] RPMsg endpoints created.\n");
    } else {
	GGML_LOG_INFO("[NX27V] RPMsg endpoints already exist, reusing.\n");
    }

    for (int i = 0; i < RPMSG_EPT_NUM; ++i) {
	char rpmsg_dev_path[32];
	snprintf(rpmsg_dev_path, sizeof(rpmsg_dev_path), "/dev/rpmsg%d", i);
	datafd[i] = open(rpmsg_dev_path, O_RDWR);
	assert(datafd[i] >= 0 && "Failed to open rpmsg device");
	GGML_LOG_INFO("[NX27V] Opened %s\n", rpmsg_dev_path);
    }

    setup_shared_memory();
    GGML_LOG_INFO("[NX27V] Shared memory mapped at address %p\n", share_addr);

    rpmsg_init = true;
    GGML_LOG_INFO("[NX27V] Initialization complete.\n");
}

void nx27v_vec_dot_q4_0_q8_0(int n, float * s, size_t bs,
			     const void * vx, size_t bx,
			     const void * vy, size_t by, int nrc, int ith_thread)
{
    assert(nrc == 1);
    assert(ith_thread >= 0 && ith_thread < RPMSG_EPT_NUM);

    assert(n * sizeof(block_q4_0) / QK4_0 <= SHARE_MEM_SRC1_OFFSET - SHARE_MEM_SRC0_OFFSET && "Input size exceeds shared memory capacity");
    assert(n * sizeof(block_q8_0) / QK8_0 <= SHARE_MEM_SRC1_OFFSET - SHARE_MEM_SRC0_OFFSET && "Input size exceeds shared memory capacity");

    static thread_local uint64_t counter = 0;
    static thread_local uint64_t n_counter = 0;
    static thread_local double memcpy_src_time = 0.0;
    static thread_local double write_req_time = 0.0;
    static thread_local double read_resp_time = 0.0;

    timespec start_time, end_time;

    uint64_t src0_addr_offset = SHARE_MEM_SRC0_OFFSET + ith_thread * SHARE_MEM_SIZE_PER_THREAD;
    uint64_t src1_addr_offset = SHARE_MEM_SRC1_OFFSET + ith_thread * SHARE_MEM_SIZE_PER_THREAD;

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    memcpy((void *)(share_addr + src0_addr_offset), vx, n * sizeof(block_q4_0));
    memcpy((void *)(share_addr + src1_addr_offset), vy, n * sizeof(block_q8_0));

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    memcpy_src_time += (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    my_msg msg = {
	.opcode = MSG_OP_VEC_DOT_Q4_0_Q8_0,
	.src0_ne = { (uint32_t)(n), 1, 1, 1 },
	.src1_ne = { (uint32_t)(n), 1, 1, 1 },
	.src0_addr = (uint64_t)(SHARE_MEM_ADDR + src0_addr_offset),
	.src1_addr = (uint64_t)(SHARE_MEM_ADDR + src1_addr_offset),
	.result_addr = 0, // not used for dot product
    };

    ssize_t ret = write(datafd[ith_thread], &msg, sizeof(msg));
    assert(ret == sizeof(msg) && "Failed to send message to accelerator");

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    write_req_time += (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;


    clock_gettime(CLOCK_MONOTONIC, &start_time);

    my_msg_ret_status ret_msg;
    ret = read(datafd[ith_thread], &ret_msg, sizeof(ret_msg));
    assert(ret == sizeof(ret_msg) && "Failed to read response from accelerator");
    assert(ret_msg.status == 0 && "Accelerator reported error");

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    read_resp_time += (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;


    *s = ret_msg.dot_result;

    n_counter += n;
    if (++counter % 1000 == 0) {
	GGML_LOG_INFO("[NX27V] T%d: Avg. ms of %lu: memcpy_src=%.6f, write_req=%.6f, read_resp=%.6f, read_resp_per_n=%.6f\n",
	       ith_thread,
	       counter,
	       memcpy_src_time * 1e3 / counter,
	       write_req_time  * 1e3 / counter,
	       read_resp_time  * 1e3 / counter,
	       read_resp_time  * 1e3 / n_counter);
    }
}


void nx27v_mat_mul_tile_q4_0_q8_0(matrix_mul_q4_0_q8_0_params &params,
				  size_t m_start, size_t m_count, size_t n_start, size_t n_count,
				  int ith_thread) {
    assert(ith_thread >= 0 && ith_thread < RPMSG_EPT_NUM && "thread idx is larger than available RPMSG channel");

    void * src0 = params.src0; // Q4_0
    void * src1 = params.src1; // Q8_0
    float *dst  = params.dst;  // F32

    const size_t m = params.ne11;
    const size_t k = params.ne10;
    const size_t n = params.ne01;

    // Profiling
    static thread_local uint64_t counter = 0;
    static thread_local uint64_t n_counter = 0;
    static thread_local double memcpy_src_time = 0.0;
    static thread_local double write_req_time = 0.0;
    static thread_local double read_resp_time = 0.0;
    static thread_local double memcpy_dst_time = 0.0;
    static thread_local uint64_t nx27v_warmup_cycles = 0;
    static thread_local uint64_t nx27v_actual_cycles = 0;
    timespec start_time, end_time;

    // Check if data is already on the remote
    static thread_local int last_ith = -1;
    static thread_local void *last_src0 = nullptr;
    static thread_local void *last_src1 = nullptr;
    static thread_local size_t last_src0_len = 0;
    static thread_local size_t last_src1_len = 0;

    static thread_local uint64_t src0_cpy = 0;
    static thread_local uint64_t src1_cpy = 0;
    static thread_local uint64_t src_whole_matrix = 0;

    bool whole_matrix = false;
    void *src0_start_ptr;
    void *src1_start_ptr;
    size_t src0_len;
    size_t src1_len;

    // Try to put the whole matrix into share memory at once
    uint64_t src0_share_mem_size = GGML_PAD(params.nb02, sizeof(uint64_t));
    uint64_t src1_share_mem_size = GGML_PAD(params.nb12, sizeof(uint64_t));
    uint64_t dst_share_mem_size = GGML_PAD(m * n, sizeof(uint64_t));

    uint64_t src0_addr_offset = 0;
    uint64_t src1_addr_offset = src0_addr_offset + src0_share_mem_size;
    uint64_t dst_addr_offset = src1_addr_offset + src1_share_mem_size;     

    if (dst_addr_offset + dst_share_mem_size <= SHARE_MEM_SIZE_PER_THREAD && false) {
        src0_start_ptr = src0;
        src1_start_ptr = src1;
        src0_len = params.nb02;
        src1_len = params.nb12;
        whole_matrix = true;
      
        src_whole_matrix++;
    } else {
        // Put only the required part into share mem
        src0_share_mem_size = GGML_PAD(n_count * params.nb01, sizeof(uint64_t));
        src1_share_mem_size = GGML_PAD(m_count * params.nb11, sizeof(uint64_t));
        dst_share_mem_size = GGML_PAD(m_count * n_count, sizeof(uint64_t));

        src0_addr_offset = 0;
        src1_addr_offset = src0_addr_offset + src0_share_mem_size;
        dst_addr_offset  = src1_addr_offset + src1_share_mem_size; 

        GGML_ASSERT(dst_addr_offset + dst_share_mem_size <= SHARE_MEM_SIZE_PER_THREAD && "src0 tile exceeds shared memory capacity");
      
        src0_start_ptr = src0 + (n_start * params.nb01);
        src1_start_ptr = src1 + (m_start * params.nb11);
        src0_len = n_count * params.nb01;
        src1_len = m_count * params.nb11;
    }

    // Add thread offset
    src0_addr_offset += ith_thread * SHARE_MEM_SIZE_PER_THREAD;
    src1_addr_offset += ith_thread * SHARE_MEM_SIZE_PER_THREAD;
    dst_addr_offset += ith_thread * SHARE_MEM_SIZE_PER_THREAD;

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    if (!(ith_thread == last_ith && src0_start_ptr == last_src0 && src0_len == last_src0_len) ||
        !(ith_thread == last_ith && src1_start_ptr == last_src1 && src1_len == last_src1_len)) {
        memcpy((void *)(share_addr + src0_addr_offset), src0_start_ptr, src0_len);
        src0_cpy++;
        memcpy((void *)(share_addr + src1_addr_offset), src1_start_ptr, src1_len);
        src1_cpy++;

        last_ith = ith_thread;
        last_src0 = src0_start_ptr;
        last_src1 = src1_start_ptr;
        last_src0_len = src0_len;
        last_src1_len = src1_len;
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    memcpy_src_time += (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    my_msg msg = {
        .opcode = MSG_OP_MAT_MUL_Q4_0_Q8_0,
        .src0_ne = { (uint32_t)(k), (uint32_t)(n_count), 1, 1 },
        .src1_ne = { (uint32_t)(k), (uint32_t)(m_count), 1, 1 },
        .src0_nb = { (uint32_t)(params.nb00), (uint32_t)(params.nb01), 0, 0 },
        .src1_nb = { (uint32_t)(params.nb10), (uint32_t)(params.nb11), 0, 0 },
        .src0_addr = (uint64_t)(SHARE_MEM_ADDR + src0_addr_offset + (whole_matrix ? n_start * params.nb01 : 0)),
        .src1_addr = (uint64_t)(SHARE_MEM_ADDR + src1_addr_offset + (whole_matrix ? m_start * params.nb11 : 0)),
        .result_addr = (uint64_t)(SHARE_MEM_ADDR + dst_addr_offset),
    };

    // GGML_LOG_INFO("[T%d] Sending message to accelerator", ith_thread);

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    ssize_t ret = write(datafd[ith_thread], &msg, sizeof(msg));
    assert(ret == sizeof(msg) && "Failed to send message to accelerator");

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    write_req_time += (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    // GGML_LOG_INFO("[T%d] Waiting for response from accelerator", ith_thread);

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    my_msg_ret_status ret_msg;
    ret = read(datafd[ith_thread], &ret_msg, sizeof(ret_msg));
    assert(ret == sizeof(ret_msg) && "Failed to read response from accelerator");
    assert(ret_msg.status == 0 && "Accelerator reported error");

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    read_resp_time += (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    nx27v_warmup_cycles += ret_msg.perf_counter[0];
    nx27v_actual_cycles += ret_msg.perf_counter[1];

    // GGML_LOG_INFO("[T%d] Copying result back to dst matrix\n", ith_thread);

    // for (size_t im = 0; im < m_count; ++im) {
    //     for (size_t in = 0; in < n_count; ++in) {
    //         void * src0_elem_ptr = src0 + ((n_start + in) * params.nb01);
    //         void * src1_elem_ptr = src1 + ((m_start + im) * params.nb11);
    //         float * dst_elem_ptr = dst + ((n_start + in) * params.nb0 + (m_start + im) * params.nb1) / sizeof(float);

    //         ggml_vec_dot_q4_0_q8_0(k, dst_elem_ptr, 0, src0_elem_ptr, 0, src1_elem_ptr, 0, 1);
    //     }
    // }

    // GGML_LOG_INFO("[NX27V] T%d: Src0:[%d, %d] Src1:[%d, %d]\n", ith_thread, params.ne00, params.ne01, params.ne10, params.ne11);
    // GGML_ASSERT(params.nb0 == sizeof(float));
    // GGML_ASSERT(params.nb1 == sizeof(float) * params.ne01);
    // GGML_ASSERT(params.ne0 == params.ne01);
    // GGML_ASSERT(params.ne1 == params.ne11);

    // copy back the result

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (size_t im = 0; im < m_count; ++im) {
        for (size_t in = 0; in < n_count; ++in) {
            float * dst_elem_ptr = dst + ((n_start + in) * params.nb0 + (m_start + im) * params.nb1) / sizeof(float);
            float * src_elem_ptr = (float *)(share_addr + dst_addr_offset
                            + (in * sizeof(float)) + (im * n_count * sizeof(float)));

            // if (std::abs(*dst_elem_ptr - *src_elem_ptr) > 1e-3) {
            //     GGML_LOG_INFO("[NX27V T%d] Dst[%d, %d] is different, ref=%f, nx27v=%f\n", ith_thread, in, im, *dst_elem_ptr, *src_elem_ptr);
            // }
            
            *dst_elem_ptr = *src_elem_ptr;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    memcpy_dst_time += (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    n_counter += k * m_count * n_count;
    if (++counter % 1000 == 0) {
        GGML_LOG_INFO("[NX27V T%d] Avg. ms of %lu matmul tiles: cpy_src=%.6f, req=%.6f, resp=%.6f, cpy_dst=%.6f, resp_per_n=%.6f\n", 
            ith_thread, counter,
            memcpy_src_time * 1e3 / counter,
            write_req_time * 1e3 / counter,
            read_resp_time * 1e3 / counter,
            memcpy_dst_time * 1e3 / counter,
            read_resp_time * 1e3 / n_counter);
        // GGML_LOG_INFO("[NX27V T%d] Avg. cycle count: warmup=%ld actual=%ld\n",
        //     ith_thread, (nx27v_warmup_cycles / counter), (nx27v_actual_cycles / counter));
        GGML_LOG_INFO("[NX27V T%d] Avg. cycle count: %ld\n",
        	    ith_thread, (nx27v_warmup_cycles / counter));
        GGML_LOG_INFO("[NX27V T%d] src0_cpy=%ld src1_cpy=%ld src_whole=%ld\n",
            ith_thread, src0_cpy, src1_cpy, src_whole_matrix);
    }

}
