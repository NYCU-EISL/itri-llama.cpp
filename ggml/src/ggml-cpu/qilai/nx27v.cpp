#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <linux/rpmsg.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <stdio.h>

#include "ggml-impl.h"
#include "quants.h"

#include "nx27v.h"
#include "my_msg.h"

static bool rpmsg_init = false;
static void *share_addr = nullptr;
static int datafd[RPMSG_EPT_NUM];

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

    assert(n * sizeof(block_q4_0) <= SHARE_MEM_SRC1_OFFSET - SHARE_MEM_SRC0_OFFSET && "Input size exceeds shared memory capacity");
    assert(n * sizeof(block_q8_0) <= SHARE_MEM_SRC1_OFFSET - SHARE_MEM_SRC0_OFFSET && "Input size exceeds shared memory capacity");


    printf("[NX27V] Thread %d: Performing vector dot product of %d elements using NX27V accelerator...\n", ith_thread, n);
    // float ref_result = 0.0f;
    // ggml_vec_dot_q4_0_q8_0_generic(n, &ref_result, bs, vx, bx, vy, by, nrc);
    // return;

    uint64_t src0_addr_offset = SHARE_MEM_SRC0_OFFSET + ith_thread * SHARE_MEM_SIZE_PER_THREAD;
    uint64_t src1_addr_offset = SHARE_MEM_SRC1_OFFSET + ith_thread * SHARE_MEM_SIZE_PER_THREAD;

    memcpy((void *)(share_addr + src0_addr_offset), vx, n * sizeof(block_q4_0));
    memcpy((void *)(share_addr + src1_addr_offset), vy, n * sizeof(block_q8_0));

    my_msg msg = {
        .opcode = MSG_OP_VEC_DOT_Q4_0_Q8_0,
        .src0_ne = { (uint32_t)(n), 1, 1, 1 },
        .src1_ne = { (uint32_t)(n), 1, 1, 1 },
        .src0_addr = (uint64_t)(SHARE_MEM_ADDR + src0_addr_offset),
        .src1_addr = (uint64_t)(SHARE_MEM_ADDR + src1_addr_offset),
        .result_addr = 0, // not used for dot product
    };

    // printf("[thread %d] Sending message to accelerator: opcode=%d, src0_ne=%u,%u,%u,%u, src1_ne=%u,%u,%u,%u, src0_addr=0x%lx, src1_addr=0x%lx\n",
    //        ith_thread,
    //        msg.opcode,
    //        msg.src0_ne[0], msg.src0_ne[1], msg.src0_ne[2], msg.src0_ne[3],
    //        msg.src1_ne[0], msg.src1_ne[1], msg.src1_ne[2], msg.src1_ne[3],
    //        msg.src0_addr,
    //        msg.src1_addr);

    ssize_t ret = write(datafd[ith_thread], &msg, sizeof(msg));
    assert(ret == sizeof(msg) && "Failed to send message to accelerator");

    // printf("[thread %d] Message sent to accelerator, waiting for response...\n", ith_thread);

    my_msg_ret_status ret_msg;
    ret = read(datafd[ith_thread], &ret_msg, sizeof(ret_msg));
    assert(ret == sizeof(ret_msg) && "Failed to read response from accelerator");
    assert(ret_msg.status == 0 && "Accelerator reported error");

    // // printf("[thread %d] Received response from accelerator: status=%d, dot_result=%f\n",
    //        ith_thread,
    //        ret_msg.status,
    //        ret_msg.dot_result);

    *s = ret_msg.dot_result;

    // Verify result against reference
    // if (fabsf(*s - ref_result) > 1e-3f) {
    //     fprintf(stderr, "[NX27V] Warning: Result mismatch! Accelerator result = %f, Reference result = %f\n", *s, ref_result);
    // } else {
    //     // GGML_LOG_INFO("[NX27V] Result verified: %f\n", *s);
    // }
}
