#ifndef MY_MSG_H_
#define MY_MSG_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define RPMSG_EPT_NAME "llama_cpp_accel"
#define RPMSG_EPT_NUM 1

#define SHARE_MEM_ADDR            0x43f800000ULL
#define SHARE_MEM_SIZE            0x0007e4000ULL
#define SHARE_MEM_SIZE_PER_THREAD 0x000700000ULL // 1 threads supported

#define SHARE_MEM_SRC_MAT_SIZE    0x000260000ULL
#define SHARE_MEM_RESULT_MAT_SIZE 0x000040000ULL
#define SHARE_MEM_SRC0_OFFSET     0
#define SHARE_MEM_SRC1_OFFSET     SHARE_MEM_SRC_MAT_SIZE * 1
#define SHARE_MEM_RESULT_OFFSET   SHARE_MEM_SRC_MAT_SIZE * 2

const uint32_t RPMSG_EPT_ADDR[] = {
    0x10,
    0x11,
    0x12,
    0x13,
};

typedef enum {
    MSG_OP_MATRIX_MUL_FLOAT32 = 0,
    MSG_OP_VEC_DOT_Q4_0_Q8_0 = 1,
    MSG_OP_MAT_MUL_Q4_0_Q8_0 = 2,
    MSG_OP_SHUTDOWN = 255,

} my_msg_opcode_t;

struct my_msg {
	my_msg_opcode_t opcode;

    uint32_t src0_ne[4];
    uint32_t src1_ne[4];
    uint32_t src0_nb[4];
    uint32_t src1_nb[4];

	uint64_t src0_addr;
	uint64_t src1_addr;
	uint64_t result_addr;
};

struct my_msg_ret_status {
	my_msg_opcode_t opcode;
	uint32_t status; // 0: success, others: error code

    float dot_result; // for vector dot product result

  uint64_t perf_counter[4];
};

// === FROM GGML ===
#ifndef QK4_0

typedef uint16_t ggml_half;

#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

#endif // QK4_0

#ifdef __cplusplus
}
#endif

#endif /* MY_MSG_H_ */
