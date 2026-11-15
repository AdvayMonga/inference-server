// The decode step as an instruction interpreter (routing path "framework", megakernel/framework.py): one persistent
// cooperative kernel, one block per SM, each block running its own queue of instructions from a schedule built in
// Python at model load. An instruction waits on up to two dependency counters (each until every instruction that
// signals it has run), runs, and signals one; no grid-wide barrier inside the step. What an opcode does is the
// program's: its header (mega_ops.cuh, found on the build's include path) defines run_op and program_smem, so a
// search swaps a program (schedule + instruction code) without touching this file. The step ends with one grid
// barrier, after which block 0 zeroes the counters for the next launch.
#include "common.cuh"

struct __align__(64) Instr {            // 16 ints, written by the schedule; a program's router may rewrite args on the GPU
    int op, layer;
    int wait0, need0, wait1, need1;     // counters waited on (-1: none) and the count each must reach
    int signal;                         // the counter bumped when done (-1: none)
    int arg[9];
};
static_assert(sizeof(Instr) == 64, "Instr is 16 ints (framework.FIELDS)");

struct Args {
    const int64_t* tokens; const bf16* embed; const int64_t* pos; const int64_t* bt; int bt_cols;   // bt [B, cols], every layer
    const Layer* layers; int n_layers; const bf16* final_norm; const bf16* lm_head; int V; bf16* logits;
    unsigned long long* keys; int64_t* tokens_out;      // greedy argmax: packed (ordered bf16, ~index) per row, the token ids
    int B; float scale; const float* inv_freq; Scratch sc;
    Instr* instrs; const int* offsets; int* counters; int n_counters;   // the schedule: queue j = instrs[offsets[j], offsets[j+1])
};

struct Ctx {                            // what an instruction sees besides its own fields
    const Args& a;
    float* red; unsigned char* smem; unsigned long long* bars; unsigned char* ring;   // decode_step.cu's shared memory layout
    uint32_t ph;                        // the weight ring's slot parities, kept across instructions
    int nsplit;                         // attention chunks of the step's longest row
    int at;                             // the running instruction's index in instrs
};

#include <mega_ops.cuh>

__device__ __forceinline__ void wait_counter(const Args& a, int c, int need) {
    if (c < 0) return;
    if (threadIdx.x == 0) {
        const volatile int* p = a.counters + c;
        while (*p < need) {}
        __threadfence();
    }
    __syncthreads();
}

template <int BT, bool FP8>
__global__ void __launch_bounds__(THREADS, 1) interp_kernel(Args a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Ctx c{a, reinterpret_cast<float*>(smem_raw), smem_raw + HDR, reinterpret_cast<unsigned long long*>(smem_raw + 128),
          smem_raw + RING_OFF, 0u, 0, 0};
    int maxlen = 0;
    for (int b = 0; b < a.B; ++b) maxlen = max(maxlen, (int)a.pos[b] + 1);
    c.nsplit = (maxlen + CHUNK - 1) / CHUNK;
    if constexpr (MEGA_TMA && BT <= 2) {
        if (threadIdx.x == 0) {
            for (int k = 0; k < NST; ++k) mbar_init(c.bars + k);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        __syncthreads();
    }
    const int end = a.offsets[blockIdx.x + 1];
    for (int i = a.offsets[blockIdx.x]; i < end; ++i) {
        // A plain load, L1 cached: only the schedule and this block's own router write this queue, and an SM's L1
        // sees its own stores. The next instruction is prefetched into L1 once this one's waits (and their fences) are over.
        const Instr in = a.instrs[i];
        c.at = i;
        run_op<BT, FP8>(c, in, [&] {
            wait_counter(a, in.wait0, in.need0);
            wait_counter(a, in.wait1, in.need1);
            if (threadIdx.x == 0 && i + 1 < end) asm volatile("prefetch.global.L1 [%0];" :: "l"(a.instrs + i + 1));
        });
        __syncthreads();                                              // the block's writes, and its shared memory, done
        if (threadIdx.x == 0 && in.signal >= 0) { __threadfence(); atomicAdd(a.counters + in.signal, 1); }
    }
    cg::this_grid().sync();
    if (blockIdx.x == 0) for (int k = threadIdx.x; k < a.n_counters; k += THREADS) a.counters[k] = 0;
}

// ------------------------------------------------------------------------------------------------ host side
template <bool FP8> static void* pick(int B) {
    return B <= 1 ? (void*)interp_kernel<1, FP8> : B <= 2 ? (void*)interp_kernel<2, FP8> : B <= 4 ? (void*)interp_kernel<4, FP8>
         : B <= 8 ? (void*)interp_kernel<8, FP8> : (void*)interp_kernel<16, FP8>;
}
static size_t program_smem_for(int B) {
    return B <= 1 ? program_smem<1>() : B <= 2 ? program_smem<2>() : B <= 4 ? program_smem<4>() : B <= 8 ? program_smem<8>() : program_smem<16>();
}

std::tuple<torch::Tensor, torch::Tensor> decode_step(torch::Tensor tokens, torch::Tensor pos, torch::Tensor bt, torch::Tensor layers, torch::Tensor embed,
                          torch::Tensor final_norm, torch::Tensor lm_head, torch::Tensor inv_freq, torch::Tensor scratch_f32,
                          torch::Tensor scratch_bf16, torch::Tensor scratch_i32, double scale, bool fp8,
                          torch::Tensor instrs, torch::Tensor offsets, torch::Tensor counters) {
    TORCH_CHECK(tokens.is_cuda() && tokens.dtype() == torch::kInt64 && tokens.dim() == 1 && tokens.is_contiguous(), "tokens [B] int64");
    const int B = tokens.size(0);
    TORCH_CHECK(B >= 1 && B <= BMAX, "batch 1..16");
    TORCH_CHECK(pos.dtype() == torch::kInt64 && pos.numel() == B && pos.is_contiguous(), "pos [B] int64");
    TORCH_CHECK(bt.dtype() == torch::kInt64 && bt.dim() == 2 && bt.size(0) == B && bt.is_contiguous(), "bt [B, cols] int64");
    TORCH_CHECK(layers.dtype() == torch::kInt64 && layers.dim() == 2 && layers.size(1) == 13 && layers.is_cuda() && layers.is_contiguous(), "layers [L, 13] int64");
    TORCH_CHECK(bt.size(1) * BS <= MAXSPLIT * CHUNK, "context longer than MAXSPLIT * CHUNK");
    TORCH_CHECK(embed.dtype() == torch::kBFloat16 && embed.size(1) == H && lm_head.dtype() == torch::kBFloat16 && lm_head.size(1) == H, "embed/lm_head [V, H] bf16");
    TORCH_CHECK(instrs.is_cuda() && instrs.dtype() == torch::kInt32 && instrs.dim() == 2 && instrs.size(1) == 16 && instrs.is_contiguous(), "instrs [N, 16] int32");
    TORCH_CHECK(offsets.is_cuda() && offsets.dtype() == torch::kInt32 && offsets.numel() == num_sms() + 1, "offsets [SMs + 1] int32: one queue per SM");
    TORCH_CHECK(counters.is_cuda() && counters.dtype() == torch::kInt32 && counters.is_contiguous(), "counters int32");
    const int V = lm_head.size(0);
    auto logits = torch::empty({B, V}, tokens.options().dtype(torch::kBFloat16));
    auto next = torch::empty({B}, tokens.options());
    Args a;
    a.tokens = tokens.data_ptr<int64_t>(); a.embed = reinterpret_cast<const bf16*>(embed.data_ptr());
    a.pos = pos.data_ptr<int64_t>(); a.bt = bt.data_ptr<int64_t>(); a.bt_cols = bt.size(1);
    a.keys = reinterpret_cast<unsigned long long*>(scratch_i32.data_ptr<int>() + BMAX * K); a.tokens_out = next.data_ptr<int64_t>();
    a.layers = reinterpret_cast<const Layer*>(layers.data_ptr<int64_t>()); a.n_layers = layers.size(0);
    a.final_norm = reinterpret_cast<const bf16*>(final_norm.data_ptr()); a.lm_head = reinterpret_cast<const bf16*>(lm_head.data_ptr());
    a.V = V; a.logits = reinterpret_cast<bf16*>(logits.data_ptr()); a.B = B; a.scale = (float)scale; a.inv_freq = inv_freq.data_ptr<float>();
    a.sc = carve_scratch(scratch_f32.data_ptr<float>(), reinterpret_cast<bf16*>(scratch_bf16.data_ptr()), scratch_i32.data_ptr<int>());
    a.instrs = reinterpret_cast<Instr*>(instrs.data_ptr<int>()); a.offsets = offsets.data_ptr<int>();
    a.counters = counters.data_ptr<int>(); a.n_counters = counters.numel();
    static bool attr_set = false;
    if (!attr_set) {
        for (int b : {1, 2, 4, 8, 16}) {
            cudaFuncSetAttribute(pick<true>(b), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)program_smem_for(b));
            cudaFuncSetAttribute(pick<false>(b), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)program_smem_for(b));
        }
        attr_set = true;
    }
    void* args[] = {&a};
    cudaError_t err = cudaLaunchCooperativeKernel(fp8 ? pick<true>(B) : pick<false>(B), dim3(num_sms()), dim3(THREADS), args,
                                                  program_smem_for(B), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "cooperative launch failed: ", cudaGetErrorString(err));
    return {logits, next};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_step", &decode_step);
    m.def("scratch_f32_elems", &scratch_f32_elems);
    m.def("scratch_bf16_elems", &scratch_bf16_elems);
    m.def("scratch_i32_elems", &scratch_i32_elems);
}
