#include "Player.hpp"
#include <curand_kernel.h>
#include <curand.h>
#include <exception>
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*#define THREADS_PER_BLOCK 256 
#define BLOCKS 12*/
#define XORSHIFT(seed) (seed ^= (seed << 13), seed ^= (seed >> 17), seed ^= (seed << 5))
#define XORSHIFT_LCG(seed) \
    (seed ^= (seed << 13), \
     seed ^= (seed >> 17), \
     seed ^= (seed << 5),  \
     seed = 1664525 * seed + 1013904223)
#define XORWOW(x0, x1, x2, x3, x4, addend) \
    ( \
        x4 ^= (x4 >> 2), \
        x4 ^= (x4 << 1), \
        x4 ^= (x4 >> 4), \
        x4 ^= (x0 ^ (x0 << 1)), \
        (x0 = x1, x1 = x2, x2 = x3, x3 = x4, x4 = x0 ^ x4), \
        (addend += 362437), \
        (x4 + addend) \
    )

__global__ void SimulateFromNodeKernel(
    uint32_t playerPieces,
    uint32_t oppoentPieces,
    uint32_t promotedPieces,
    uint8_t nonAdvancingMoves,
    uint8_t* results,
    unsigned int seed,
    uint64_t* globalSum)
{
    __shared__ uint64_t blockSum; // Wspó³dzielona suma w bloku
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Inicjalizacja stanu XORWOW dla ka¿dego w¹tku
    uint32_t x0 = seed + idx;
    uint32_t x1 = seed ^ 0x12345678;
    uint32_t x2 = seed + 0x87654321;
    uint32_t x3 = seed ^ 0xAABBCCDD;
    uint32_t x4 = seed + 0xDEADBEEF;
    uint32_t addend = 362437;

    if (threadIdx.x == 0) {
        blockSum = 0;
    }
    __syncthreads();

    uint8_t moveNum;
    uint32_t move;
    CUDA_Vector<uint32_t> availableMoves;
    int i = 0;

    uint8_t localResult = 0; 

    while (true) {
        if (nonAdvancingMoves > MAX_NON_ADVANCING_MOVES) {
            localResult = 1;
            break;
        }

        i++;

        availableMoves = GeneratePossibleMoves(playerPieces, oppoentPieces, promotedPieces);

        if (availableMoves.size() == 0) {
            localResult = 0;
            break;
        }

        // Generowanie losowej liczby za pomoc¹ XORWOW
        uint32_t randomNumber = XORWOW(x0, x1, x2, x3, x4, addend);
        moveNum = randomNumber % availableMoves.size();

        move = availableMoves.at(moveNum);
        if (MakeMove(playerPieces, oppoentPieces, promotedPieces, move)) {
            nonAdvancingMoves = 0;
        }
        else {
            nonAdvancingMoves++;
        }

        if (nonAdvancingMoves > MAX_NON_ADVANCING_MOVES) {
            localResult = 1;
            break;
        }

        playerPieces = REVERSE_BITS(playerPieces);
        oppoentPieces = REVERSE_BITS(oppoentPieces);
        promotedPieces = REVERSE_BITS(promotedPieces);

        availableMoves = GeneratePossibleMoves(oppoentPieces, playerPieces, promotedPieces);
        if (availableMoves.size() == 0) {
            localResult = 2;
            break;
        }

        randomNumber = XORWOW(x0, x1, x2, x3, x4, addend);
        moveNum = randomNumber % availableMoves.size();
        move = availableMoves.at(moveNum);

        if (MakeMove(oppoentPieces, playerPieces, promotedPieces, move)) {
            nonAdvancingMoves = 0;
        }
        else {
            nonAdvancingMoves++;
        }

        playerPieces = REVERSE_BITS(playerPieces);
        oppoentPieces = REVERSE_BITS(oppoentPieces);
        promotedPieces = REVERSE_BITS(promotedPieces);
    }

    results[idx] = localResult;
    atomicAdd(&blockSum, localResult);

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(globalSum, blockSum);
    }
}
__global__ void InitializeCurandStates(curandState_t* states, unsigned long long seed) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &states[idx]);
}

class PlayerGPU : public Player
{
	curandState_t* d_states; 
	uint8_t* d_results;   
    uint64_t* d_globalSum;
	unsigned long seed;
	uint8_t* h_results;
    int threadsPerBlock;
    int blocks;
public:
	PlayerGPU(bool isWhite, uint32_t timePerMove): Player(isWhite, timePerMove,0),seed(time(NULL))
	{     
        cudaDeviceProp deviceProp;
        int device;

       
        // Get the active device
        if (cudaGetDevice(&device) != cudaSuccess)
            throw std::runtime_error("Failed to get current device.");

        // Get device properties
        if (cudaGetDeviceProperties(&deviceProp, device) != cudaSuccess)
            throw std::runtime_error("Failed to get device properties.");
        threadsPerBlock = deviceProp.maxThreadsPerBlock/4;
        int maxThreads = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount;
        blocks = maxThreads / threadsPerBlock/2;

        std::cout << "Using " << blocks << " blocks with " << threadsPerBlock << " threads each." << std::endl;

        simulationsPerIteration = threadsPerBlock * blocks;
        h_results = new uint8_t[simulationsPerIteration];

        if (cudaMalloc(&d_states, simulationsPerIteration * sizeof(curandState_t)) != cudaSuccess)
            throw std::runtime_error("Failed to allocate memory for d_states.");

        if (cudaMalloc(&d_results, simulationsPerIteration * sizeof(uint8_t)) != cudaSuccess)
            throw std::runtime_error("Failed to allocate memory for d_results.");

        if (cudaMalloc(&d_globalSum, sizeof(uint64_t)) != cudaSuccess)
            throw std::runtime_error("Failed to allocate memory for d_globalSum.");

        if (cudaMemset(d_globalSum, 0, sizeof(uint64_t)) != cudaSuccess)
            throw std::runtime_error("Failed to initialize d_globalSum with cudaMemset.");

        if (cudaDeviceSynchronize() != cudaSuccess)
            throw std::runtime_error("Failed to synchronize device after initialization.");
	}
    ~PlayerGPU() {
        if (d_states) cudaFree(d_states);
        if (d_results) cudaFree(d_results);
        if (d_globalSum) cudaFree(d_globalSum);
        delete[] h_results;
    }
	void Simulation(treeNode* node) override {

		unsigned int seed = rand();
        cudaMemset(d_globalSum, 0, sizeof(uint64_t));

        SimulateFromNodeKernel << <blocks, threadsPerBlock >> > (
            node->playerPieces,
            node->opponentPieces,
            node->promotedPieces,
            node->nonAdvancingMoves,
            d_results,
            seed,
            d_globalSum
            );
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) 
            throw std::runtime_error(std::string("Kernel execution failed: ") + cudaGetErrorString(kernelErr));

        if (cudaDeviceSynchronize() != cudaSuccess) 
            throw std::runtime_error("Failed to synchronize device after kernel execution.");

        uint64_t h_globalSum = 0;
        if (cudaMemcpy(&h_globalSum, d_globalSum, sizeof(uint64_t), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("Failed to copy d_globalSum to host memory.");
       
        if (cudaMemcpy(h_results, d_results, simulationsPerIteration * sizeof(uint8_t), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("Failed to copy d_results to host memory.");
       


		node->totalPoints += h_globalSum;
		node->gamesPlayed += simulationsPerIteration;
	}
};

