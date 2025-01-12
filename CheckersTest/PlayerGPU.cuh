#include "Player.hpp"
#include <curand_kernel.h>
#include <curand.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREADS_PER_BLOCK 256 
#define BLOCKS 4
#define XORSHIFT(seed) (seed ^= (seed << 13), seed ^= (seed >> 17), seed ^= (seed << 5))
#define XORSHIFT_LCG(seed) \
    (seed ^= (seed << 13), \
     seed ^= (seed >> 17), \
     seed ^= (seed << 5),  \
     seed = 1664525 * seed + 1013904223)

__global__ void SimulateFromNodeKernel(uint32_t playerPieces, uint32_t oppoentPieces, uint32_t promotedPieces, uint8_t nonAdvancingMoves, uint8_t* results, unsigned int seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int localSeed = seed + idx;

	uint8_t moveNum;
	uint32_t move;
	CUDA_Vector<uint32_t> availableMoves;
	int i = 0;
	while (true)
	{
		if (nonAdvancingMoves > 10)
		{
			results[idx] = 1;
			return;
		}

		i++;

		availableMoves = GeneratePossibleMoves(playerPieces, oppoentPieces, promotedPieces);

		if (availableMoves.size() == 0)
		{
			results[idx] = 0;
			return;
		}

		moveNum = XORSHIFT_LCG(localSeed) % availableMoves.size();
		move = availableMoves.at(moveNum);
		if (MakeMove(playerPieces, oppoentPieces, promotedPieces, move))
			nonAdvancingMoves = 0;
		else
			nonAdvancingMoves++;

		if (nonAdvancingMoves > 10)
		{
			results[idx] = 1;
			return;
		}

		playerPieces = REVERSE_BITS(playerPieces);
		oppoentPieces = REVERSE_BITS(oppoentPieces);
		promotedPieces = REVERSE_BITS(promotedPieces);


		availableMoves = GeneratePossibleMoves(oppoentPieces, playerPieces, promotedPieces);
		if (availableMoves.size() == 0)
		{
			results[idx] = 2;
			return;
		}

		moveNum = XORSHIFT_LCG(localSeed) % availableMoves.size();
		move = availableMoves.at(moveNum);

		if (MakeMove(oppoentPieces, playerPieces, promotedPieces, move))
			nonAdvancingMoves = 0;
		else
			nonAdvancingMoves++;

		playerPieces = REVERSE_BITS(playerPieces);
		oppoentPieces = REVERSE_BITS(oppoentPieces);
		promotedPieces = REVERSE_BITS(promotedPieces);

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
	unsigned long seed;
	uint8_t* h_results;
public:
	PlayerGPU(bool isWhite, uint32_t timePerMove): Player(isWhite, timePerMove,THREADS_PER_BLOCK*BLOCKS),seed(time(NULL))
	{
		h_results = new uint8_t[THREADS_PER_BLOCK * BLOCKS];
		cudaMalloc(&d_states, THREADS_PER_BLOCK * BLOCKS * sizeof(curandState_t));
		cudaMalloc(&d_results, THREADS_PER_BLOCK * BLOCKS * sizeof(uint8_t));
		//InitializeCurandStates << <BLOCKS, THREADS_PER_BLOCK >> > (d_states, seed);
		cudaDeviceSynchronize();
	}
	~PlayerGPU() {
		cudaFree(d_states);
		cudaFree(d_results);
	}
	void Simulation(treeNode* node) override {

		unsigned int seed = rand();
		SimulateFromNodeKernel << <BLOCKS, THREADS_PER_BLOCK >> > (
			node->playerPieces,
			node->opponentPieces,
			node->promotedPieces,
			node->nonAdvancingMoves,
			d_results,
			seed
			);
		cudaDeviceSynchronize();

		uint64_t totalPoints = thrust::reduce(thrust::device, d_results, d_results + THREADS_PER_BLOCK * BLOCKS, 0, thrust::plus<uint64_t>());

		node->totalPoints += totalPoints;
		node->gamesPlayed += THREADS_PER_BLOCK * BLOCKS;
	}
};

