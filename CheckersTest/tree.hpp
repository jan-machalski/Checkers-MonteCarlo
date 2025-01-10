#include<iostream>
#include<stdlib.h>
#include<vector>
#include<queue>
#include<cstdlib>

#include"containers.hpp"

#define BIT(a, b)   (bool)((a>>b)&1)

#define SET_BIT(a, b, v)    a &= (~(1 << (b))); \
                            a |= ((v) << (b));

#define REVERSE_BITS(b)  b = (b & 0xFFFF0000) >> 16 | (b & 0x0000FFFF) << 16; \
                        b = (b & 0xFF00FF00) >> 8 | (b & 0x00FF00FF) << 8; \
                        b = (b & 0xF0F0F0F0) >> 4 | (b & 0x0F0F0F0F) << 4; \
                        b = (b & 0xCCCCCCCC) >> 2 | (b & 0x33333333) << 2; \
                        b = (b & 0xAAAAAAAA) >> 1 | (b & 0x55555555) << 1;

#define MINUS3_AVAILABLE 0x07070700
#define MINUS5_AVAILABLE 0xE0E0E0E0
#define PLUS3_AVAILABLE  0x00E0E0E0
#define PLUS5_AVAILABLE  0x07070707
#define MINUS4_AVAILABLE 0xFFFFFFF0
#define PLUS4_AVAILABLE  0x0FFFFFFF
#define RIGHT_UP_CAPTURE_AVAILABLE 0x77777700
#define LEFT_UP_CAPTURE_AVAILABLE  0xEEEEEE00
#define RIGHT_DOWN_CAPTURE_AVAILABLE 0x00777777
#define LEFT_DOWN_CAPTURE_AVAILABLE  0x00EEEEEE
#define DIAGONAL_LEFT_END  0x10101010
#define DIAGONAL_RIGHT_END 0x08080808
#define PROMOTION_SQUARES 0x0000000F
#define PLAYER_PIECES_INIT 0xFFF00000
#define OPPONENT_PIECES_INIT 0x00000FFF


struct treeNode {
	uint32_t playerPieces = PLAYER_PIECES_INIT; // player to move from the node
	uint32_t opponentPieces = OPPONENT_PIECES_INIT; // opponent of the player who has the move
	uint32_t promotedPieces = 0;
	treeNode* parent;
	std::vector<treeNode*> children;

	bool playerForTheWin = true;
	uint8_t nonAdvancingMoves = 0; // moves without take or advancing
	uint64_t gamesPlayed = 0;
	uint64_t totalPoints = 0; // 2 points for win, 1 point for draw, 0 points for loss
};

void deleteNode(treeNode* node)
{
	if (node == nullptr) return;
	for (auto child : node->children)
		deleteNode(child);
	delete node;
}


__host__ __device__ CUDA_Vector<uint32_t> GetAllCaptures(CUDA_Queue<uint32_t> captureQueue, uint32_t playerPieces, uint32_t opponentPieces, uint32_t promotedPieces)
{
	CUDA_Vector<uint32_t> fullCaptures;
	uint32_t  it, capturedPawn;
	bool furtherCaptures;

	while (!captureQueue.empty())
	{
		furtherCaptures = false;
		uint32_t currentCapture = captureQueue.front();
		captureQueue.pop();
		uint32_t capturableOpponentPieces = opponentPieces & ~(currentCapture);
		uint32_t currentPos = (currentCapture & (~(playerPieces | opponentPieces))) ? (currentCapture & (~(playerPieces | opponentPieces))) : currentCapture & (~opponentPieces);
		uint32_t freeTiles = (~(playerPieces | opponentPieces) | (playerPieces & currentCapture)) & (~currentPos);

		if (promotedPieces & playerPieces & currentCapture)
		{
			// capture right and up
			it = (MINUS3_AVAILABLE & currentPos) ? currentPos >> 3 : (currentPos & DIAGONAL_RIGHT_END ? 0 : currentPos >> 4);
			capturedPawn = 0;
			while (it)
			{
				if ((it & freeTiles) && capturedPawn)
				{
					furtherCaptures = true;
					captureQueue.push((currentCapture & playerPieces) | (opponentPieces & currentCapture) | capturedPawn | it);
				}
				else if (it & capturableOpponentPieces && !capturedPawn)
					capturedPawn = it;
				else if (!(it & freeTiles))
					break;
				it = (MINUS3_AVAILABLE & it) ? it >> 3 : (it & DIAGONAL_RIGHT_END ? 0 : it >> 4);
			}

			// capture left and up
			it = (MINUS5_AVAILABLE & currentPos) ? currentPos >> 5 : (currentPos & DIAGONAL_LEFT_END ? 0 : currentPos >> 4);
			capturedPawn = 0;
			while (it)
			{
				if (it & freeTiles && capturedPawn)
				{
					furtherCaptures = true;
					captureQueue.push((currentCapture & playerPieces) | (opponentPieces & currentCapture) | capturedPawn | it);
				}
				else if (it & capturableOpponentPieces && !capturedPawn)
					capturedPawn = it;
				else if (!(it & freeTiles))
					break;
				it = (MINUS5_AVAILABLE & it) ? it >> 5 : (it & DIAGONAL_LEFT_END ? 0 : it >> 4);
			}

			// captures right and down
			it = (PLUS5_AVAILABLE & currentPos) ? currentPos << 5 : (currentPos & DIAGONAL_RIGHT_END ? 0 : currentPos << 4);
			capturedPawn = 0;
			while (it)
			{
				if (it & freeTiles && capturedPawn)
				{
					furtherCaptures = true;
					captureQueue.push((currentCapture & playerPieces) | (opponentPieces & currentCapture) | capturedPawn | it);
				}
				else if (it & capturableOpponentPieces && !capturedPawn)
					capturedPawn = it;
				else if (!(it & freeTiles))
					break;
				it = (PLUS5_AVAILABLE & it) ? it << 5 : (it & DIAGONAL_RIGHT_END ? 0 : it << 4);
			}

			// captures left and down
			it = (PLUS3_AVAILABLE & currentPos) ? currentPos << 3 : (currentPos & DIAGONAL_LEFT_END ? 0 : currentPos << 4);
			capturedPawn = 0;
			while (it)
			{
				if (it & freeTiles && capturedPawn)
				{
					furtherCaptures = true;
					captureQueue.push((currentCapture & playerPieces) | (opponentPieces & currentCapture) | capturedPawn | it);
				}
				else if (it & capturableOpponentPieces && !capturedPawn)
					capturedPawn = it;
				else if (!(it & freeTiles))
					break;
				it = (PLUS3_AVAILABLE & it) ? it << 3 : (it & DIAGONAL_LEFT_END ? 0 : it << 4);
			}
		}
		else
		{
			// right up capture
			if (currentPos & RIGHT_UP_CAPTURE_AVAILABLE)
			{
				if (MINUS3_AVAILABLE & currentPos && MINUS4_AVAILABLE & (currentPos >> 3) && capturableOpponentPieces & (currentPos >> 3) && freeTiles & (currentPos >> 7))
				{
					furtherCaptures = true;
					uint32_t newCapture = (currentCapture & playerPieces) | (opponentPieces & currentCapture) | (currentPos >> 3) | (currentPos >> 7);
					captureQueue.push(newCapture);
				}
				else if (MINUS4_AVAILABLE & currentPos && MINUS3_AVAILABLE & (currentPos >> 4) && capturableOpponentPieces & (currentPos >> 4) && freeTiles & (currentPos >> 7))
				{
					furtherCaptures = true;
					uint32_t newCapture = (currentCapture & playerPieces) | (opponentPieces & currentCapture) | (currentPos >> 4) | (currentPos >> 7);
					captureQueue.push(newCapture);
				}
			}
			// up left capture
			if (currentPos & LEFT_UP_CAPTURE_AVAILABLE)
			{
				if (MINUS5_AVAILABLE & currentPos && MINUS4_AVAILABLE & (currentPos >> 5) && capturableOpponentPieces & (currentPos >> 5) && freeTiles & (currentPos >> 9))
				{
					furtherCaptures = true;
					uint32_t newCapture = (currentCapture & playerPieces) | (opponentPieces & currentCapture) | (currentPos >> 5) | (currentPos >> 9);
					captureQueue.push(newCapture);
				}
				else if (MINUS4_AVAILABLE & currentPos && MINUS5_AVAILABLE & (currentPos >> 4) && capturableOpponentPieces & (currentPos >> 4) && freeTiles & (currentPos >> 9))
				{
					furtherCaptures = true;
					uint32_t newCapture = (currentCapture & playerPieces) | (opponentPieces & currentCapture) | (currentPos >> 4) | (currentPos >> 9);
					captureQueue.push(newCapture);
				}
			}
			// down right capture
			if (currentPos & RIGHT_DOWN_CAPTURE_AVAILABLE)
			{
				if (PLUS5_AVAILABLE & currentPos && PLUS4_AVAILABLE & (currentPos << 5) && capturableOpponentPieces & (currentPos << 5) && freeTiles & (currentPos << 9))
				{
					furtherCaptures = true;
					uint32_t newCapture = (currentCapture & playerPieces) | (opponentPieces & currentCapture) | (currentPos << 5) | (currentPos << 9);
					captureQueue.push(newCapture);
				}
				else if (PLUS4_AVAILABLE & currentPos && PLUS5_AVAILABLE & (currentPos << 4) && capturableOpponentPieces & (currentPos << 4) && freeTiles & (currentPos << 9))
				{
					furtherCaptures = true;
					uint32_t newCapture = (currentCapture & playerPieces) | (opponentPieces & currentCapture) | (currentPos << 4) | (currentPos << 9);
					captureQueue.push(newCapture);
				}
			}
			// down left capture
			if (currentPos & LEFT_DOWN_CAPTURE_AVAILABLE)
			{
				if (PLUS3_AVAILABLE & currentPos && PLUS4_AVAILABLE & (currentPos << 3) && capturableOpponentPieces & (currentPos << 3) && freeTiles & (currentPos << 7))
				{
					furtherCaptures = true;
					uint32_t newCapture = (currentCapture & playerPieces) | (opponentPieces & currentCapture) | (currentPos << 3) | (currentPos << 7);
					captureQueue.push(newCapture);
				}
				else if (PLUS4_AVAILABLE & currentPos && PLUS3_AVAILABLE & (currentPos << 4) && capturableOpponentPieces & (currentPos << 4) && freeTiles & (currentPos << 7))
				{
					furtherCaptures = true;
					uint32_t newCapture = (currentCapture & playerPieces) | (opponentPieces & currentCapture) | (currentPos << 4) | (currentPos << 7);
					captureQueue.push(newCapture);
				}
			}

		}
		if (!furtherCaptures)
			fullCaptures.push_back(currentCapture);
	}

	return fullCaptures;
}

__host__ __device__ bool AddMoveOfPromotedFigure(const uint32_t pos, uint32_t& it, uint32_t& capturedPawn, const uint32_t freeTiles, const uint32_t opponentPieces, CUDA_Vector<uint32_t>& movesWithoutCapture, CUDA_Queue<uint32_t>& captureQueue)
{
	if (it & freeTiles)
	{
		if (capturedPawn)
			captureQueue.push(pos | capturedPawn | it);
		else
			movesWithoutCapture.push_back(pos | it);
	}
	else
	{
		if (it & opponentPieces && !capturedPawn)
			capturedPawn = it;
		else
			return false;
	}
	return true;
}

__host__ __device__ CUDA_Vector<uint32_t> GeneratePossibleMoves(uint32_t playerPieces, uint32_t opponentPieces, uint32_t promotedPieces)
{
	CUDA_Vector<uint32_t> movesWithoutCapture;
	CUDA_Queue<uint32_t> captureQueue;
	uint32_t freeTiles = ~(playerPieces | opponentPieces);
	uint32_t it, capturedPawn;

	for (uint32_t pos = 1; pos != 0; pos <<= 1)
	{
		if (!(playerPieces & pos)) continue;

		if (promotedPieces & pos)
		{
			// moves and captures right and up
			it = (MINUS3_AVAILABLE & pos) ? pos >> 3 : (pos & DIAGONAL_RIGHT_END ? 0 : pos >> 4);
			capturedPawn = 0;
			while (it)
			{
				if (!AddMoveOfPromotedFigure(pos, it, capturedPawn, freeTiles, opponentPieces, movesWithoutCapture, captureQueue)) break;
				it = (MINUS3_AVAILABLE & it) ? it >> 3 : (it & DIAGONAL_RIGHT_END ? 0 : it >> 4);
			}

			// moves and captures left and up
			it = (MINUS5_AVAILABLE & pos) ? pos >> 5 : (pos & DIAGONAL_LEFT_END ? 0 : pos >> 4);
			capturedPawn = 0;
			while (it)
			{
				if (!AddMoveOfPromotedFigure(pos, it, capturedPawn, freeTiles, opponentPieces, movesWithoutCapture, captureQueue)) break;
				it = (MINUS5_AVAILABLE & it) ? it >> 5 : (it & DIAGONAL_LEFT_END ? 0 : it >> 4);
			}

			// moves and captures right and down
			it = (PLUS5_AVAILABLE & pos) ? pos << 5 : (pos & DIAGONAL_RIGHT_END ? 0 : pos << 4);
			capturedPawn = 0;
			while (it)
			{
				if (!AddMoveOfPromotedFigure(pos, it, capturedPawn, freeTiles, opponentPieces, movesWithoutCapture, captureQueue)) break;
				it = (PLUS5_AVAILABLE & it) ? it << 5 : (it & DIAGONAL_RIGHT_END ? 0 : it << 4);
			}

			// moves and captures left and down
			it = (PLUS3_AVAILABLE & pos) ? pos << 3 : (pos & DIAGONAL_LEFT_END ? 0 : pos << 4);
			capturedPawn = 0;
			while (it)
			{
				if (!AddMoveOfPromotedFigure(pos, it, capturedPawn, freeTiles, opponentPieces, movesWithoutCapture, captureQueue)) break;
				it = (PLUS3_AVAILABLE & it) ? it << 3 : (it & DIAGONAL_LEFT_END ? 0 : it << 4);
			}

		}
		else
		{
			// move or capture up where the field above has index smaller by 4
			if (pos >> 4)
			{
				// move without capture
				if (freeTiles & (pos >> 4))
				{
					movesWithoutCapture.push_back(pos | (pos >> 4));
				}
				// capture right up 
				else if (MINUS3_AVAILABLE & (pos >> 4) && opponentPieces & (pos >> 4) && freeTiles & (pos >> 7))
				{
					captureQueue.push(pos | (pos >> 4) | (pos >> 7));
				}
				// capture left up
				else if (MINUS5_AVAILABLE & (pos >> 4) && opponentPieces & (pos >> 4) && freeTiles & (pos >> 9))
				{
					captureQueue.push(pos | (pos >> 4) | (pos >> 9));
				}
			}
			// move or capture right up if the left up field has index smaller by 3
			if (pos & MINUS3_AVAILABLE)
			{
				if (freeTiles & (pos >> 3))
				{
					movesWithoutCapture.push_back(pos | (pos >> 3));
				}
				else if (opponentPieces & (pos >> 3) && freeTiles & (pos >> 7))
				{
					captureQueue.push(pos | (pos >> 3) | pos >> 7);
				}
			}
			// move or capture left up if the left up field has index smaller by 5
			if (pos & MINUS5_AVAILABLE)
			{
				if (freeTiles & (pos >> 5))
				{
					movesWithoutCapture.push_back(pos | (pos >> 5));
				}
				else if (opponentPieces & (pos >> 5) && freeTiles & (pos >> 9))
				{
					captureQueue.push(pos | (pos >> 5) | (pos >> 9));
				}
			}
			// captures down
			if (pos & PLUS5_AVAILABLE && opponentPieces & (pos << 5) && freeTiles & (pos << 9))
			{
				captureQueue.push(pos | (pos << 5) | (pos << 9));
			}
			if (PLUS3_AVAILABLE & pos && opponentPieces & (pos << 3) && freeTiles & (pos << 7))
			{
				captureQueue.push(pos | (pos << 3) | (pos << 7));
			}
			if (PLUS5_AVAILABLE & (pos << 4) && opponentPieces & (pos << 4) && freeTiles & (pos << 9))
			{
				captureQueue.push(pos | (pos << 4) | (pos << 9));
			}
			if (PLUS3_AVAILABLE & (pos << 4) && opponentPieces & (pos << 4) && freeTiles & (pos << 7))
			{
				captureQueue.push(pos | (pos << 4) | (pos << 7));
			}
		}
	}
	if (captureQueue.empty())
		return movesWithoutCapture;
	return GetAllCaptures(captureQueue, playerPieces, opponentPieces, promotedPieces);

}

void printBoard(uint32_t playerPieces, uint32_t opponentPieces, uint32_t promotedPieces, bool whitePlayer = true)
{
	std::string columns = " | A| B| C| D| E| F| G| H| ";
	if (!whitePlayer)
		reverse(columns.begin(), columns.end());
	std::string separator = "-+--+--+--+--+--+--+--+--+-";
	std::cout << std::endl;
	std::cout << '\t' << columns << std::endl;
	std::cout << '\t' << separator << std::endl;

	for (int i = 0; i < 8; i++)
	{
		std::cout << '\t' << (char)(whitePlayer ? 7 - i + 49 : i + 49) << '|';

		for (int j = 0; j < 4; j++)
		{
			if (i % 2 == 0)
				std::cout << "  |";
			if (playerPieces & (1 << (i * 4 + j)))
			{
				std::cout << (whitePlayer ? 'W' : 'B');
				if (promotedPieces & (1 << (i * 4 + j)))
					std::cout << (whitePlayer ? 'W' : 'B');
				else
					std::cout << ' ';
			}
			else if (opponentPieces & (1 << (i * 4 + j)))
			{
				std::cout << (whitePlayer ? 'B' : 'W');
				if (promotedPieces & (1 << (i * 4 + j)))
					std::cout << (whitePlayer ? 'B' : 'W');
				else
					std::cout << ' ';
			}
			else
				std::cout << "  ";
			if (i % 2 == 0)
				std::cout << '|';
			else
				std::cout << "|  |";
		}

		std::cout << (char)(whitePlayer ? 7 - i + 49 : i + 49) << std::endl;
		std::cout << '\t' << separator << std::endl;
	}

	std::cout << '\t' << columns << std::endl;
	std::cout << '\t' << std::endl;
}



__host__ __device__ bool MakeMove(uint32_t& playerPieces, uint32_t& opponentPieces, uint32_t& promotedPieces, uint32_t move)
{
	uint32_t startingSquare = playerPieces & move;
	uint32_t capturedPieces = move & opponentPieces;
	uint32_t endSquare = move & ~(playerPieces | opponentPieces) ? move & ~(playerPieces | opponentPieces) : move & ~opponentPieces;
	bool kingUsed = startingSquare & promotedPieces;

	opponentPieces &= (~capturedPieces);
	promotedPieces &= (~capturedPieces);
	playerPieces = (playerPieces & (~startingSquare)) | endSquare;
	promotedPieces |= endSquare & PROMOTION_SQUARES;
	if (promotedPieces & startingSquare)
		promotedPieces = (promotedPieces & (~startingSquare)) | endSquare;

	//return true if advancement or capture has been made
	return capturedPieces || !(kingUsed);
}
void GenerateChildren(treeNode* root)
{
	CUDA_Vector<uint32_t> availableMoves = GeneratePossibleMoves(root->playerPieces, root->opponentPieces, root->promotedPieces);
	for (uint32_t move : availableMoves)
	{
		treeNode* newChild = new treeNode();
		newChild->playerPieces = root->opponentPieces;
		newChild->opponentPieces = root->playerPieces;
		newChild->promotedPieces = root->promotedPieces;
		if (MakeMove(newChild->opponentPieces, newChild->playerPieces, newChild->promotedPieces, move))
			newChild->nonAdvancingMoves = 0;
		else
			newChild->nonAdvancingMoves = root->nonAdvancingMoves + 1;
		newChild->playerPieces = REVERSE_BITS(newChild->playerPieces);
		newChild->opponentPieces = REVERSE_BITS(newChild->opponentPieces);
		newChild->promotedPieces = REVERSE_BITS(newChild->promotedPieces);
		newChild->parent = root;
		newChild->playerForTheWin = !root->playerForTheWin;
		root->children.push_back(newChild);
	}
}

uint8_t SimulateFromNode(uint32_t playerPieces, uint32_t oppoentPieces, uint32_t promotedPieces, uint8_t nonAdvancingMoves)
{
	uint8_t moveNum;
	uint32_t move;
	CUDA_Vector<uint32_t> availableMoves;
	int i = 0;
	while (true)
	{
		if (nonAdvancingMoves > 10)
			return 1;
		i++;

		availableMoves = GeneratePossibleMoves(playerPieces, oppoentPieces, promotedPieces);

		if (availableMoves.size() == 0)
			return 0;

		moveNum = rand() % availableMoves.size();
		move = availableMoves.at(moveNum);
		if (MakeMove(playerPieces, oppoentPieces, promotedPieces, move))
			nonAdvancingMoves = 0;
		else
			nonAdvancingMoves++;

		if (nonAdvancingMoves > 10)
			return 1;

		playerPieces = REVERSE_BITS(playerPieces);
		oppoentPieces = REVERSE_BITS(oppoentPieces);
		promotedPieces = REVERSE_BITS(promotedPieces);


		availableMoves = GeneratePossibleMoves(oppoentPieces, playerPieces, promotedPieces);
		if (availableMoves.size() == 0)
			return 2;


		moveNum = rand() % availableMoves.size();
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


