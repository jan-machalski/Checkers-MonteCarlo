#include<algorithm>
#include<chrono>
#include<iomanip>

#include "tree.hpp"

//#define SHOW_EVALUATIONS 

class Player
{
public:
	const uint32_t timePerMove;
	const uint32_t simulationsPerIteration;
	const bool isWhite;
	struct treeNode* root;

	Player(bool isWhite, uint32_t timePerMove, uint32_t simulationsPerIteration) :isWhite(isWhite), timePerMove(timePerMove), simulationsPerIteration(simulationsPerIteration)
	{
		root = new treeNode();
		root->playerForTheWin = isWhite;
	}
	~Player()
	{
		deleteNode(root);
	}
	treeNode* Selection()
	{
		treeNode* p = root;
		treeNode* bestNode = nullptr;
		double bestScore = 0;
		while (!(p->children.empty()))
		{
			for (auto child : p->children)
			{
				if (child->gamesPlayed == 0)
				{
					bestNode = child;
					break;
				}

				double score = 2 * child->gamesPlayed - child->totalPoints;
				score = score / 2 / child->gamesPlayed + sqrt(2 * log(p->gamesPlayed) / child->gamesPlayed);
				if (score >= bestScore)
				{
					bestScore = score;
					bestNode = child;
				}
			}

			p = bestNode;
			bestNode = nullptr;
			bestScore = 0;
		}
		return p;
	}

	virtual void Simulation(treeNode* node) = 0;

	void BackPropagation(treeNode* node)
	{
		treeNode* p = node->parent;
		while (p)
		{
			if (p->playerForTheWin == node->playerForTheWin)
				p->totalPoints += node->totalPoints;
			else
				p->totalPoints += 2 * simulationsPerIteration - node->totalPoints;
			p->gamesPlayed += simulationsPerIteration;

			p = p->parent;
		}
	}
	void PerformMC()
	{
		treeNode* next = root;
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() < timePerMove)
		{
			next = Selection();
			GenerateChildren(next);
			if (next->children.size() > 0)
			{
				next = next->children.at(rand() % next->children.size());
				Simulation(next);
			}
			BackPropagation(next);
		}
	}
	bool MakeBestMove()
	{
		PerformMC();
		std::cout << "Total simulations: " << root->gamesPlayed << std::endl;
		std::cout << "Total points: " << root->totalPoints << std::endl;
		double evaluation = (double)root->totalPoints / root->gamesPlayed - 1;
		std::cout << "Evaluation: "<<std::fixed<<std::setprecision(3)<< evaluation << std::endl;
		if (root->children.size() == 0)
			return false;
		treeNode* bestChild = nullptr;
		uint64_t maxSim = 0;
		for (treeNode* c : root->children)
		{
			if (c->gamesPlayed > maxSim)
			{
				maxSim = c->gamesPlayed;
				bestChild = c;
			}
		}
		CUDA_Vector<std::pair<uint32_t, std::string>> availableMoves = GeneratePossibleMovesWithNotation(root->playerPieces, root->opponentPieces, root->promotedPieces, isWhite);
		uint32_t newPlayerPieces = reverseBits(bestChild->opponentPieces);
		uint32_t newOpponentPieces = reverseBits(bestChild->playerPieces);
		uint32_t newPromotedPieces = reverseBits(bestChild->promotedPieces);
		bool found = false;
		std::string bestMove;
		for (auto move : availableMoves)
		{
			uint32_t playerTmp = root->playerPieces;
			uint32_t opponentTmp = root->opponentPieces;
			uint32_t promotedTmp = root->promotedPieces;
			MakeMove(playerTmp, opponentTmp, promotedTmp, move.first);
#ifdef SHOW_EVALUATIONS
			std::cout << "Available moves and their evaluations:" << std::endl;
			for (auto c : root->children)
			{
				uint32_t playerTmp2 = reverseBits(c->opponentPieces);
				uint32_t opponentTmp2 = reverseBits(c->playerPieces);
				uint32_t promotedTmp2 = reverseBits(c->promotedPieces);
				if (playerTmp == playerTmp2 && opponentTmp == opponentTmp2 && promotedTmp == promotedTmp2)
				{
					std::cout << move.second <<" Evaluation: "<< - ((double)c->totalPoints / c->gamesPlayed - 1) << " Simulations: "<<c->gamesPlayed<<std::endl;
					break;
				}
			}
#endif
			if (playerTmp == newPlayerPieces && opponentTmp == newOpponentPieces && promotedTmp == newPromotedPieces && !found)
			{
				bestMove = move.second;
				found = true;
			}
		}
		std::cout << "Move: " << bestMove << std::endl;

		UpdateRoot(bestChild->playerPieces, bestChild->opponentPieces, bestChild->promotedPieces);
		return true;

	}
	void UpdateRoot(uint32_t playerPieces, uint32_t opponentPieces, uint32_t promotedPieces)
	{
		bool found = false;
		treeNode* result = nullptr;
		if (root->children.size() == 0)
			GenerateChildren(root);
		for (treeNode* c : root->children)
		{
			if (!found && playerPieces == c->playerPieces && opponentPieces == c->opponentPieces && promotedPieces == c->promotedPieces)
			{
				found = true;
				result = c;
			}
			else
				deleteNode(c);
		}
		if (result)
		{
			delete result->parent;
			root = result;
			root->parent = nullptr;
		}
		else
			std::cout << "INVALID MOVE ERROR" << std::endl;
	}

};

class PlayerCPU : public Player
{
public:
	PlayerCPU(bool isWhite, uint32_t timePerMove, uint32_t simulationsPerIteration) :Player(isWhite, timePerMove, simulationsPerIteration) {}

	void Simulation(treeNode* node) override
	{
		for (uint32_t i = 0; i < simulationsPerIteration; i++)
		{
			node->totalPoints += SimulateFromNode(node->playerPieces, node->opponentPieces, node->promotedPieces, node->nonAdvancingMoves);
		}
		node->gamesPlayed += simulationsPerIteration;
	}
};