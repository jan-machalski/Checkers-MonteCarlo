#include <fstream>

#include "PlayerGPU.cuh"

std::pair<uint32_t,std::string> InputMove(CUDA_Vector<std::pair<uint32_t,std::string>> availableMoves)
{
	uint32_t result = 0;
	std::string move;
	do
	{
		std::cout << "Input your move (e.g a4-b5 or d2:f4:d6)" << std::endl;
		std::cin >> move; 
		for (auto p : availableMoves)
		{
			if (p.second == move)
			{
				result = p.first;
				break;
			}
		}

	}while (result == 0);

	return std::make_pair(result,move);

}

std::unique_ptr<Player> selectPlayerMode(bool isWhite) {
	int choice;

	do {
		std::cout << "Choose the game mode for the " << (isWhite ? "White" : "Black") << " player:\n";
		std::cout << "1. Human\n";
		std::cout << "2. CPU\n";
		std::cout << "3. GPU\n";
		std::cout << "Your choice: ";
		std::cin >> choice;

		if (std::cin.fail() || (choice != 1 && choice != 2 && choice != 3)) {
			std::cin.clear(); // Clear the error flag
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
			std::cout << "Invalid choice. Please enter 1 (Human) or 2 (CPU) or 3 (GPU).\n";
		}
	} while (choice != 1 && choice != 2 && choice != 3);

	if (choice == 1) 
	{
		// Human player, return nullptr
		std::cout << "You have selected Human mode for the " << (isWhite ? "White" : "Black") << " player.\n";
		return nullptr;
	}
	else if(choice == 2)
	{
		// CPU player, ask for parameters
		uint32_t timePerMove;
		uint32_t simulationsPerIteration;

		do {
			std::cout << "Enter the maximum time per move (in miliseconds): ";
			std::cin >> timePerMove;

			if (std::cin.fail() || timePerMove == 0) {
				std::cin.clear();
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				std::cout << "Invalid input. Time per move must be a positive integer.\n";
			}
		} while (timePerMove == 0);

		do {
			std::cout << "Enter the number of simulations per iteration: ";
			std::cin >> simulationsPerIteration;

			if (std::cin.fail() || simulationsPerIteration == 0) {
				std::cin.clear();
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				std::cout << "Invalid input. Simulations per iteration must be a positive integer.\n";
			}
		} while (simulationsPerIteration == 0);

		std::cout << "You have selected CPU mode for the " << (isWhite ? "White" : "Black") << " player.\n";
		return std::make_unique<PlayerCPU>(isWhite, timePerMove, simulationsPerIteration);
	}
	else
	{
		uint32_t timePerMove;
		do {
			std::cout << "Enter the maximum time per move (in miliseconds): ";
			std::cin >> timePerMove;

			if (std::cin.fail() || timePerMove == 0) {
				std::cin.clear();
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				std::cout << "Invalid input. Time per move must be a positive integer.\n";
			}
		} while (timePerMove == 0);
		std::cout << "You have selected GPU mode for the " << (isWhite ? "White" : "Black") << " player.\n";
		return std::make_unique<PlayerGPU>(isWhite, timePerMove);
	}
}

void saveGameToFile(const std::vector<std::string>& moves, const std::string& filename) {
	std::ofstream outFile(filename);

	if (!outFile.is_open()) {
		std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
		return;
	}
	printf("Saving game to %s\n", filename.c_str());
	for (size_t i = 0; i < moves.size(); i += 2) {
		outFile << (i / 2 + 1) << ". "; 
		outFile << moves[i];           
		if (i + 1 < moves.size()) {
			outFile << " " << moves[i + 1]; 
		}
		outFile << "\n"; 
	}

	outFile.close();
	std::cout << "Game saved to " << filename << std::endl;
}


void PrintMoveInfo(uint32_t move)
{
	for (int i = 0; i < 32; i++)
		if (BIT(move, i))
			std::cout << i << " ";
	std::cout << std::endl;
}

void UpdateBoard(treeNode* root, uint32_t& playerPieces, uint32_t& opponentPieces, uint32_t& promotedPieces, uint32_t& playerPiecesOpposite, uint32_t& opponentPiecesOpposite, uint32_t& promotedPiecesOpposite)
{
	playerPieces = root->playerPieces;
	opponentPieces = root->opponentPieces;
	promotedPieces = root->promotedPieces;
	playerPiecesOpposite = reverseBits(playerPieces);
	opponentPiecesOpposite = reverseBits(opponentPieces);
	promotedPiecesOpposite = reverseBits(promotedPieces);
}



int main(int argc, char* argv[])
{
	std::string outputFile = "";
	if (argc > 2)
	{
		printf("Usage: %s [filename]\n", argv[0]);
		return 0;
	}
	else if (argc == 2)
	{
		outputFile = argv[1];
	}
	srand(time(NULL));

	printf("Checkers, author: Jan Machalski\n");

	uint32_t whitePieces = PLAYER_PIECES_INIT;
	uint32_t blackPieces = OPPONENT_PIECES_INIT;

	uint32_t promotedPieces = 0;
	uint32_t whitePiecesOpposite = OPPONENT_PIECES_INIT;
	uint32_t blackPiecesOpposite = PLAYER_PIECES_INIT;
	uint32_t promotedPiecesOpposite = 0;
	CUDA_Vector<std::pair<uint32_t,std::string>> availableMoves;
	uint32_t move;
	std::pair<std::string,bool> move_pair;
	std::vector<std::string> gameNotation;
	int moveCounter = 1;
	int nonAdvancingMoveCounter = 0;
	std::unique_ptr<Player> whitePlayer;
	std::unique_ptr<Player> blackPlayer;
	try
	{
		whitePlayer = selectPlayerMode(true);
		blackPlayer = selectPlayerMode(false);

		printf("GAME START\n");
		if (!blackPlayer && whitePlayer)
			printBoard(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite, false);
		else
			printBoard(whitePieces, blackPieces, promotedPieces);


		while (true)
		{
			printf("MOVE %d:\n", moveCounter++);
			printf("WHITE TO MOVE\n");
			if (whitePlayer)
			{
				auto start = std::chrono::high_resolution_clock::now();
				move_pair = whitePlayer->MakeBestMove();
				auto end = std::chrono::high_resolution_clock::now();
				printf("Move computing time: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
				if (!move_pair.second)
				{
					printf("WHITE HAS NO MOVES LEFT\n");
					printf("BLACK WINS!\n");
					break;
				}
				gameNotation.push_back(move_pair.first);
				nonAdvancingMoveCounter = whitePlayer->root->nonAdvancingMoves;
				printf("Non advancing moves: %d\n", nonAdvancingMoveCounter);

				UpdateBoard(whitePlayer->root, blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite, blackPieces, whitePieces, promotedPieces);

				if (blackPlayer)
				{
					blackPlayer->UpdateRoot(whitePlayer->root->playerPieces, whitePlayer->root->opponentPieces, whitePlayer->root->promotedPieces);
					printBoard(whitePieces, blackPieces, promotedPieces);
				}
				else
				{
					printBoard(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite, false);
				}
				if (nonAdvancingMoveCounter >= MAX_NON_ADVANCING_MOVES)
				{
					printf("DRAW - %d MOVES WITHOUT CAPTURES OR PAWNS ADVANCING\n", MAX_NON_ADVANCING_MOVES);
					break;
				}
			}
			else
			{
				availableMoves = GeneratePossibleMovesWithNotation(whitePieces, blackPieces, promotedPieces, true);
				if (availableMoves.size() == 0)
				{
					printf("WHITE HAS NO MOVES LEFT\n");
					printf("BLACK WINS!\n");
					break;
				}

				std::pair<uint32_t,std::string> new_move = InputMove(availableMoves);
				move = new_move.first;
				if (MakeMove(whitePieces, blackPieces, promotedPieces, move))
					nonAdvancingMoveCounter = 0;
				else
					nonAdvancingMoveCounter++;
				gameNotation.push_back(new_move.second);
				blackPiecesOpposite = reverseBits(blackPieces);
				whitePiecesOpposite = reverseBits(whitePieces);
				promotedPiecesOpposite = reverseBits(promotedPieces);
				if (blackPlayer)
					printBoard(whitePieces, blackPieces, promotedPieces);
				else
					printBoard(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite, false);
				if (nonAdvancingMoveCounter >= MAX_NON_ADVANCING_MOVES)
				{
					printf("DRAW - %d MOVES WITHOUT CAPTURES OR PAWNS ADVANCING\n", MAX_NON_ADVANCING_MOVES);
					break;
				}
				if (blackPlayer)
					blackPlayer->UpdateRoot(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite);
			}
			printf("BLACK TO MOVE\n");

			if (blackPlayer)
			{
				auto start = std::chrono::high_resolution_clock::now();
				move_pair = blackPlayer->MakeBestMove();
				auto end = std::chrono::high_resolution_clock::now();
				printf("Move computing time: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
				if (!move_pair.second)
				{
					printf("BLACK HAS NO MOVES LEFT\n");
					printf("WHITE WINS!\n");
					break;
				}
				gameNotation.push_back(move_pair.first);
				nonAdvancingMoveCounter = blackPlayer->root->nonAdvancingMoves;
				printf("Non advancing moves: %d\n", nonAdvancingMoveCounter);

				UpdateBoard(blackPlayer->root, whitePieces, blackPieces, promotedPieces, whitePiecesOpposite, blackPiecesOpposite, promotedPiecesOpposite);
				if (whitePlayer)
				{
					whitePlayer->UpdateRoot(blackPlayer->root->playerPieces, blackPlayer->root->opponentPieces, blackPlayer->root->promotedPieces);
					printBoard(whitePieces, blackPieces, promotedPieces);
				}
				else
				{
					printBoard(whitePieces, blackPieces, promotedPieces);
				}
				if (nonAdvancingMoveCounter >= MAX_NON_ADVANCING_MOVES)
				{
					printf("DRAW - %d MOVES WITHOUT CAPTURES OR PAWNS ADVANCING\n", MAX_NON_ADVANCING_MOVES);
					break;
				}
			}
			else
			{
				availableMoves = GeneratePossibleMovesWithNotation(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite,false);
				if (availableMoves.size() == 0)
				{
					printf("BLACK HAS NO MOVES LEFT\n");
					printf("WHITE WINS!\n");
					break;
				}
				std::pair<uint32_t, std::string> new_move = InputMove(availableMoves);
				move = new_move.first;
				if (MakeMove(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite, move))
					nonAdvancingMoveCounter = 0;
				else
					nonAdvancingMoveCounter++;
				gameNotation.push_back(new_move.second);
				blackPieces = reverseBits(blackPiecesOpposite);
				whitePieces = reverseBits(whitePiecesOpposite);
				promotedPieces = reverseBits(promotedPiecesOpposite);
				if (whitePlayer)
					printBoard(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite, false);
				else
					printBoard(whitePieces, blackPieces, promotedPieces);
				if (nonAdvancingMoveCounter >= MAX_NON_ADVANCING_MOVES)
				{
					printf("DRAW - %d MOVES WITHOUT CAPTURES OR PAWNS ADVANCING\n", MAX_NON_ADVANCING_MOVES);
					break;
				}
				if (whitePlayer)
					whitePlayer->UpdateRoot(whitePieces, blackPieces, promotedPieces);
			}

		}
	}
	catch (const std::exception& e)
	{
		std::cout << "Error: " << e.what() << std::endl;
		if (whitePlayer)
			whitePlayer.reset();
		if(blackPlayer)
			blackPlayer.reset();
	}

	if (argc==2)
		saveGameToFile(gameNotation, outputFile);

	return 0;
}