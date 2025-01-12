#include "PlayerGPU.cuh"

using namespace std;

uint32_t InputMove(uint32_t playerPieces, uint32_t opponentPieces, uint32_t promotedPieces, bool reverseBoard = false)
{


	uint32_t obstacles = playerPieces;
	uint32_t result = 0;
	std::string move;
	std::cout << "Input your move" << std::endl;
	std::cin >> move;
	if (move.length() % 2)
		return 0;
	if (reverseBoard)
	{
		for (int i = 0; i < move.length(); i++)
		{
			if (move[i] <= '8' && move[i] >= '1')
				move[i] = '1' + '8' - move[i];
			else if (move[i] <= 'h' && move[i] >= 'a')
				move[i] = 'a' + 'h' - move[i];
			else
				return 0;
		}
	}
	int startPos = 4 * ('8' - move[1]) + (move[0] - 'a') / 2;
	int endPos = 4 * ('8' - move[move.length() - 1]) + (move[move.length() - 2] - 'a') / 2;
	SET_BIT(result, startPos, 1);
	SET_BIT(result, endPos, 1);
	for (int i = 0; i < move.length() / 2 - 1; i++)
	{
		std::string s1 = move.substr(i * 2, 2);
		std::string s2 = move.substr(i * 2 + 2, 2);
		if ((s1[0] - 'a' + s1[1] - '1') % 2 == 1 || (s2[0] - 'a' + s2[1] - '1') % 2 == 1)
			return 0;
		if (abs(s1[0] - s2[0]) != abs(s1[1] - s2[1]))
			return 0;
		int r0 = (s2[0] - s1[0]) / abs(s2[0] - s1[0]);
		int r1 = (s2[1] - s1[1]) / abs(s2[1] - s1[1]);
		int pos0 = s1[0] + r0;
		int pos1 = s1[1] + r1;
		while (pos0 != s2[0])
		{
			int posNr = 4 * ('8' - pos1) + (pos0 - 'a') / 2;
			if (BIT(obstacles, posNr))
				return 0;
			if (BIT(opponentPieces, posNr))
			{
				SET_BIT(obstacles, posNr, 1);
				SET_BIT(result, posNr, 1);
			}

			pos0 += r0;
			pos1 += r1;
		}
	}
	return result;

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

int main()
{
	srand(time(NULL));

	uint32_t pl = 0;
	uint32_t opp = 0;
	uint32_t prom = 0;
	SET_BIT(pl, 0, 1);
	SET_BIT(pl, 3, 1);
	//SET_BIT(prom, 0, 1);
	//SET_BIT(prom, 3, 1);

	SET_BIT(opp, 7, 1);
	SET_BIT(opp, 15, 1);
	SET_BIT(opp, 14, 1);
	SET_BIT(opp, 22, 1);
	SET_BIT(opp, 23, 1);
	SET_BIT(opp, 21, 1);

	CUDA_Vector<pair<uint32_t,string>> moves = GeneratePossibleMovesWithNotation(pl, opp, prom,true);
	printBoard(pl, opp, prom, true);
	for (auto m : moves)
	{
		cout << m.second << std::endl;
		/*for (int i = 0; i < 32; i++)
			if (BIT(m.first, i))
				cout << i << " ";
		cout << std::endl;*/
	}
	/*treeNode* root = new treeNode();
	GenerateChildren(root);
	for (auto c : root->children)
	{
		printBoard(c->playerPieces, c->opponentPieces, c->promotedPieces, false);
	}*/
	/*uint32_t tmpPlayer, tmpPromoted, tmpOpponent;
	vector<uint32_t> availableMoves;
	uint32_t move;
	Player player(true, 10000, 500);
	while (true)
	{
		if (!player.MakeBestMove())
		{
			;
			cout << "YOU WIN";
			break;
		}
		if (player.root->nonAdvancingMoves >= 10)
		{
			cout << "DRAW - 10 non advancing moves";
			break;
		}
		printBoard(player.root->playerPieces, player.root->opponentPieces, player.root->promotedPieces, false);
		availableMoves = GeneratePossibleMoves(player.root->playerPieces, player.root->opponentPieces, player.root->promotedPieces);
		//for (auto m : availableMoves)
		//	PrintMoveInfo(m);
		if (availableMoves.size() == 0)
		{
			cout << "YOU LOSE";
			break;
		}
		do
		{
			move = InputMove(player.root->playerPieces, player.root->opponentPieces, player.root->promotedPieces,true);
			//PrintMoveInfo(move);

		} while (find(availableMoves.begin(), availableMoves.end(), move) == availableMoves.end());
		tmpPlayer = player.root->playerPieces;
		tmpOpponent = player.root->opponentPieces;
		tmpPromoted = player.root->promotedPieces;
		MakeMove(tmpPlayer, tmpOpponent, tmpPromoted, move);
		printBoard(tmpPlayer, tmpOpponent, tmpPromoted, false);
		tmpPlayer = REVERSE_BITS(tmpPlayer);
		tmpOpponent = REVERSE_BITS(tmpOpponent);
		tmpPromoted = REVERSE_BITS(tmpPromoted);
		player.UpdateRoot(tmpOpponent, tmpPlayer, tmpPromoted);
		if (player.root->nonAdvancingMoves >= 10)
		{
			cout << "DRAW - 10 non advancing moves";
			break;
		}

	}*/

	/*uint32_t whitePieces = PLAYER_PIECES_INIT;
	uint32_t blackPieces = OPPONENT_PIECES_INIT;
	uint32_t promotedPieces = 0;
	uint32_t whitePiecesOpposite = OPPONENT_PIECES_INIT;
	uint32_t blackPiecesOpposite = PLAYER_PIECES_INIT;
	uint32_t promotedPiecesOpposite = 0;
	CUDA_Vector<uint32_t> availableMoves;
	uint32_t move;
	int moveCounter = 1;
	int nonAdvancingMoveCounter = 0;
	auto whitePlayer = selectPlayerMode(true);
	auto blackPlayer = selectPlayerMode(false);

	std::cout << "GAME START" << std::endl;
	if (!blackPlayer && whitePlayer)
		printBoard(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite, false);
	else
		printBoard(whitePieces, blackPieces, promotedPieces);

	while (true)
	{
		std::cout << "MOVE " << moveCounter++ << ":" << std::endl;
		std::cout << "WHITE TO MOVE" << std::endl;
		if (whitePlayer)
		{
			if (!(whitePlayer->MakeBestMove()))
			{
				std::cout << "WHITE HAS NO MOVES LEFT" << std::endl;
				std::cout << "BLACK WINS!" << std::endl;
				break;
			}
			nonAdvancingMoveCounter = whitePlayer->root->nonAdvancingMoves;
			std::cout << "Non advancing moves: " << nonAdvancingMoveCounter << std::endl;

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
			if (nonAdvancingMoveCounter >= 10)
			{
				std::cout << "DRAW - 10 MOVES WITHOUT CAPTURES OR PAWNS ADVANCING" << std::endl;
				break;
			}
		}
		else
		{
			availableMoves = GeneratePossibleMoves(whitePieces, blackPieces, promotedPieces);
			if (availableMoves.size() == 0)
			{
				std::cout << "WHITE HAS NO MOVES LEFT" << std::endl;
				std::cout << "BLACK WINS!" << std::endl;
				break;
			}
			do
			{
				move = InputMove(whitePieces, blackPieces, promotedPieces);
			} while (std::find(availableMoves.begin(), availableMoves.end(), move) == availableMoves.end());
			if (MakeMove(whitePieces, blackPieces, promotedPieces, move))
				nonAdvancingMoveCounter = 0;
			else
				nonAdvancingMoveCounter++;
			blackPiecesOpposite = reverseBits(blackPieces);
			whitePiecesOpposite = reverseBits(whitePieces);
			promotedPiecesOpposite = reverseBits(promotedPieces);
			if (blackPlayer)
				printBoard(whitePieces, blackPieces, promotedPieces);
			else
				printBoard(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite, false);
			if (nonAdvancingMoveCounter >= 10)
			{
				std::cout << "DRAW - 10 MOVES WITHOUT CAPTURES OR PAWNS ADVANCING" << std::endl;
				break;
			}
			if (blackPlayer)
				blackPlayer->UpdateRoot(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite);
		}

		std::cout << "BLACK TO MOVE" << std::endl;

		if (blackPlayer)
		{
			if (!(blackPlayer->MakeBestMove()))
			{
				std::cout << "BLACK HAS NO MOVES LEFT" << std::endl;
				std::cout << "WHITE WINS!" << std::endl;
				break;
			}
			nonAdvancingMoveCounter = blackPlayer->root->nonAdvancingMoves;
			std::cout << "Non advancing moves: " << nonAdvancingMoveCounter << std::endl;

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
			if (nonAdvancingMoveCounter >= 10)
			{
				std::cout << "DRAW - 10 MOVES WITHOUT CAPTURES OR PAWNS ADVANCING" << std::endl;
				break;
			}
		}
		else
		{
			availableMoves = GeneratePossibleMoves(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite);
			if (availableMoves.size() == 0)
			{
				std::cout << "BLACK HAS NO MOVES LEFT" << std::endl;
				std::cout << "WHITE WINS!" << std::endl;
				break;
			}
			do
			{
				move = InputMove(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite, true);
			} while (std::find(availableMoves.begin(), availableMoves.end(), move) == availableMoves.end());
			if (MakeMove(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite, move))
				nonAdvancingMoveCounter = 0;
			else
				nonAdvancingMoveCounter++;
			blackPieces = reverseBits(blackPiecesOpposite);
			whitePieces = reverseBits(whitePiecesOpposite);
			promotedPieces = reverseBits(promotedPiecesOpposite);
			if (whitePlayer)
				printBoard(blackPiecesOpposite, whitePiecesOpposite, promotedPiecesOpposite, false);
			else
				printBoard(whitePieces, blackPieces, promotedPieces);
			if (nonAdvancingMoveCounter >= 10)
			{
				std::cout << "DRAW - 10 MOVES WITHOUT CAPTURES OR PAWNS ADVANCING" << std::endl;
				break;
			}
			if (whitePlayer)
				whitePlayer->UpdateRoot(whitePieces, blackPieces, promotedPieces);
		}

	}*/

	return 0;
}