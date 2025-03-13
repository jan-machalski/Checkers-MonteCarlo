# Monte Carlo Checkers Bot

The program contains a bot to play [8x8 Polish checkers](https://pl.wikipedia.org/wiki/Warcaby).
It uses [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) (MCTS) to find the best moves.

---

## Parallelisation

To maximize the number of simulations per move, the bot utilizes GPU threads to conduct multiple simulations simultaneously.
On an **RTX 2060**, this allows it to reach up to **3.2 million simulations per second** from the starting position—**20 times more than the CPU**.

---

## Gameplay Example

![Gameplay Example](https://github.com/user-attachments/assets/10cadd83-109e-43dd-b7c8-75e94920d3d2)

