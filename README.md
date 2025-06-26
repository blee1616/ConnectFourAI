# Connect Four AI

This project presents a Connect Four game featuring two distinct AI opponents: a classic Minimax AI with alpha-beta pruning and a modern Reinforcement Learning AI powered by a Deep Q-Network (DQN).

Users can play against either AI, interactively train the Reinforcement Learning agent, and benchmark the performance of these two powerful approaches against each other.

## Features

* **Complete Connect Four Game:** A fully playable Connect Four game with a graphical user interface (GUI).
* **Minimax AI:** A highly optimized AI that uses alpha-beta pruning to determine the optimal move.
* **Reinforcement Learning AI:** A sophisticated AI that learns and improves its strategy through gameplay.
* **Interactive Training:** Train the RL agent simply by playing against it.
* **Performance Comparison:** A mode to pit the Minimax AI against the trained RL AI to compare their effectiveness.
* **Model Persistence:** Save and load the RL agent's state to continue training across multiple sessions.

## Requirements

* Python 3.x
* NumPy
* Pygame
* PyTorch
* Matplotlib

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/ConnectFourAI.git](https://github.com/yourusername/ConnectFourAI.git)
    cd ConnectFourAI
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install numpy pygame torch matplotlib
    ```

## Usage

To start the application, run the `RL.py` file from your terminal:

```bash
python RL.py
```

Upon launching, you will be prompted to choose one of three modes:

1.  **Train the RL agent:** Play against the AI to teach it winning strategies.
2.  **Test a trained agent against Minimax:** Evaluate your trained RL agent's performance against the classic Minimax AI.
3.  **Play against a trained agent:** Challenge your previously trained RL agent to a game.

## How It Works

### Minimax AI

The Minimax algorithm is a staple of traditional game theory and operates by:

* Recursively exploring all possible future moves up to a predefined search depth.
* Using a scoring function to evaluate the strategic value of board positions.
* Selecting the move that maximizes its own score while assuming the opponent will always act to minimize it.
* Employing **alpha-beta pruning** to dramatically reduce the number of game states it needs to evaluate, making the search more efficient.

Our implementation evaluates board states by prioritizing center control, identifying connected pieces, and defensively blocking the opponent's potential wins.

### Reinforcement Learning AI

The RL agent utilizes a Deep Q-Network (DQN) with experience replay to learn effective strategies through trial and error. Over time, it refines its decision-making process to adapt to various playing styles.

**Key Components:**

* **Neural Network Architecture:** A 3-layer fully connected network serves as the brain of the AI.
* **State Representation:** The input to the network is a flattened representation of the game board ($6 \times 7 = 42$ nodes) combined with the current player's turn.
* **Action Selection:** An **epsilon-greedy policy** is used to balance exploration (trying new, random moves) and exploitation (choosing the move the network believes is optimal).
* **Experience Replay:** The agent stores past experiences (state, action, reward, next state) in a memory buffer and replays them during training. This technique leads to more stable and effective learning.
* **Reward Structure:** The agent is incentivized with the following rewards:
    * **+100** for a winning move.
    * **+50** for blocking an opponent's winning move.
    * **+3** for controlling the center column.
    * **-10** for attempting an invalid move.

## Training the RL Agent

When you choose to train the agent, you play against it directly. The agent's learning process is as follows:

1.  **High Initial Exploration:** Initially, the agent has a high "epsilon" value, meaning it will frequently make random moves to explore the game.
2.  **Gradual Exploitation:** As the agent plays more games (and is trained), its epsilon value decreases, causing it to more often exploit the strategies it has learned.
3.  **Learning from Outcomes:** The agent learns from every move, associating actions with positive or negative outcomes.
4.  **Model Saving:** Once the training session is complete, the trained model is saved to `human_trained_rl_model.pth` for future use.

## Project Structure

```
ConnectFourAI/
│
├── main.py                   # Core game logic and Minimax implementation
├── RL.py                     # Reinforcement Learning implementation and main entry point
├── human_trained_rl_model.pth # Saved model weights (created after training)
└── README.md                 # This file
```

## Future Improvements

* **Self-Play Training:** Implement a mode where the RL agent trains by playing against itself, accelerating the learning process.
* **Monte Carlo Tree Search (MCTS):** Introduce MCTS as a third AI opponent, a popular algorithm in modern game AI.
* **Convolutional Neural Network (CNN):** Enhance the RL agent's neural network by using convolutional layers to better recognize spatial patterns on the board.
* **Web Interface:** Create a web-based version of the game for online play.
* **Curriculum Learning:** Introduce a structured training regimen where the agent first learns simple concepts before moving to more complex strategies.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

* The Minimax implementation is inspired by classic algorithms in game theory.
* The DQN implementation draws upon the foundational work on deep reinforcement learning by DeepMind.
