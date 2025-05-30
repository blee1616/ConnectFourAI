Overview
This project implements a Connect Four game with two types of AI opponents:

A traditional Minimax AI with alpha-beta pruning
A Reinforcement Learning AI using Deep Q-Networks (DQN)
The project allows you to play against either AI, train the RL agent through interactive gameplay, and compare the performance of both AI approaches.

Features
Complete Connect Four game with GUI support
Minimax AI that uses alpha-beta pruning for optimal moves
Reinforcement Learning AI that learns from gameplay
Interactive training - teach the RL agent by playing against it
Performance comparison between Minimax and RL approaches
Model saving and loading for continued training
Installation
'''bash
Clone the repository:
git clone https://github.com/yourusername/ConnectFourAI.git
cd ConnectFourAI
'''
Install the required dependencies:
'''bash
pip install numpy pygame torch matplotlib
'''
Usage
Run the main program:
'''bash
python RL.py
'''
You'll be presented with three options:

Train the RL agent by playing against it - This allows you to teach the agent through interactive gameplay.
Test a trained agent against minimax - Compare how well your trained agent performs against the traditional minimax AI.
Play against a trained agent - Play a game against your previously trained RL agent.
How It Works
Minimax AI
The minimax algorithm works by:

Exploring all possible future game states up to a certain depth
Evaluating board positions with a scoring function
Choosing the move that maximizes the AI's score while minimizing the opponent's score
Using alpha-beta pruning to significantly reduce the search space
Key components:

Board state evaluation based on piece configurations
Prioritization of center control and connected pieces
Defensive blocking of opponent's potential wins
Reinforcement Learning AI
The RL agent uses Deep Q-Learning with experience replay to:

Learn effective strategies through trial and error
Improve its decision making over time
Adapt to different playing styles
Key components:

Neural Network Architecture: 3-layer fully connected network
State Representation: Flattened board (6×7=42) + current player
Action Selection: Epsilon-greedy policy (balances exploration and exploitation)
Reward Structure:
+100 for winning moves
+50 for blocking opponent's winning moves
+3 for controlling the center
-10 for invalid moves
Experience Replay: Stores and reuses past experiences for stable learning
Project Structure
main.py - Core game logic and minimax implementation
RL.py - Reinforcement learning implementation
human_trained_rl_model.pth - Saved model weights (created after training)
Training the RL Agent
When training the agent by playing against it:

The agent starts with a high exploration rate (trying random moves)
As training progresses, it gradually shifts toward exploitation (making moves it believes are optimal)
The agent learns from both wins and losses
After training, the model is saved for future use
Performance Comparison
The two AI approaches have different strengths:

Minimax AI: Perfect play up to its search depth, but computationally expensive
RL AI: Can develop novel strategies and adapt to opponents, but performance depends on training quality
Future Improvements
Implement a self-play training mode for the RL agent
Add Monte Carlo Tree Search (MCTS) as another AI approach
Create a web interface for online play
Improve the neural network architecture with convolutional layers
Implement curriculum learning for faster training

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The minimax implementation was inspired by traditional game theory algorithms
The DQN implementation draws from DeepMind's groundbreaking work on deep reinforcement learning