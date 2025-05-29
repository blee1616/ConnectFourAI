#RL Bot
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from main import (
    create_board, drop_piece, is_valid_location, get_next_open_row,
    winning_move, get_valid_locations, is_terminal_node, print_board,
    ROW_COUNT, COLUMN_COUNT, PLAYER_PIECE, AI_PIECE, EMPTY
)

# Neural Network for Q-learning
class ConnectFourQNetwork(nn.Module):
    def __init__(self):
        super(ConnectFourQNetwork, self).__init__()
        # Input: flattened board state (6x7=42) + 1 for current player
        self.fc1 = nn.Linear(ROW_COUNT * COLUMN_COUNT + 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, COLUMN_COUNT)  # Output: Q-value for each column
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer()
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = ConnectFourQNetwork()
        self.target_model = ConnectFourQNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def encode_state(self, board, player_turn):
        # Flatten the board and add player turn information
        state = np.append(board.flatten(), player_turn)
        return torch.FloatTensor(state)
    
    def act(self, state, valid_locations):
        if np.random.rand() <= self.epsilon:
            # Exploration: choose a random valid column
            return random.choice(valid_locations)
        
        # Exploitation: choose the best action based on Q-values
        with torch.no_grad():
            q_values = self.model(state)
        
        # Filter to only valid actions
        valid_q_values = {col: q_values[col].item() for col in valid_locations}
        return max(valid_q_values, key=valid_q_values.get)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.target_model(next_state))
            
            # Get current Q values
            current_q = self.model(state)
            # Update the Q value for the action
            target_q = current_q.clone()
            target_q[action] = target
            
            # Compute loss and update weights
            loss = nn.MSELoss()(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Function to calculate reward
def calculate_reward(board, action, player):
    # Base reward
    reward = 0
    
    # Check if action is valid
    if not is_valid_location(board, action):
        return -10  # Negative reward for invalid action
    
    # Copy the board and make the move
    row = get_next_open_row(board, action)
    next_board = board.copy()
    drop_piece(next_board, row, action, player)
    
    # Check if the move wins the game
    if winning_move(next_board, player):
        reward += 100  # High reward for winning
    
    # Check if the move blocks opponent's winning move
    opponent = AI_PIECE if player == PLAYER_PIECE else PLAYER_PIECE
    for col in get_valid_locations(board):
        temp_row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, temp_row, col, opponent)
        if winning_move(temp_board, opponent):
            if action == col:  # If our move blocks this winning move
                reward += 50   # Reward for blocking
    
    # Center control is generally good
    if action == COLUMN_COUNT // 2:
        reward += 3
    
    return reward

# Training function
def train_agent(episodes=1000, batch_size=32):
    agent = DQNAgent(ROW_COUNT * COLUMN_COUNT + 1, COLUMN_COUNT)
    scores = []
    
    for episode in range(episodes):
        board = create_board()
        game_over = False
        turn = random.randint(0, 1)  # Randomly decide who goes first
        
        # For plotting progress
        score = 0
        
        while not game_over:
            # Current player's turn
            player = PLAYER_PIECE if turn == 0 else AI_PIECE
            valid_locations = get_valid_locations(board)
            
            if not valid_locations:  # Draw
                game_over = True
                continue
            
            # Get current state
            state = agent.encode_state(board, player)
            
            # Choose action
            action = agent.act(state, valid_locations)
            
            # Make sure action is valid before executing
            if action not in valid_locations:
                # If somehow an invalid action was selected, pick a random valid one
                if valid_locations:
                    action = random.choice(valid_locations)
                else:
                    game_over = True
                    continue
            
            # Execute action
            row = get_next_open_row(board, action)
            drop_piece(board, row, action, player)
            
            # Calculate reward
            reward = calculate_reward(board, action, player)
            score += reward
            
            # Check if game is over
            if winning_move(board, player):
                game_over = True
                reward += 100  # Additional reward for winning
            elif len(get_valid_locations(board)) == 0:
                game_over = True  # Draw
            
            # Next state
            next_state = agent.encode_state(board, AI_PIECE if player == PLAYER_PIECE else PLAYER_PIECE)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, game_over)
            
            # Train model
            agent.replay(batch_size)
            
            # Switch turns
            turn = (turn + 1) % 2

def play_and_train(episodes=50, batch_size=32):
    """Train the RL agent by playing against it"""
    agent = DQNAgent(ROW_COUNT * COLUMN_COUNT + 1, COLUMN_COUNT)
    win_stats = {'Human': 0, 'AI': 0, 'Draw': 0}
    
    for episode in range(episodes):
        board = create_board()
        game_over = False
        turn = 0  # Human always goes first for simplicity
        
        print(f"\nEpisode {episode+1}/{episodes}")
        print("Current stats: Human wins: {}, AI wins: {}, Draws: {}".format(
            win_stats['Human'], win_stats['AI'], win_stats['Draw']))
        print("AI exploration rate (epsilon): {:.2f}".format(agent.epsilon))
        print_board(board)
        
        while not game_over:
            if turn == 0:  # Human's turn
                valid_cols = get_valid_locations(board)
                col = None
                while col is None:
                    try:
                        col_input = input("Your turn! Choose column (0-6): ")
                        col = int(col_input)
                        if col not in valid_cols:
                            print("Invalid column! Try again.")
                            col = None
                    except ValueError:
                        print("Please enter a number between 0 and 6.")
                
                # Execute human move
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)
                print_board(board)
                
                # Check if human won
                if winning_move(board, PLAYER_PIECE):
                    print("You win! Agent will learn from this.")
                    win_stats['Human'] += 1
                    game_over = True
                
                # Get state after human move for the agent to learn
                next_state = agent.encode_state(board, AI_PIECE)
                
            else:  # RL Agent's turn
                valid_locations = get_valid_locations(board)
                if not valid_locations:
                    win_stats['Draw'] += 1
                    print("It's a draw!")
                    game_over = True
                    continue
                
                # Get current state
                state = agent.encode_state(board, AI_PIECE)
                
                # Agent chooses action
                action = agent.act(state, valid_locations)
                print(f"AI is thinking... and chooses column {action}")
                
                # Execute action
                row = get_next_open_row(board, action)
                drop_piece(board, row, action, AI_PIECE)
                print_board(board)
                
                # Calculate reward
                reward = 0
                
                # Check if agent won
                if winning_move(board, AI_PIECE):
                    print("AI wins this round!")
                    win_stats['AI'] += 1
                    reward = 100
                    game_over = True
                elif len(get_valid_locations(board)) == 0:  # Draw
                    print("It's a draw!")
                    win_stats['Draw'] += 1
                    reward = 0
                    game_over = True
                else:
                    # Reward for blocking potential human wins
                    # Check if agent blocked a winning move
                    temp_board = board.copy()
                    for test_col in valid_locations:
                        if test_col == action:  # Skip the move we just made
                            continue
                        test_row = get_next_open_row(temp_board, test_col)
                        temp_board[test_row][test_col] = PLAYER_PIECE
                        if winning_move(temp_board, PLAYER_PIECE):
                            reward += 50  # Reward for blocking
                        # Reset the temp board
                        temp_board[test_row][test_col] = EMPTY
                    
                    # Center control is generally good
                    if action == COLUMN_COUNT // 2:
                        reward += 3
                
                # Store experience
                next_state = agent.encode_state(board, PLAYER_PIECE) if not game_over else None
                agent.remember(state, action, reward, next_state, game_over)
                
                # Train the model
                agent.replay(batch_size)
            
            # Switch turns
            turn = (turn + 1) % 2
        
        # After each game, update target network
        if episode % 5 == 0:
            agent.update_target_model()
    
    # Save the trained model
    torch.save(agent.model.state_dict(), "human_trained_rl_model.pth")
    print("\nTraining complete! Model saved to 'human_trained_rl_model.pth'")
    
    # Show final stats
    print(f"Final stats: Human wins: {win_stats['Human']}, AI wins: {win_stats['AI']}, Draws: {win_stats['Draw']}")
    
    return agent

# Function to test the trained agent against minimax
def test_trained_agent_vs_minimax(agent, depth=5, num_games=5):
    from main import minimax
    
    win_stats = {'RL': 0, 'Minimax': 0, 'Draw': 0}
    
    for game in range(num_games):
        board = create_board()
        game_over = False
        turn = random.randint(0, 1)  # Randomly decide who goes first
        
        print(f"\nGame {game+1}/{num_games}")
        print_board(board)
        
        while not game_over:
            if turn == 0:  # RL Agent's turn
                valid_locations = get_valid_locations(board)
                state = agent.encode_state(board, PLAYER_PIECE)
                col = agent.act(state, valid_locations)
                
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, PLAYER_PIECE)
                    
                    print(f"RL Agent dropped piece in column {col}")
                    print_board(board)
                    
                    if winning_move(board, PLAYER_PIECE):
                        print("RL Agent wins!")
                        win_stats['RL'] += 1
                        game_over = True
            
            else:  # Minimax AI's turn
                col, _ = minimax(board, depth, -float('inf'), float('inf'), True)
                
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, AI_PIECE)
                    
                    print(f"Minimax AI dropped piece in column {col}")
                    print_board(board)
                    
                    if winning_move(board, AI_PIECE):
                        print("Minimax AI wins!")
                        win_stats['Minimax'] += 1
                        game_over = True
            
            # Check for draw
            if len(get_valid_locations(board)) == 0:
                print("Game is a draw!")
                win_stats['Draw'] += 1
                game_over = True
            
            # Switch turns
            turn = (turn + 1) % 2
    
    print("\nTest results:")
    print(f"RL Agent wins: {win_stats['RL']}")
    print(f"Minimax AI wins: {win_stats['Minimax']}")
    print(f"Draws: {win_stats['Draw']}")

# Function to load a trained model
def load_trained_model(model_path="human_trained_rl_model.pth"):
    agent = DQNAgent(ROW_COUNT * COLUMN_COUNT + 1, COLUMN_COUNT)
    agent.model.load_state_dict(torch.load(model_path))
    agent.target_model.load_state_dict(agent.model.state_dict())
    agent.epsilon = 0.05  # Low epsilon for exploitation
    return agent

# Main execution
if __name__ == "__main__":
    print("What would you like to do?")
    print("1. Train the RL agent by playing against it")
    print("2. Test a trained agent against minimax")
    print("3. Play against a trained agent")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        episodes = int(input("How many games would you like to play? "))
        agent = play_and_train(episodes=episodes)
        
        # Ask if user wants to test against minimax
        test_choice = input("Would you like to test your trained agent against minimax? (y/n): ")
        if test_choice.lower() == 'y':
            num_games = int(input("How many test games? "))
            test_trained_agent_vs_minimax(agent, num_games=num_games)
    
    elif choice == '2':
        try:
            agent = load_trained_model()
            num_games = int(input("How many test games? "))
            test_trained_agent_vs_minimax(agent, num_games=num_games)
        except FileNotFoundError:
            print("No trained model found. Please train the agent first.")
    
    elif choice == '3':
        try:
            agent = load_trained_model()
            board = create_board()
            game_over = False
            turn = 0  # Human goes first
            
            print("Playing against trained RL agent")
            print_board(board)
            
            while not game_over:
                if turn == 0:  # Human's turn
                    valid_cols = get_valid_locations(board)
                    col = None
                    while col is None:
                        try:
                            col_input = input("Your turn! Choose column (0-6): ")
                            col = int(col_input)
                            if col not in valid_cols:
                                print("Invalid column! Try again.")
                                col = None
                        except ValueError:
                            print("Please enter a number between 0 and 6.")
                    
                    # Execute human move
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, PLAYER_PIECE)
                    print_board(board)
                    
                    # Check if human won
                    if winning_move(board, PLAYER_PIECE):
                        print("You win!")
                        game_over = True
                
                else:  # RL Agent's turn
                    valid_locations = get_valid_locations(board)
                    if not valid_locations:
                        print("It's a draw!")
                        game_over = True
                        continue
                    
                    # Get current state
                    state = agent.encode_state(board, AI_PIECE)
                    
                    # Agent chooses action
                    col = agent.act(state, valid_locations)
                    print(f"AI chooses column {col}")
                    
                    # Execute action
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, AI_PIECE)
                    print_board(board)
                    
                    # Check if agent won
                    if winning_move(board, AI_PIECE):
                        print("AI wins!")
                        game_over = True
                
                # Check for draw
                if len(get_valid_locations(board)) == 0:
                    print("Game is a draw!")
                    game_over = True
                
                # Switch turns
                turn = (turn + 1) % 2
                
        except FileNotFoundError:
            print("No trained model found. Please train the agent first.")
    
    else:
        print("Invalid choice")