# Blackjack Deep Q-Learning (DQN) Agent

# Introduction:
- This project implements a Deep Q-Learning (DQN) agent to play the game of Blackjack.
- The DQN algorithm is a reinforcement learning approach that uses a neural network to approximate the Q-value function.
- The agent is trained to make decisions in the game (Hit or Stay) based on the current state of the game.

# Requirements:
- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- torch.cuda.amp
- collections
- Blackjack Game (custom module)

# Installation:
- Clone the repository: git clone https://github.com/GregW55/Deep-Q-Network-BlackJack.git
- cd Deep-Q-Network-BlackJack
-  Install the required Python packages: pip install torch numpy matplotlib seaborn

# Usage:
- Run the main script to start training the DQN agent:
- python DQN.py
- The script will save the trained model periodically and plot activations and evaluation metrics after the training process.

# Code Structure
- BlackJackGame: Custom module that handles the Blackjack game logic.
- BlackJackDQN: Defines the neural network architecture for the DQN agent.
- DQNAgent: Handles the DQN algorithm, including training and action selection.
- BlackJackTrainer: Manages the training process and rewards.
- main: Entry point for training and evaluating the agent.
- visualize: Contains functions for visualizing the agent's policy and state values.

# Training - The training process involves the following steps:
- Initialize the Environment and Agent:
- game = BlackJackGame()
- trainer = BlackJackTrainer()
- Train the Agent: The train method in BlackJackTrainer runs episodes of the game, where the agent interacts with the environment, collects experiences, and updates the Q-network.
        
# Save the Model:
- The model is saved periodically to avoid losing progress.

# Evaluation:
- After training, the agent is evaluated on a set of episodes to determine its performance. The evaluation metrics include average reward, win rate, loss rate, and draw rate.
- def evaluate_agent(self)
    
# Visualization -The project includes several visualization tools:
- Activation Plots: Visualize the average activations of the neural network layers during training.
- def plot_activations(activations, epoch):
    
# Value and Policy Grids:
- Visualize the agent's learned state values and policy for states with and without a usable ace.
- value_grid, policy_grid = create_grids(trainer.agent, usable_ace=True)
- fig1 = create_plots(value_grid, policy_grid, title="With Usable Ace")
- plt.show()
- value_grid, policy_grid = create_grids(trainer.agent, usable_ace=False)
- fig2 = create_plots(value_grid, policy_grid, title="Without Usable Ace")
- plt.show()

# Results - After training, the results of the agent's performance are printed and plotted. The following metrics are calculated:
- The average reward per episode during evaluation.
- Win Rate: The percentage of games won by the agent.
- Loss Rate: The percentage of games lost by the agent.
- Draw Rate:The percentage of games that ended in a draw.


# Conclusion:
- This project demonstrates the implementation of a DQN agent for playing Blackjack. It showcases the process of training a reinforcement learning agent, evaluating its performance, and visualizing its learned policies and state values.
-  The code provides a foundation for further experimentation and improvement in training strategies, reward structures, and network architectures.
-  For any questions or contributions, feel free to open an issue or submit a pull request.
