import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random
from collections import deque
from Blackjack import BlackJackGame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import seaborn as sns
from collections import defaultdict

device = torch.device("cuda")

learning_rate = 0.001
input_size = 4  # Player_value, Dealer_showing_value, Player_usable_ace, dealer_usable_ace
hidden_size = 4
output_size = 2  # 0:Hit 1:Stay
episodes = 5000
eval_episodes = 10000
batch_size = 64

game = BlackJackGame()
scaler = GradScaler()  # Initialize Gradscaler


class BlackJackDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BlackJackDQN, self).__init__()
        self.activations = None
        self.fc1 = nn.Linear(input_size, hidden_size)  # First Layer
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)  # Add a Dropout layer

        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second Layer
        self.bn2 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)  # Add a second Dropout layer

        self.fc3 = nn.Linear(hidden_size, hidden_size)  # Third Layer
        self.bn3 = nn.BatchNorm1d(hidden_size)  # add batch normalization
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)  # Add a third Dropout layer

        self.fc4 = nn.Linear(hidden_size, output_size)  # Forth layer

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        if x.size(0) > 256:  # Apply batch normalization if the batch size is equal (batch_size)
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        if x.size(0) > 256:  # Apply batch normalization if the batch size is equal (batch_size)
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        if x.size(0) > 256:  # Apply batch normalization if the batch size is equal (batch_size)
            x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        self.activations = x.clone()  # Capture activations before final layer
        x = self.fc4(x)
        return x


class DQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.action_size = output_size  # Hit or Stay
        self.model = BlackJackDQN(state_size, hidden_size, self.action_size).to(device)
        self.target_model = BlackJackDQN(state_size, hidden_size, self.action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.crtierion = nn.MSELoss()
        self.activations = []

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.update_target_model()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        act_values = self.model(state)
        self.activations.append(self.model.activations.cpu().detach().numpy())  # Store activations
        return torch.argmax(act_values[0]).item()

    def replay(self):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.activations = []  # Reset activations at the start of the replay
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                next_action = self.model(next_state).argmax(dim=1).unsqueeze(1)
                t = self.target_model(next_state).gather(1, next_action).squeeze(1).item()
                target = self.model(state).detach().clone()
                if done:
                    target[0, action] = reward
                else:
                    target[0, action] = reward + self.gamma * t
            self.optimizer.zero_grad()
            with autocast():
                output = self.model(state)
                loss = self.crtierion(output, target.detach())
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            scaler.step(self.optimizer)
            scaler.update()
        self.scheduler.step()
        print(f"Epsilon: {self.epsilon:.2f}, Loss: {loss.item():.4f}")

    def evaluate(self):
        self.load_model('dqn_blackjack2.pth')
        self.model.eval()  # Set the model to evaluation mode
        total_reward = 0
        total_wins = 0
        total_losses = 0
        total_draws = 0

        for e in range(eval_episodes):
            game.reset()
            game.start_game()
            state = trainer.get_state(game.current_hand_index, first=True)
            hand_indices = list(game.player_hand.keys())
            i = 0
            while i < len(hand_indices):
                hand_index = hand_indices[i]
                game.current_hand_index = hand_index
                if game.status[hand_index] == game.GameStatus.BLACKJACK:
                    i += 1
                    continue
                else:
                    game.status[hand_index] = game.GameStatus.CONTINUE  # Reset the status for each hand
                while game.status[hand_index] == game.GameStatus.CONTINUE:
                    action = self.act(state)
                    if action == 0:
                        game_action = "stay"
                    elif action == 1:
                        game_action = "hit"
                    if game_action == "stay":
                        break
                    game.player_action(game_action, hand_index)

                    next_state = trainer.get_state(game.current_hand_index, first=False)
                    state = next_state
                i += 1
                # Dealer's turn
                game.dealer_action(output=False)
                player_value = game.player_hand_value[game.current_hand_index]
                result = game.game_result()
                reward = trainer.get_reward(result, player_value, game_action)
                total_reward += reward
                if result in ["Player Value Greater", "Dealer Bust", "Blackjack"]:
                    total_wins += 1
                elif result in ["Player Bust", "Dealer Value Greater"]:
                    total_losses += 1
                else:
                    total_draws += 1
                break

        avg_reward = total_reward / eval_episodes
        win_rate = (total_wins / eval_episodes) * 100
        loss_rate = (total_losses / eval_episodes) * 100
        draw_rate = (total_draws / eval_episodes) * 100

        print(f"Evaluation over {eval_episodes} episodes:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Loss Rate: {loss_rate:.2f}%")
        print(f"Draw Rate: {draw_rate:.2f}%")


# noinspection PyUnboundLocalVariable
class BlackJackTrainer:
    def __init__(self):
        self.agent = DQNAgent(state_size=input_size)  # player_value, dealer_showing_value, player_aces, dealer_aces (4)
        self.reward = 0
        self.total_losses = 0
        self.total_wins = 0
        self.total_draws = 0
        self.total_rewards = 0

    @staticmethod
    def card_to_index(card):
        # Convert a card to an index based on its rank and suit
        suits = {'hearts': 0, 'diamonds': 1, 'clubs': 2, 'spades': 3}
        ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
                 '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        return ranks[card['number']] + 13 * suits[card['suit']]

    def get_reward(self, result, player_value, game_action):
        # Reward based on game outcome
        reward = 0
        if result == "Player Bust":
            reward += -2
            self.total_losses += 1
        elif result == "Dealer Value Greater":
            reward += -1.25
            self.total_losses += 1
        elif result == "Dealer Bust":
            reward += 1
            self.total_wins += 1
        elif result == "Player Value Greater":
            reward += 1.25
            self.total_wins += 1
        elif result == "Draw":
            reward += 0.5
            self.total_draws += 1
        elif result == "Blackjack":
            self.total_wins += 1
            reward += 1

        # Reward based on player hand value
        if 21 >= player_value >= 20:
            reward += 0.75
        elif 19 >= player_value >= 17:
            reward += 0.5
        elif 16 >= player_value >= 14:
            reward += 0.25
        elif 10 >= player_value:
            reward -= 0.75
        elif player_value == 11:
            reward -= 1
        if 11 >= player_value and game.player_aces[0] != 0:
            reward -= 2

        if game_action == "stay":
            if 12 > player_value:
                reward -= 0.5
            if 11 == player_value:
                reward -= 1
            if 10 >= player_value:
                reward -= 1
            elif player_value == 21:
                reward += 2
            elif 20 >= player_value >= 19:
                reward += 1
            elif 18 >= player_value >= 17:
                reward += 0.5

        # Handle Hit rewards
        if game_action == "hit":
            if player_value > 15:
                reward -= 1
            elif player_value > 18:
                reward -= 2
            if player_value <= 11:
                reward += 1

        return reward

    @staticmethod
    def get_state(hand_index, first=False):
        dealer_aces = game.dealer_aces
        player_value = game.player_hand_value[hand_index]
        player_usable_ace = game.usable_ace
        if first:
            dealer_showing_value = game.card_values[game.dealer_hand[0]['number']]
        if not first:
            dealer_showing_value = game.dealer_value
        state = np.array([player_value, dealer_showing_value, player_usable_ace, dealer_aces]).astype(
            np.float32)
        return state

    def evaluate_agent(self):
        self.agent.evaluate()

    def visualize(self):
        # Replace later with a visualizer
        winrate = round((self.total_wins / episodes) * 100)

        blackjacks = episodes - (self.total_wins + self.total_losses + self.total_draws)
        print(f"Total winrate: {winrate}%, wins: {self.total_wins}, Losses: {self.total_losses} "
              f"Draws: {self.total_draws}, Blackjack's: {blackjacks}, Total Games: {episodes}")

    def train(self, e):
        # Load pre-trained model if available
        try:
            self.agent.load_model('dqn_blackjack2.pth')
            print("Loaded pre-trained model.")
        except FileNotFoundError:
            print("No pre-trained model found, starting training from scratch.")

        game.reset()
        game.start_game()
        state = self.get_state(game.current_hand_index, first=True)
        hand_indices = list(game.player_hand.keys())
        i = 0
        action = None
        next_state = state

        while i < len(hand_indices):
            hand_index = hand_indices[i]
            game.current_hand_index = hand_index
            if game.status[hand_index] == game.GameStatus.BLACKJACK:
                i += 1
                continue
            else:
                game.status[hand_index] = game.GameStatus.CONTINUE  # Reset the status for each hand
            while game.status[hand_index] == game.GameStatus.CONTINUE:
                action = self.agent.act(state)
                game_action = "stay"  # Set a default action

                if action == 0:
                    game_action = "stay"
                    print("Action: Stay")
                elif action == 1:
                    game_action = "hit"
                    print("Action: Hit")
                if game_action == "stay":
                    break
                game.player_action(game_action, hand_index)

                # Check if the player's current hand has resulted in a bust
                if game.status[hand_index] == game.GameStatus.BUST:
                    print(f"{hand_index + 1}: Player bust")
                next_state = self.get_state(game.current_hand_index, first=False)
            i += 1

            # Dealer's turn
            print(f"Dealer reveals second card: {game.format_cards(game.dealer_hand)} {game.dealer_value}")
            game.dealer_action(output=True)
            player_value = game.player_hand_value[game.current_hand_index]
            result = game.game_result()
            reward = self.get_reward(result, player_value, game_action)
            done = True
            self.agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(self.agent.memory) > batch_size:
                self.agent.replay()
                if self.agent.epsilon > self.agent.epsilon_min:
                    self.agent.epsilon *= self.agent.epsilon_decay
            if done:
                print(f"Episode {e + 1}/{episodes} - Reward: {reward}, Epsilon: {self.agent.epsilon:.2f}")
                self.total_rewards += reward
                break


agent = DQNAgent(input_size)
trainer = BlackJackTrainer()


def plot_activations(activations, epoch):
    plt.figure(figsize=(12, 6))
    sns.heatmap(np.mean(activations, axis=0), cmap='viridis')
    plt.title(f'Average Activations at Epoch {epoch}')
    plt.xlabel('Samples')
    plt.ylabel('Neurons')
    plt.show()


def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    state_value = defaultdict(float)
    policy = defaultdict(int)

    for player_value in range(12, 22):
        for dealer_value in range(1, 11):
            for player_usable_ace in [0, 1]:
                for dealer_usable_ace in [0, 1]:
                    player_value_norm = player_value
                    dealer_value_norm = dealer_value

                    state_tensor = torch.tensor(
                        [[player_value_norm, dealer_value_norm, player_usable_ace, dealer_usable_ace]]
                    ).to(device)

                    with torch.no_grad():
                        q_values = agent.model(state_tensor).cpu().numpy().flatten()

                    obs = (player_value, dealer_value, player_usable_ace)
                    state_value[obs] = float(np.max(q_values))
                    policy[obs] = int(np.argmax(q_values))

    player_count, dealer_count = np.meshgrid(
        np.arange(12, 22),
        np.arange(1, 11),
    )

    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], 0 if usable_ace else 1)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    value_grid = player_count, dealer_count, value

    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], 0 if usable_ace else 1)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    return value_grid, policy_grid


# noinspection PyTypeChecker
def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


def main():
    for e in range(1, episodes):
        trainer.train(e)
        # Save the model
        trainer.agent.save_model('dqn_blackjack2.pth')
        print(f"Model saved at episode {e}")
        if e % 10 == 0:
            trainer.agent.update_target_model()
            agent.update_epsilon()
        if e % 1000 == 0:
            if len(trainer.agent.activations) > 0:
                # Save and plot activations only if there are activations
                plot_activations(trainer.agent.activations, e)
    trainer.visualize()
    # Evaluate the agent
    trainer.evaluate_agent()


"""Visdom"""
if __name__ == '__main__':
    main()

    value_grid, policy_grid = create_grids(trainer.agent, usable_ace=True)
    fig1 = create_plots(value_grid, policy_grid, title="With Usable Ace")
    plt.show()

    value_grid, policy_grid = create_grids(trainer.agent, usable_ace=False)
    fig2 = create_plots(value_grid, policy_grid, title="Without Usable Ace")
    plt.show()
