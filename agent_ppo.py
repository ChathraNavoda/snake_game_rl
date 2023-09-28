import random
import torch
import numpy as np
from collections import deque
from game_ppo import SnakeGamePPO
from model_ppo import ActorCritic, PPO
import torch.nn.functional as F
from helper_ppo import plot_ppo

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR_ACTOR = 0.001
LR_CRITIC = 0.001
GAMMA = 0.95
K_EPOCHS = 5
EPS_CLIP = 0.2

# Define the dimension of your state space (state_dim)
# and the number of actions (num_actions) based on your game
state_dim = 4
num_actions = 3

class PPOAgent:

    def __init__(self, state_dim, num_actions):
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.actor_critic = ActorCritic(state_dim, num_actions)
        self.ppo = PPO(state_dim, num_actions, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP)

    def get_state(self, game):
        # Extract relevant information from the game
        snake_head = game.snake[0]
        food_position = game.food
        # Create a state vector (this is just an example)
        state = [snake_head.x, snake_head.y, food_position.x, food_position.y]
        return np.array(state, dtype=np.float32)

    def remember(self, state, action, log_prob, value, state_new):
        self.memory.append((state, action, log_prob, value, state_new))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        for _ in range(K_EPOCHS):
            # Sample a batch from memory
            batch = random.sample(self.memory, BATCH_SIZE)
            states, actions, old_logprobs, old_values, next_states = zip(*batch)

            # Calculate advantages and returns
            advantages, returns = self.calculate_advantages_returns(old_values, next_states)

            # Update the actor-critic network using PPO
            self.ppo.update(states, actions, old_logprobs, old_values, advantages, returns)

    def calculate_advantages_returns(self, old_values, next_states):
        advantages = []
        returns = []
        advantage = 0
        prev_value = 0

        for i in reversed(range(len(old_values) - 1)):  # Adjust the loop indexing
            delta = GAMMA * old_values[i + 1] - old_values[i]
            advantage = delta + GAMMA * advantage
            prev_value = old_values[i]

            advantages.insert(0, advantage)
            returns.insert(0, advantage + prev_value)

        advantages = (torch.tensor(advantages) - torch.tensor(advantages).mean()) / (torch.tensor(advantages).std() + 1e-10)

        return advantages, returns




    def select_action(self, state):
        # Convert state to a tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0)
        
        logits, _ = self.actor_critic(state)
        action_probs = F.softmax(logits, dim=-1)
        
        # Ensure action_probs is a 1D array (flattened) for a single action selection
        action_probs = action_probs.view(-1).detach().numpy()
        
        action = np.random.choice(len(action_probs), p=action_probs)
        return action


def train():
    agent = PPOAgent(state_dim, num_actions)
    game = SnakeGamePPO()
    scores = []  # Initialize an empty list to store scores
    mean_scores = []  # Initialize an empty list to store mean scores

    while True:
        state_old = agent.get_state(game)
        action = agent.select_action(state_old)
        log_prob, value = agent.actor_critic(torch.FloatTensor(state_old))

        _, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        agent.remember(state_old, action, log_prob, value, state_new)
        agent.train()

        if done:
            game.reset()
            agent.n_games += 1
            scores.append(score)  # Append the score to the scores list
            if agent.n_games % 10 == 0:  # Calculate and plot mean scores every 10 games
                mean_score = np.mean(scores[-10:])
                mean_scores.append(mean_score)
                plot_ppo(scores, mean_scores)  # Call the plot_ppo function

if __name__ == '__main__':
    train()
