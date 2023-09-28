import random
import torch
import numpy as np
from collections import deque
from game_ppo import SnakeGamePPO
from model_ppo import ActorCritic, PPO
import torch.nn.functional as F

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

    def remember(self, state, action, log_prob, value, reward, next_state, done):
        self.memory.append((state, action, log_prob, value, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        for _ in range(K_EPOCHS):
            # Sample a batch from memory
            batch = random.sample(self.memory, BATCH_SIZE)
            states, actions, old_logprobs, old_values, rewards, next_states, dones = zip(*batch)

            # Calculate advantages and returns
            advantages, returns = self.calculate_advantages_returns(rewards, old_values, next_states, dones)

            # Update the actor-critic network using PPO
            self.ppo.update(states, actions, old_logprobs, old_values, advantages, returns)

    def calculate_advantages_returns(self, rewards, old_values, next_states, dones):
        advantages = []
        returns = []
        advantage = 0

        for i in reversed(range(len(rewards))):
            if dones[i]:
                delta = rewards[i] - old_values[i]
                advantage = delta
            else:
                delta = rewards[i] + GAMMA * old_values[i + 1] - old_values[i]
                advantage = delta + GAMMA * advantage
                advantages.insert(0, advantage)

        returns = [a + v for a, v in zip(advantages, old_values)]
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        return advantages, returns


    def select_action(self, state):
        state = torch.FloatTensor(state)
        logits, _ = self.actor_critic(state)
        action_probs = F.softmax(logits, dim=-1)
        
        if len(action_probs.shape) != 2:
            print(f"action_probs shape is unexpected: {action_probs.shape}")
        
        action = np.random.choice(len(action_probs[0]), p=action_probs[0].detach().numpy())
        return action


def train():
    agent = PPOAgent(state_dim, num_actions)
    game = SnakeGamePPO()

    while True:
        state_old = agent.get_state(game)
        action = agent.select_action(state_old)
        log_prob, value = agent.actor_critic(torch.FloatTensor(state_old))

        reward, done, _ = game.play_step(action)  # Removed the score variable
        state_new = agent.get_state(game)

        agent.remember(state_old, action, log_prob, value, reward, state_new, done)
        agent.train()

        if done:
            game.reset()
            agent.n_games += 1


if __name__ == '__main__':
    train()
