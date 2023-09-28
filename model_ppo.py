import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_size, num_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc_actor = nn.Linear(128, num_actions)  # Output layer for actor
        self.fc_critic = nn.Linear(128, 1)  # Output layer for critic

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits_actor = self.fc_actor(x)  # Logits for actor
        value = self.fc_critic(x)  # Value for critic
        return logits_actor, value


class PPO:
    def __init__(self, input_size, num_actions, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.actor_critic = ActorCritic(input_size, num_actions)
        self.optimizer_actor = optim.Adam(self.actor_critic.fc_actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.actor_critic.fc_critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip

    def select_action(self, state):
        state = torch.FloatTensor(state)
        logits, _ = self.actor_critic(state)
        action_probs = F.softmax(logits, dim=-1)
        action = np.random.choice(len(action_probs[0]), p=action_probs[0].detach().numpy())
        return action

    # Inside PPO class
 # Add this update method
    def update(self, states, actions, old_logprobs, old_values, advantages, returns):
        for _ in range(self.K_epochs):
            new_logprobs, values = self.actor_critic(states)
            entropy = -torch.mean(new_logprobs)

            ratios = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2)
            critic_loss = F.mse_loss(returns, values)

            loss = actor_loss.mean() + 0.5 * critic_loss.mean() - 0.01 * entropy

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

