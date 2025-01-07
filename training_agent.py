# training_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from maze_env import SimpleMazeEnv

class SimpleDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(SimpleDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Basic setup
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Learning parameters
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Neural Network
        self.model = SimpleDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=2000)
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
    
    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values - 
        #   how good model thinks our current actions are
        #   initially a random guess 
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values 
        #   how good model thinks the best move for the next position is
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            # This is to be used in the loss function which examines the differences between 
            #      1. How good our current position is assumed to be (effectively what model thinks are good actions to take) 
            #      2. What are actually good position (given REWARD) + best score of our next position the model assumes 
            # We want to reduce the gap between the two elements 
            target_q_values = rewards + self.gamma * next_q_values
        
        # Compute loss and optimize
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train_agent(self, env, agent, episodes=200, max_steps=1000):
        training_history = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Choose and perform action
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Store transition and train
                self.store_transition(state, action, reward, next_state, done)
                self.train()

                # Decay epsilon
                if step%100 == 0: 
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Store episode statistics
            training_history.append({
                'episode': episode,
                'total_reward': total_reward,
                'epsilon': self.epsilon
            })
            
            
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}, Steps: {step}")
        
        return training_history

if __name__ == "__main__":
    # Create maze environment
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0]
    ]
    env = SimpleMazeEnv(maze)
    
    # Initialize agent
    state_size = 2  # x and y coordinates
    action_size = 4  # up, down, left, right
    agent = DQNAgent(state_size, action_size)
    
    # Train agent
    history = agent.train_agent(env, agent)
    
    # Save the trained model
    torch.save(agent.model.state_dict(), 'output/dqn_maze_solver.pth')
