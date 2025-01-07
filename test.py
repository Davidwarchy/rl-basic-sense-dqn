# test.py
import torch
import numpy as np
from maze_env import SimpleMazeEnv
from training_agent import SimpleDQN, DQNAgent
import time

def test_trained_agent(env, model_path, max_steps=50):
    # Initialize agent with same parameters as training
    agent = DQNAgent()
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()
    agent.epsilon = 0  # No exploration during testing
    
    # Reset environment and get initial state
    pos = env.reset()
    state = agent.get_state(env, pos)
    env.render()
    
    path = [pos]
    
    for step in range(max_steps):
        # Get action from model
        action = agent.choose_action(state)
        
        # Take step in environment
        next_pos, reward, done, _ = env.step(action)
        next_state = agent.get_state(env, next_pos)
        
        # Update visualization
        env.render()
        time.sleep(0.1)
        
        # Store path
        path.append(next_pos)
        
        print(f'Step: {step + 1}, Action: {action}, Position: {next_pos}')
        
        if done:
            print(f"Goal reached in {step + 1} steps!")
            break
            
        state = next_state
        
    return path

if __name__ == "__main__":
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0]
    ]
    env = SimpleMazeEnv(maze)
    
    model_path = 'dqn_maze_solver.pth'
    path = test_trained_agent(env, model_path)
    
    # Keep visualization window open
    input("Press Enter to close...")