import torch
import numpy as np
import matplotlib.pyplot as plt
from maze_env import SimpleMazeEnv
from training_agent import SimpleDQN
import time

def test_trained_agent(model_path, env, max_steps=50):
    # Load the trained model
    state_size = 2
    action_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize and load the model
    model = SimpleDQN(state_size, action_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Reset environment
    state = env.reset()
    env.render()
    
    # List to store path
    path = [state]
    
    for step in range(max_steps):
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(device)
        
        # Get action from model
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
        
        # Take step in environment
        next_state, reward, done, _ = env.step(action)
        
        # Update visualization
        env.render()
        time.sleep(0.1)  # Pause to make visualization visible
        
        # Store path
        path.append(next_state)
        
        # Update state
        state = next_state

        print(f'Step: {step} Action: {action}')
        
        if done:
            print(f"Goal reached in {step + 1} steps!")
            break
    
    return path

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
    
    # Test the trained agent
    model_path = 'output/dqn_maze_solver.pth'
    path = test_trained_agent(model_path, env)
    
    # Keep the visualization window open
    plt.show()