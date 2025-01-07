import torch, numpy as np

def print_q_values_for_all_states(agent, maze_size):
    """
    Prints the Q-values for all states (positions) in the maze.
    :param agent: DQNAgent object containing the model.
    :param maze_size: Tuple (rows, cols) representing the size of the maze.
    """
    rows, cols = maze_size
    for i in range(rows):
        for j in range(cols):
            state = np.array([i, j])  # Representing the position (i, j)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.model(state_tensor).cpu().data.numpy()
            print(f"Q-values for state ({i}, {j}): {q_values}")
