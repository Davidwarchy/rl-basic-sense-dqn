# maze_env.py
import gym
import numpy as np
import matplotlib.pyplot as plt

class SimpleMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, maze):
        super(SimpleMazeEnv, self).__init__()
        self.maze = maze
        self.goal_position = (len(maze)-1, len(maze[0])-1)  # Goal position is bottom-right corner
        self.agent_position = (0, 0)  # Agent starts at top-left corner
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.Discrete(len(maze) * len(maze[0]))  # Total number of states

        # Visualization setup
        self.fig, self.ax = plt.subplots()
        self.ax.set_xticks(np.arange(-.5, len(maze[0]), 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, len(maze), 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        self.ax.invert_yaxis()  # Flip the y-axis
        
        self.agent_marker = self.ax.scatter(*self._maze_to_plot(self.agent_position), color='red', marker='o', s=100)
        self.goal_marker = self.ax.scatter(*self._maze_to_plot(self.goal_position), color='green', marker='*', s=100)
        self.ax.imshow(self.maze, cmap='binary', interpolation='nearest')

    def _maze_to_plot(self, position):
        """Convert maze coordinates to plot coordinates."""
        return (position[1], position[0])

    def _plot_to_maze(self, position):
        """Convert plot coordinates to maze coordinates."""
        return (position[1], position[0])

    def reset(self):
        self.agent_position = (0, 0)  # Reset agent to top-left corner
        self.agent_marker.set_offsets(self._maze_to_plot(self.agent_position))
        return self.agent_position

    def step(self, action):
        x, y = self.agent_position

        if action == 0:  # Up
            next_position = (x - 1, y)
        elif action == 1:  # Down
            next_position = (x + 1, y)
        elif action == 2:  # Left
            next_position = (x, y - 1)
        elif action == 3:  # Right
            next_position = (x, y + 1)
        else:
            raise ValueError("Invalid action")

        # Check if the next position is within bounds and not a wall
        if (0 <= next_position[0] < len(self.maze) and
            0 <= next_position[1] < len(self.maze[0]) and
            self.maze[next_position[0]][next_position[1]] != 1):  # 1 represents a wall
            self.agent_position = next_position
            self.agent_marker.set_offsets(self._maze_to_plot(self.agent_position))
        else:
            # Penalize for hitting a wall or going out of bounds
            return self.agent_position, -1.0, False, {}

        # Determine reward
        if self.agent_position == self.goal_position:
            reward = 200.0  # Reward for reaching the goal
            done = True
        elif self.maze[self.agent_position[0]][self.agent_position[1]] == 1:
            reward = -1.0  # Penalty for hitting a wall
            done = False
        else:
            reward = 0.1  # Small reward for moving towards the goal
            done = False

        return self.agent_position, reward, done, {}

    def render(self, mode='human'):
        self.fig.canvas.draw()
        plt.pause(0.1)  # Adjust the pause duration for visualization

    def close(self):
        plt.close(self.fig)

    def update_agent_position(self, new_position):
        self.agent_position = new_position
        self.agent_plot_position = self._maze_to_plot(self.agent_position)
        self.agent_marker.set_offsets(self.agent_plot_position)
        self.fig.canvas.draw()

if __name__ == "__main__":
    
    # Example usage:
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0]
    ]

    env = SimpleMazeEnv(maze)
    env.reset()
    env.render()

    # Example of changing coordinates
    new_positions = [(0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4)]
    for pos in new_positions:
        env.update_agent_position(pos)
        plt.pause(.5)  # Pause to visualize the change

    x = input()