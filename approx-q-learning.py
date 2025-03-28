import gymnasium as gym
import ale_py
import numpy as np
import random
import time
from skimage.measure import label, regionprops

np.set_printoptions(threshold=np.inf)

# Register the environment (make sure to not include the render_mode during training)
gym.register_envs(ale_py)
env = gym.make('ALE/Tetris-v5', render_mode=None)  # No rendering during training

obs, info = env.reset()

# Function to process the image into a 22x10 binary grid
def process_image_to_grid(image):
    x_start, y_start = 22, 27
    grid_width, grid_height = 42, 175

    cropped_image = image[y_start:y_start + grid_height, x_start:x_start + grid_width]

    intensity = np.sum(cropped_image, axis=-1)  # Sum of RGB

    threshold = 350  # grey color has an intensity of 333
    binary_image = (intensity > threshold).astype(int)
    
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)

    new_grid_height, new_grid_width = 22, 10  # size of the tetris grid
    resized_grid = np.zeros((new_grid_height, new_grid_width), dtype=int)

    grid_height, grid_width = binary_image.shape

    block_height = grid_height / new_grid_height
    block_width = grid_width / new_grid_width

    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox

        center_y = (min_row + max_row) / 2
        center_x = (min_col + max_col) / 2

        grid_y = int(center_y // block_height)
        grid_x = int(center_x // block_width)

        grid_y = min(grid_y, new_grid_height - 1)
        grid_x = min(grid_x, new_grid_width - 1)

        if region.area > 9:
            resized_grid[grid_y, grid_x] = 1

    return resized_grid

gamma = 0.99
alpha = 0.01
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.1
n_actions = env.action_space.n
n_epochs = 200

q_table = {}

def get_state_key(grid):
    """Convert the game grid into a tuple to use as a key in the Q-table"""
    return tuple(grid.flatten())

def select_action(state_key):
    """Select action using epsilon-greedy policy"""
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # Explore: random action
    else:
        q_values = q_table.get(state_key, np.zeros(n_actions))
        return np.argmax(q_values)

global previous_height
global current_height
previous_height = 0
current_height = 0

def compute_height(grid):
    global current_height
    global previous_height

    column_heights = []
    for col in range(grid.shape[1]):
        for row in range(grid.shape[0]):
            if grid[row, col] == 1:
                column_heights.append(grid.shape[0] - row)  # Height is distance from bottom
                break
        else:
            column_heights.append(0)  # No block in this column

    height = max(column_heights)  # The max height across all columns
    if height > 18:
        current_height = previous_height
    else:
        previous_height = height

    return current_height

def compute_reward(grid, survived_steps):
    lines_cleared = np.sum(np.all(grid == 1, axis=1))
    
    height = compute_height(grid)
    
    if height < 5:
        height_bonus = height * 2
        height_penalty = 0
    else:
        height_penalty = height * 2
        height_bonus = 0
    
    gaps = 0
    for col in range(grid.shape[1]):
        column = grid[:, col]
        first_one = np.argmax(column == 1) if np.any(column == 1) else len(column)
        last_one = len(column) - np.argmax(np.flip(column == 1)) - 1 if np.any(column == 1) else -1
        if first_one < last_one:
            gaps += last_one - first_one - 1
    gap_penalty = gaps

    survival_bonus = 0.001 * survived_steps
    
    # Final reward calculation
    reward = (lines_cleared * 50) - height_penalty + survival_bonus - gap_penalty + height_bonus
    return reward

# Training loop
for epoch in range(n_epochs):
    state, info = env.reset()
    done = False
    total_reward = 0
    total_score = 0
    survived_steps = 0  # Track the number of survived steps

    while not done:
        game_grid = process_image_to_grid(state)
        state_key = get_state_key(game_grid)

        action = select_action(state_key)

        next_state, env_reward, terminated, truncated, info = env.step(action)

        current_score = info.get('score', 0)
        total_score = current_score

        # Increment survived steps
        survived_steps += 1

        # Calculate the custom reward using compute_reward
        reward = compute_reward(game_grid, survived_steps)

        total_reward += reward

        game_grid_next = process_image_to_grid(next_state)
        next_state_key = get_state_key(game_grid_next)

        q_values = q_table.get(state_key, np.zeros(n_actions))
        next_q_values = q_table.get(next_state_key, np.zeros(n_actions))

        target = reward + gamma * np.max(next_q_values)

        q_values[action] = (1 - alpha) * q_values[action] + alpha * target

        q_table[state_key] = q_values

        state = next_state

        done = terminated or truncated

    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{n_epochs} | Total Reward: {total_reward} | Epsilon: {epsilon} | Score: {total_score}")


# Run the trained agent for 3 episodes
n_test_runs = 3
env = gym.make('ALE/Tetris-v5', render_mode='human')  # Rendering enabled for testing

for run in range(n_test_runs):

    state, info = env.reset()
    done = False
    total_reward = 0
    total_score = 0
    epsilon = 0

    print(f"Run {run + 1}/{n_test_runs}")
    
    while not done:
        game_grid = process_image_to_grid(state)
        state_key = get_state_key(game_grid)

        action = select_action(state_key)

        next_state, reward, terminated, truncated, info = env.step(action)

        current_score = info.get('score', 0)
        total_score = current_score

        total_reward += reward

        state = next_state

        done = terminated or truncated

    print(f"Total Reward for Run {run + 1}: {total_reward} | Final Score: {total_score}")

env.close()