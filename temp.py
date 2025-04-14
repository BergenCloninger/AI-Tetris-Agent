import random
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.autograd import Variable
from skimage.measure import label, regionprops

np.set_printoptions(threshold=np.inf)

gym.register_envs(ale_py)
env = gym.make('ALE/Tetris-v5', render_mode=None)  # Disable rendering during training

obs, info = env.reset()

def process_image_to_grid(image):  # Process game board into 22x10 binary grid
    x_start, y_start = 22, 27
    grid_width, grid_height = 42, 175

    cropped_image = image[y_start:y_start + grid_height, x_start:x_start + grid_width]

    intensity = np.sum(cropped_image, axis=-1)  # Sum of RGB

    threshold = 350  # Grey color of background has an intensity of 333
    binary_image = (intensity > threshold).astype(int)

    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)

    new_grid_height, new_grid_width = 22, 10  # Size of the Tetris grid
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

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

input_dim = 22 * 10
output_dim = env.action_space.n
model = QNetwork(input_dim, output_dim)
target_model = QNetwork(input_dim, output_dim)
target_model.load_state_dict(model.state_dict())  # Initialize target model with the same weights
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.1
n_epochs = 300
batch_size = 64
memory = deque(maxlen=100000)
frames_since_last_spawned_piece = 0

def select_action(state):
    if np.random.rand() < epsilon:  # Random action if generated number is < epsilon, otherwise best action
        return np.random.choice(output_dim)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state).float().unsqueeze(0)
            q_values = model(state_tensor)
            return torch.argmax(q_values).item()  # Best action

previous_game_grid = None
static_game_grid = None

def check_grid(game_grid):
    global frames_since_last_spawned_piece
    found_piece = False

    for col in range(game_grid.shape[1]):
        for row in range(game_grid.shape[0]):
            if game_grid[row, col] == 1:
                if (game_grid.shape[0] - row > 19 and frames_since_last_spawned_piece > 10): #scuffed, but required for it to work properly
                    frames_since_last_spawned_piece = 0
                if (game_grid.shape[0] - row == 22 and frames_since_last_spawned_piece == 0):
                    found_piece = True
                    break
                else:
                    found_piece = False
        if found_piece:
            break

def compute_grid_features(game_grid):
    total_heights = 0
    bumpiness = 0
    holes = 0
    y_pos = 0

    column_heights = []
    for col in range(game_grid.shape[1]):
        for row in range(game_grid.shape[0]):
            if game_grid[row, col] == 1:
                column_heights.append(game_grid.shape[0] - row)
                break
        else:
            column_heights.append(0) 

    total_heights = sum(column_heights)

    for i in range(1, len(column_heights)):
        bumpiness += abs(column_heights[i] - column_heights[i - 1])

    for col in range(game_grid.shape[1]):
        filled = False
        for row in range(game_grid.shape[0]):
            if game_grid[row, col] == 1:
                filled = True
            elif filled and game_grid[row, col] == 0:
                holes += 1

    for row in range(game_grid.shape[0]):
        if 1 in game_grid[row]:
            y_pos = row
            break
    return total_heights, bumpiness, holes, y_pos

def calculate_rewards(game_grid, survived_steps, total_lines_cleared, terminated):
    global previous_game_grid
    global static_game_grid
    check_grid(game_grid)
    if (frames_since_last_spawned_piece == 0):
        static_game_grid = previous_game_grid
    total_heights, bumpiness, holes, y_pos = compute_grid_features(static_game_grid)

    calc_reward = 0

    # Board is half full or bumpiness is high
    board_half_full = total_heights >= 110 or (total_heights >= 90 and bumpiness >= 10)

    if total_heights >= 140 or (total_heights >= 110 and bumpiness >= 12):
        hole_penalty = -2.743561101942274
    elif total_heights >= 90 or (total_heights >= 70 and bumpiness >= 9):
        hole_penalty = -4.743561101942274
    else:
        hole_penalty = -1

    pillar_penalty = 0
    if holes > 0 or board_half_full:
        pillar_penalty = -1

    high_placement_penalty = 0
    if total_heights <= 40:
        high_placement_penalty = (10 - y_pos) * 2
    elif total_heights <= 100: 
        high_placement_penalty = (10 - y_pos)

    if y_pos >= 12:
        calc_reward -= high_placement_penalty

    if terminated:
        calc_reward -= 10

    survival_reward = 0.01 * survived_steps
    calc_reward += survival_reward

    if y_pos >= 9:
        calc_reward += 5
    else:
        calc_reward -= (10 - y_pos) * 0.2

    height_penalty = 0.05 * total_heights
    calc_reward -= height_penalty

    line_clear_reward = (2 ** total_lines_cleared) * 10
    calc_reward += line_clear_reward

    if total_lines_cleared == 4:
        calc_reward += 5000

    hole_penalty_total = hole_penalty * holes
    calc_reward += hole_penalty_total

    bumpiness_penalty = 0.1 * bumpiness
    calc_reward -= bumpiness_penalty

    calc_reward += pillar_penalty

    return calc_reward

for epoch in range(n_epochs):
    state, info = env.reset()
    done = False
    total_reward = 0
    survived_steps = 0
    game_grid = process_image_to_grid(state)
    previous_game_grid = game_grid
    static_game_grid = game_grid

    while not done:
        previous_game_grid = game_grid

        game_grid = process_image_to_grid(state)
        state_flat = game_grid.flatten()

        action = select_action(state_flat)

        next_state, total_lines_cleared, terminated, truncated, info = env.step(action)

        survived_steps += 1

        reward = calculate_rewards(game_grid, survived_steps, total_lines_cleared, terminated=False)

        total_reward += reward

        game_grid_next = process_image_to_grid(next_state)
        next_state_flat = game_grid_next.flatten()

        # Add to memory
        memory.append((state_flat, action, reward, next_state_flat, terminated or truncated))

        # Ensure we have enough samples in memory before starting training
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            for s, a, r, next_s, done in batch:
                state_tensor = torch.tensor(s).float().unsqueeze(0)
                next_state_tensor = torch.tensor(next_s).float().unsqueeze(0)
                q_values = model(state_tensor)
                next_q_values = target_model(next_state_tensor)

                target = r + (gamma * torch.max(next_q_values) * (1 - done))

                target_f = q_values.clone()
                target_f[0][a] = target

                loss = criterion(q_values, target_f)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        state = next_state
        frames_since_last_spawned_piece = frames_since_last_spawned_piece + 1

        done = terminated or truncated



    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    if epoch % 1 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs} | Total Reward: {total_reward} | Epsilon: {epsilon}")

    if epoch % 10 == 0:
        target_model.load_state_dict(model.state_dict())  # Update target model

n_test_runs = 20  # Display agent n times after training
env = gym.make('ALE/Tetris-v5', render_mode='human')  # Enable rendering to test

for run in range(n_test_runs):
    state, info = env.reset()
    done = False
    total_reward = 0
    total_score = 0
    epsilon = 0

    print(f"Run {run + 1}/{n_test_runs}")
    
    while not done:
        game_grid = process_image_to_grid(state)
        state_flat = game_grid.flatten()

        action = select_action(state_flat)

        next_state, reward, terminated, truncated, info = env.step(action)

        current_score = info.get('score', 0)
        total_score = current_score

        total_reward += reward
        state = next_state

        done = terminated or truncated

    print(f"Total Reward for Run {run + 1}: {total_reward} | Final Score: {total_score}")

env.close()