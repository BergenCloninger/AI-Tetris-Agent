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
env = gym.make('ALE/Tetris-v5', render_mode=None)

obs, info = env.reset()

tetromino_shapes = {
    0: np.array([[1, 1, 1, 1]]),                      # I
    1: np.array([[1, 1], [1, 1]]),                    # O
    2: np.array([[0, 1, 0], [1, 1, 1]]),              # T
    3: np.array([[1, 0], [1, 0], [1, 1]]),            # L
    4: np.array([[0, 1], [0, 1], [1, 1]]),            # J
    5: np.array([[0, 1, 1], [1, 1, 0]]),              # S
    6: np.array([[1, 1, 0], [0, 1, 1]])               # Z
}

def match_tetromino(shape):
    shape = np.array(shape)

    if shape.shape == (2, 2) and np.all(shape == 1):
        return 1

    for tid, template in tetromino_shapes.items():
        for k in range(4):
            rotated = np.rot90(template, k)
            if shape.shape == rotated.shape and np.all(shape == rotated):
                return tid
    return -1

def identify_falling_piece(prev_grid, curr_grid, static_grid):
    if prev_grid is None or static_grid is None:
        return -1

    falling_piece_mask = (curr_grid == 1) & (static_grid == 0)
    top_rows = 4
    top_falling = falling_piece_mask[:top_rows]

    labeled = label(top_falling)
    props = regionprops(labeled)

    if not props:
        return -1

    region = max(props, key=lambda r: r.area)
    shape = region.image
    return match_tetromino(shape)

def process_image_to_grid(image):
    x_start, y_start = 22, 27
    grid_width, grid_height = 42, 175

    cropped_image = image[y_start:y_start + grid_height, x_start:x_start + grid_width]

    intensity = np.sum(cropped_image, axis=-1)

    threshold = 350
    binary_image = (intensity > threshold).astype(int)

    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)

    new_grid_height, new_grid_width = 22, 10
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
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

input_dim = 22 * 10 + 1 + 1
output_dim = env.action_space.n
model = QNetwork(input_dim, output_dim)
target_model = QNetwork(input_dim, output_dim)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.1
n_epochs = 1000
batch_size = 64
memory = deque(maxlen=10000)
frames_since_last_spawned_piece = 0
current_falling_piece_label = 1

def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(output_dim)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state).float().unsqueeze(0)
            q_values = model(state_tensor)
            return torch.argmax(q_values).item()

previous_game_grid = None
static_game_grid = None

def check_grid(game_grid):
    global frames_since_last_spawned_piece, current_falling_piece_label
    found_piece = False

    for col in range(game_grid.shape[1]):
        for row in range(game_grid.shape[0]):
            if game_grid[row, col] == 1:
                if (game_grid.shape[0] - row > 20 and frames_since_last_spawned_piece > 10):
                    current_falling_piece_label = identify_falling_piece(previous_game_grid, game_grid, static_game_grid) # 0 = I, 1 = O, 2 = T, 3 = L, 4 = J, 5 = S, 6 = Z
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

    return total_heights, bumpiness, holes, (22 - y_pos), column_heights

def calculate_rewards(game_grid, survived_steps, temp_lines_cleared, terminated):
    global previous_game_grid, static_game_grid, frames_since_last_spawned_piece, lines_cleared

    if temp_lines_cleared > 0:
        lines_cleared = temp_lines_cleared

    check_grid(game_grid)

    if frames_since_last_spawned_piece == 0:
        static_game_grid = previous_game_grid

    if np.count_nonzero(static_game_grid) == 0:
        return 0

    total_heights, bumpiness, holes, y_pos, column_heights = compute_grid_features(static_game_grid)

    active_columns = sum(1 for h in column_heights if h > 0)
    hole_density = holes / active_columns if active_columns else 0

    max_height = max(column_heights)
    min_height = min(h for h in column_heights if h > 0) if active_columns else 0

    weights = {
        'hole_penalty': -5.0,
        'line_clear_base': 100,
        'tetris_bonus': 5000,
        'low_stack_bonus': 3.0,
        'clean_stack_bonus': 10.0,
        'flatness_penalty': -0.1,
        'game_over': 1000
    }

    reward = 0

    reward += weights['hole_penalty'] * holes

    if max_height <= 12 and holes == 0:
        reward += weights['clean_stack_bonus']
        reward += weights['low_stack_bonus'] * (1 - (max_height / 12)) 

    if max_height > 0:
        flatness = sum(abs(column_heights[i] - column_heights[i - 1]) for i in range(1, len(column_heights)))
        reward += weights['flatness_penalty'] * flatness

    if terminated:
        reward -= weights['game_over']

    if lines_cleared > 0:
        reward += (2 ** lines_cleared) * weights['line_clear_base']
        if lines_cleared == 4:
            reward += weights['tetris_bonus']

    if frames_since_last_spawned_piece == 0:
        previous_game_grid = np.copy(static_game_grid)
        lines_cleared = 0
        return reward

    return 0

for epoch in range(n_epochs):
    state, info = env.reset()
    done = False
    total_reward = 0
    survived_steps = 0
    game_grid = process_image_to_grid(state)
    previous_game_grid = game_grid
    static_game_grid = np.zeros((22, 10), dtype=int)
    lines_cleared = 0
    current_falling_piece_label = 1

    while not done:
        previous_game_grid = game_grid

        game_grid = process_image_to_grid(state)
        state_flat = np.concatenate([game_grid.flatten(), [lines_cleared], [current_falling_piece_label]])

        action = select_action(state_flat)

        next_state, temp_lines_cleared, terminated, truncated, info = env.step(action)
        frames_since_last_spawned_piece += 1

        survived_steps += 1

        reward = calculate_rewards(game_grid, survived_steps, temp_lines_cleared, terminated)
        total_reward += reward

        lines_cleared = temp_lines_cleared

        game_grid_next = process_image_to_grid(next_state)
        next_state_flat = np.concatenate([game_grid_next.flatten(), [lines_cleared], [current_falling_piece_label]])

        memory.append((state_flat, action, reward, next_state_flat, terminated or truncated))

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
        frames_since_last_spawned_piece += 1

        done = terminated or truncated

    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    if epoch % 1 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs} | Total Reward: {total_reward} | Epsilon: {epsilon}")

    if epoch % 10 == 0:
        target_model.load_state_dict(model.state_dict())

n_test_runs = 20
env = gym.make('ALE/Tetris-v5', render_mode='human')

for run in range(n_test_runs):
    state, info = env.reset()
    done = False
    total_reward = 0
    total_score = 0
    epsilon = 0

    print(f"Run {run + 1}/{n_test_runs}")
    
    while not done:
        game_grid = process_image_to_grid(state)
        state_flat = np.concatenate([game_grid.flatten(), [lines_cleared], [current_falling_piece_label]])

        action = select_action(state_flat)

        next_state, reward, terminated, truncated, info = env.step(action)

        current_score = info.get('score', 0)
        total_score = current_score

        total_reward += reward
        state = next_state

        done = terminated or truncated

    print(f"Total Reward for Run {run + 1}: {total_reward} | Final Score: {total_score}")

env.close()