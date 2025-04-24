import pygame
import gymnasium as gym
import ale_py
import numpy as np
from skimage.measure import label, regionprops

pygame.init()

# === Setup the Tetris environment ===
gym.register_envs(ale_py)
env = gym.make('ALE/Tetris-v5', render_mode='human')

# === Globals ===
frames_since_last_spawned_piece = 0
previous_game_grid = None
static_game_grid = None
lines_cleared = 0  # Global variable to track how many lines cleared this step

# === Keyboard Input Mapping ===
def get_keyboard_input():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]: return 3  # Left
    if keys[pygame.K_RIGHT]: return 2  # Right
    if keys[pygame.K_DOWN]: return 1  # Down
    if keys[pygame.K_UP]: return 0    # Rotate
    if keys[pygame.K_SPACE]: return 4 # Drop
    return -1

# === Image to Grid Conversion ===
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

    block_height = binary_image.shape[0] / new_grid_height
    block_width = binary_image.shape[1] / new_grid_width

    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        center_y = (min_row + max_row) / 2
        center_x = (min_col + max_col) / 2
        grid_y = min(int(center_y // block_height), new_grid_height - 1)
        grid_x = min(int(center_x // block_width), new_grid_width - 1)
        if region.area > 9:
            resized_grid[grid_y, grid_x] = 1

    return resized_grid

# === Check Grid for New Piece Spawn ===
def check_grid(game_grid):
    global frames_since_last_spawned_piece
    found_piece = False
    for col in range(game_grid.shape[1]):
        for row in range(game_grid.shape[0]):
            if game_grid[row, col] == 1:
                if (game_grid.shape[0] - row > 20 and frames_since_last_spawned_piece > 5):
                    frames_since_last_spawned_piece = 0
                if (game_grid.shape[0] - row == 22 and frames_since_last_spawned_piece == 0):
                    found_piece = True
                    break
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

    # New: define "hole density" = holes / total columns with at least one block
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

    # Heavily penalize holes
    reward += weights['hole_penalty'] * holes

    # Bonus for low stacks with no holes
    if max_height <= 12 and holes == 0:
        reward += weights['clean_stack_bonus']
        reward += weights['low_stack_bonus'] * (1 - (max_height / 12))  # Bonus scales with how low it is

    # Slight penalty for uneven stacks (flatness)
    if max_height > 0:
        flatness = sum(abs(column_heights[i] - column_heights[i - 1]) for i in range(1, len(column_heights)))
        reward += weights['flatness_penalty'] * flatness

    # Game over
    if terminated:
        reward -= weights['game_over']

    # Reward for lines cleared
    if lines_cleared > 0:
        reward += (2 ** lines_cleared) * weights['line_clear_base']
        if lines_cleared == 4:
            reward += weights['tetris_bonus']

    # Only return the reward once when a new piece spawns
    if frames_since_last_spawned_piece == 0:
        previous_game_grid = np.copy(static_game_grid)
        lines_cleared = 0
        return reward

    # If no new piece has spawned, skip reward
    return 0


# === Main Loop ===
n_test_runs = 20
for run in range(n_test_runs):
    state, info = env.reset()
    done = False
    total_reward = 0
    total_score = 0
    survived_steps = 0

    game_grid = process_image_to_grid(state)
    previous_game_grid = game_grid
    static_game_grid = game_grid

    print(f"\nRun {run + 1}/{n_test_runs}")

    while not done:
        print(static_game_grid)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        game_grid = process_image_to_grid(state)
        action = get_keyboard_input()
        if action == -1:
            continue

        next_state, temp_lines_cleared, terminated, truncated, info = env.step(action)

        survived_steps += 1
        frames_since_last_spawned_piece += 1

        reward = calculate_rewards(game_grid, survived_steps, temp_lines_cleared, terminated)

        print(f"Lines cleared this step: {lines_cleared}")
        print(f"Reward at step: {reward}")

        total_reward += reward

        previous_game_grid = game_grid
        state = next_state
        done = terminated or truncated

    print(f"Total Reward for Run {run + 1}: {total_reward} | Final Score: {info.get('score', 0)}")

env.close()
pygame.quit()
