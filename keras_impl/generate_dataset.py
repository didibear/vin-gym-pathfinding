import gym
import gym_pathfinding

from gym_pathfinding.games.gridworld import generate_grid, MOUVEMENT
from gym_pathfinding.games.astar import astar
from tqdm import tqdm
import numpy as np
import operator
import itertools

ACTION_SIZE = 4

def generate_dataset(size, shape, *, grid_type="free", verbose=False):
    """
    Arguments
    ---------
    size : number of training set generated
    shape : the grid shape
    grid_type : the type of grid ("free", "obstacle", "maze")

    Return
    ------
    return states, goals, starts, actions

    state : grid like shape with 1 and 0
    goal : grid like shape with 1 at goal position
    start : (1, 1) player position
    action : (4) the action (in one hot shape)
    """
    if verbose: progress_bar = tqdm(total=size)

    states = []
    goals = []
    starts = []
    actions = []
    n = 0

    while True:

        grid, start, goal = generate_grid(shape, grid_type=grid_type)
        path, action_planning = compute_action_planning(grid, start, goal)

        goal_grid = create_goal_grid(grid.shape, goal)

        for action, position in zip(action_planning, path):
            states.append(grid)
            goals.append(goal_grid)
            starts.append(position)
            actions.append(one_hot_value(ACTION_SIZE, action))            

            if verbose : progress_bar.update(1)

            n += 1 
            if n >= size:
                if verbose : progress_bar.close()
                return states, goals, starts, actions

# reversed MOUVEMENT dict
ACTION = {mouvement: action for action, mouvement in dict(enumerate(MOUVEMENT)).items()}

def compute_action_planning(grid, start, goal):
    path = astar(grid, start, goal)

    action_planning = []
    for i in range(len(path) - 1):
        pos = path[i]
        next_pos = path[i+1]
        
        # mouvement = (-1, 0), (1, 0), (0, -1), (0, 1)
        mouvement = tuple(map(operator.sub, next_pos, pos))

        action_planning.append(ACTION[mouvement])
        
    return path, action_planning


def create_goal_grid(shape, goal):
    goal_grid = np.zeros(shape, dtype=np.int8)
    goal_grid[goal] = 1
    return goal_grid

def one_hot_value(size, value):
    one_hot = np.zeros((size))
    one_hot[value] = 1
    return one_hot


def main():
    import joblib
    import argparse

    parser = argparse.ArgumentParser(description='Generate training data (states, goals, starts, actions)')
    parser.add_argument('--out', '-o', type=str, default='./data/training_data.pkl', help='Path to save the training_data')
    parser.add_argument('--size', '-s', type=int, default=10000, help='Number of training example')
    parser.add_argument('--shape', type=int, default=[9, 9], nargs=2, help='Shape of the grid (e.g. --shape 9 9)')
    parser.add_argument('--grid_type', type=str, default='obstacle', help='Type of grid : "free", "obstacle" or "maze"')
    args = parser.parse_args()

    training_data = generate_dataset(args.size, args.shape, 
        grid_type=args.grid_type, verbose=True
    )

    print("saving data into : {}".format(args.out))

    joblib.dump(training_data, args.out)

    print("done")

if __name__ == "__main__":
    main()
