import gym
import gym_pathfinding
from time import sleep
from model import vin_model
import numpy as np
import argparse
import random
from itertools import permutations

def main():
    parser = argparse.ArgumentParser(description='VIN')
    parser.add_argument('--model', '-m', type=str, default='./model/weights-checkpoint.h5', help='Model from given file')
    args = parser.parse_args()

    env = gym.make('pathfinding-free-9x9-v0')
    

    k = 10
    model = vin_model(n=9, k=k, Q_size=4)
    model.load_weights(args.model)

    actions = (2, 3, 0, 1)
    for episode in range(10):
        env.seed(episode)
        state = env.reset()

        for timestep in range(20):

            state, goal, start = parse_state(state)

            action_probabilities = model.predict([np.array([state]), np.array([goal]), np.array(start)])

            # if (random.random() < 0.1):
                # action = env.action_space.sample()
            # else :
            action = np.argmax(action_probabilities)
            
            env.render()
            sleep(0.05)
            state, reward, done, _ = env.step(actions[action])
            if done:
                break

    env.close()

def parse_state(state):
    goal = np.argwhere(state == 2)
    state[state == 2] = 0

    start = np.argwhere(state == 3)
    state[state == 3] = 0

    return state, create_goal_grid(state.shape, goal), start

def create_goal_grid(shape, goal):
    goal_grid = np.zeros(shape, dtype=np.int8)
    goal_grid[goal] = 1
    return goal_grid

if __name__ == "__main__":
    main()
