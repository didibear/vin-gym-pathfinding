import gym
import gym_pathfinding
from time import sleep
from model import vin_model
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='VIN')
    parser.add_argument('--model', '-m', type=str, default='./model/weights-checkpoint.h5', help='Model from given file')
    args = parser.parse_args()

    env = gym.make('pathfinding-free-9x9-v0')

    k = 20
    model = vin_model(n=9, k=k, Q_size=4)
    model.load_weights(args.model)

    for episode in range(5):
        state = env.reset()

        for timestep in range(10):

            state, goal, start = parse_state(state)

            action_probabilities = model.predict([np.array([state]), np.array([goal]), np.array(start)])
            print(action_probabilities)
            action = np.argmax(action_probabilities)

            if action == 0:
                action = 1
            elif action == 1:
                action = 0
            elif action == 2:
                action = 3
            elif action == 3:
                action = 2
            
            
            # reward = get_layer_output(model, 'reward', im_ary)
            # value = get_layer_output(model, 'value{}'.format(k), im_ary)
            # reward = np.reshape(reward, state.shape)
            # value = np.reshape(value, state.shape)
            
            env.render()
            sleep(0.2)
            state, reward, done, _ = env.step(action)

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
