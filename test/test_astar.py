import gym
import gym_pathfinding
import operator
from time import sleep

from gym_pathfinding.games.gridworld import generate_grid, MOUVEMENT
from astar import astar

ACTION = {mouv: action for action, mouv in dict(enumerate(MOUVEMENT)).items()}
env = gym.make('pathfinding-obstacle-25x25-v0')
# env.seed(5) # full deterministic env

for episode in range(20):
    s = env.reset()
    path = astar(env.game.grid, env.game.player, env.game.target)
    

    # compute action planning
    action_planning = []
    for i in range(len(path) - 1):
        pos = path[i]
        next_pos = path[i+1]
        
        mouvement = tuple(map(operator.sub, next_pos, pos))

        action_planning.append(ACTION[mouvement])

    # Use action in env
    for timestep in range(50):
        env.render()
        sleep(0.1)

        action = action_planning[timestep]

        s, r, done, _ = env.step(action)

        if done:
            break


env.close()