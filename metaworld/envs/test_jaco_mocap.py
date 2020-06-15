#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym

import jaco_env

# Register a keyboard handler
import glfw

from gym.envs.registration import registry, register, make, spec

register(
    id='Jaco-v0',
    entry_point='jaco_env:JacoEnv',
    max_episode_steps=200,
)


def on_press(window, key, scancode, action, mods):
    global action_to_take

    pos = np.zeros(6)
    _pos_step = 1.0
    
    # controls for moving position
    if key == glfw.KEY_A:
        pos[1] -= _pos_step  # dec x
    elif key == glfw.KEY_D:
        pos[1] += _pos_step  # inc x
    elif key == glfw.KEY_W:
        pos[0] -= _pos_step  # dec y
    elif key == glfw.KEY_S:
        pos[0] += _pos_step  # inc y
    elif key == glfw.KEY_DOWN:
        pos[2] -= _pos_step  # dec z
    elif key == glfw.KEY_UP:
        pos[2] += _pos_step  # inc z
    elif key == glfw.KEY_LEFT:
        pos[3:] = 0
    elif key == glfw.KEY_RIGHT:
        pos[3:] = 1
    elif key == glfw.KEY_ESCAPE:
        exit()

    action_to_take = pos

"""

As a reference, you can also register a scroll callback, but then you can't
do things with zoom!

def on_scroll_action(window, x_offset, y_offset):
    global action_to_take, grasp, action_array
    
    grasp = np.clip(grasp + 0.05 * y_offset, -1, 1)
    action_to_take = [0., 0., 0., grasp]
    action_array.append(action_to_take)
"""
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="Jaco-v0")
    parser.add_argument('--seed', default=1, type=int, help='seed')
    parser.add_argument('--record',action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env._max_episode_steps = 10000
    env.seed(args.seed)
    env.reset()

    env.render()

    action_to_take = np.zeros(6)

    glfw.set_key_callback(env.unwrapped.viewer.window, on_press)
    # glfw.set_scroll_callback(env.unwrapped.viewer.window, on_scroll_action)

    while True:
        env.render()

        if not np.array_equal(action_to_take, np.zeros(6)):
            _, _, d , _ = env.step(action_to_take)
            if d: 
                env.seed(args.seed)
                env.reset()                   
                env.render()
            
            # @Melissa: Commenting this out makes the mocap faster but introduces some instabilities.
            action_to_take = np.zeros(6)