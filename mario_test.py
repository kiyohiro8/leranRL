from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import numpy as np
import retro
import time
import matplotlib.pyplot as plt
from matplotlib import animation

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from baselines.common.retro_wrappers import *

from utils import log_dir, callback, CustomRewardAndDoneEnv
from my_policy import MyPPO2


def display_frames_as_movie(frames, fname):
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=66)
    anim.save(fname, writer='ffmpeg')


if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-6-3-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = CustomRewardAndDoneEnv(env)
    env = StochasticFrameSkip(env, n=4, stickprob=0)
    env = Downsample(env, 2)
    env = Rgb2gray(env)
    env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)
    env = Monitor(env, log_dir, allow_early_resets=True)

    env.seed(1)
    set_global_seeds(0)
    env = DummyVecEnv([lambda: env])

    #model = PPO2.load("mario_model", env, verbose=0)
    model = MyPPO2.load("mario_model_6-3_2", env, verbose=0)

    state = env.reset()
    total_reward = 0
    frames = []
    while True:

        frames.append(env.render(mode='rgb_array').copy())
        
        action, _ = model.predict(state)
        state, rewards, done, info = env.step(action)
        
        if done:
            print(info[0]["x_pos"])
            frames.append(env.render(mode='rgb_array').copy())
            break

    display_frames_as_movie(frames, "mario_model_6-3_8m.mp4")
    env.close()