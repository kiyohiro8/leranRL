import os
import datetime

import numpy as np
import pytz
import gym
from stable_baselines.results_plotter import ts2xy
from stable_baselines.bench.monitor import load_results

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)


best_mean_reward = -np.inf
nupdates = 1

def callback(_locals, _globals):
    global nupdates
    global best_mean_reward

    if (nupdates + 1) % 10 == 0:
        x, y = ts2xy(load_results(log_dir), "timesteps")
        if len(y) > 0:
            mean_reward = np.mean(y[-10:])
            update_model = mean_reward > best_mean_reward
            if update_model:
                best_mean_reward = mean_reward
                _locals["self"].model.save("mario_model_6-3_2")
            now = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
            print(f"time: {now}, nupdates: {nupdates}, mean: {mean_reward:.2f}, best_mean: {best_mean_reward}, model_update: {update_model}")
    nupdates += 1
    return True


# Environmentのラッパー。rewardを正規化
class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0
        self._time_points = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)         

        self._cur_x = info["x_pos"]
        reward = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._cur_x, self._max_x)

        reward = reward / 10

        self._time_points += 1

        if info["life"] < 2:
            done = True     
        
        return state, reward, done, info

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        self._time_points = 0
        return self.env.reset(**kwargs)


# Environmentのラッパー。rewardを正規化
class CustomRewardAndDoneEnv2(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardAndDoneEnv2, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        reward = reward / 10

        if info["flag_get"]:
            reward += 100

        #self._cur_x = info["x_pos"]
        #reward = max(0, self._cur_x - self._max_x)
        #self._max_x = max(self._cur_x, self._max_x)

        if info["life"] < 2:
            done = True
        
        return state, reward, done, info