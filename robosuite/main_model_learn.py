"""
This script shows how to adapt an environment to be compatible
with the OpenAI Gym-style API. This is useful when using
learning pipelines that require supporting these APIs.

For instance, this can be used with OpenAI Baselines
(https://github.com/openai/baselines) to train agents
with RL.


We base this script off of some code snippets found
in the "Getting Started with Gym" section of the OpenAI 
gym documentation.

The following snippet was used to demo basic functionality.

    import gym
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

To adapt our APIs to be compatible with OpenAI Gym's style, this script
demonstrates how this can be easily achieved by using the GymWrapper.
"""
from typing import Callable, List, Optional, Tuple, Union
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, TD3
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

import torch
import robosuite as suite
from robosuite.wrappers import GymWrapper

from robosuite.wrappers import VisualizationWrapper


def make_env(env, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'medium_best_model_30_11_n5')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


class PlottingCallback(BaseCallback):
    """
    Callback for plotting the performance in realtime.

    :param verbose: (int)
    """

    def __init__(self, verbose=1):
        super(PlottingCallback, self).__init__(verbose)
        self._plot = None

    def _on_step(self) -> bool:
        # get the monitor's data
        x, y = ts2xy(load_results(log_dir), 'timesteps')

        if self._plot is None:  # make the plot
            plt.ion()
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            line, = ax.plot(x, y)
            self._plot = (line, ax, fig)
            plt.show()
        else:  # update and rescale the plot
            self._plot[0].set_data(x, y)
            self._plot[-2].relim()
            self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02,
                                     self.locals["total_timesteps"] * 1.02])
            self._plot[-2].autoscale_view(True, True, True)
            self._plot[-1].canvas.draw()


def evaluate(model: "base_class.BaseAlgorithm",
             env: Union[gym.Env, VecEnv],
             n_eval_episodes: int = 10,
             deterministic: bool = True,
             render: bool = True,
             callback: Optional[Callable] = None,
             reward_threshold: Optional[float] = None,
             return_episode_rewards: bool = False,
             ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean.
    :return: Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    global _info, obs
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"\

    actions = []
    episode_rewards, episode_lengths = [], []
    episode_success = 0
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            actions.append(action)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_success += int(_info.get('is_success'))
        print(f"episode number: {i},reward: {episode_reward}, episode lenght: {_info.get('time')} ")
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    # a = np.array(actions)
    # print(a.shape)
    # np.savetxt('action_matrix.csv', a, delimiter=',')
    # print('Saved CSV')
    return mean_reward, std_reward, episode_success


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if __name__ == "__main__":
    # Create log dir
    log_dir = "robosuite/best_models/"
    os.makedirs(log_dir, exist_ok=True)

    # Load the desired controller's default config as a dict
    # config = load_controllesshir_config(default_controller='OSC_POSE')
    control_param = dict(type='IMPEDANCE_POSE_Partial', input_max=1, input_min=-1,
                         output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                         output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], kp=700, damping_ratio=np.sqrt(2),
                         impedance_mode='fixed', kp_limits=[0, 100000], damping_ratio_limits=[0, 10],
                         position_limits=None, orientation_limits=None, uncouple_pos_ori=True, control_delta=True,
                         interpolation=None, ramp_ratio=0.2, control_dim=26, plotter=False, ori_method='rotation', show_params=False)

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "PegInHoleSmall",
            robots="UR5e",  # use UR5e robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=True,  ##True  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            ignore_done=False,
            horizon=500,
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
            controller_configs=control_param,
            r_reach_value=0.2,
            tanh_value=20.0,
            error_type='fixed',
            control_spec=26,
            dist_error=0.0008
        )
    )
    # env = DummyVecEnv([lambda: env_id])
    # env.reset()
    env = Monitor(env, log_dir, allow_early_resets=True)
    # Create the callback: check every 200 steps
    reward_callback = SaveOnBestTrainingRewardCallback(check_freq=200, log_dir=log_dir)

    policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU, net_arch=[32, 32])
    model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./learning_log/ppo_tensorboard/",
                n_steps=10, seed=4)  # ,batch_size=3, n_steps=500*10

    # model = PPO.load("./daniel_sim_results/daniel_original_benchmark/Daniel_n5_banchmark_single.zip", verbose=1)
    # model.set_env(env)
    model.learn(total_timesteps=10000, tb_log_name="learning", callback=reward_callback)#, eval_env=evaluate(model, env, n_eval_episod         es=10))
    print("Done")
    model.save('Daniel__single_noseed_20steps')
    # changed: multiply by *10 instead of 5 in env

    mean_reward, std_reward, episode_success = evaluate(model, env, n_eval_episodes=50, render=False)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f} \nsuccess rate: {episode_success / 50 * 100:.1f}")
