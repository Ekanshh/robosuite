from typing import Callable, List, Optional, Tuple, Union
import os
import gym
import numpy as np
import random

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

import torch
import robosuite as suite
from robosuite.wrappers import GymWrapper


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1, mean_eps=100):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model_callback')
        self.best_mean_reward = -np.inf
        self.mean_eps = mean_eps
        self.temp = {"Max Reward": [], "Episode": []}

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        assert self.check_freq >= self.mean_eps, "Check freq needs to be larger than mean_eps"

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print('----CallBack----')
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-self.mean_eps:])
                if self.verbose > 0:
                    print(f'Evaluating model at episode: {self.num_timesteps}')
                    print(f"Current mean reward over last {self.mean_eps} episodes is: {mean_reward}")
                    print(f"Previous best mean reward was {self.best_mean_reward}")
                # New best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    self.temp['Max Reward'].append(self.best_mean_reward)
                    self.temp['Episode'].append(self.num_timesteps)

                    path = os.path.join(self.save_path, f'best_model_{self.num_timesteps}')
                    print("Saving new best model to {}".format(path))
                    path2 = os.path.join(self.save_path, 'callback_best_runs')
                    # save every best episode and reward to csv file
                    dict_csv(path2, self.temp)
                    # save every model when it is best for the current point of simulation
                    self.model.save(path)
                    # save one final best model
                    self.model.save(self.save_path)

        return True


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
        assert env.num_envs == 1, "You must pass only one environment when using this function"
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


def dict_csv(name, dict):
    '''
    Function saving dictionary as csv file:
        Parameters:
            name (str): name of the created csv file
            dict (dict): dictionary to be saved
    '''
    file_name = name + '.csv'
    with open(file_name, 'w') as f:
        for key in dict.keys():
            f.write("%s,%s\n" % (key, dict[key]))
    return

def seed_initializer():
    '''
    Function used to define seed as well as to save it in csv file:
        Returns:
            seed:
    '''
    seed = random.randint(0, 1000)
    print('Seed used', seed)
    return seed

def model_info_collect(model):

    init_weights = model.get_parameters()
    dict_csv(name=os.path.join(log_dir_extras, 'init_weights'), dict=init_weights)
    learn_info = {}
    learn_info['Num of envs'] = 1
    learn_info['Initialization seed of NN'] = model.seed
    learn_info['N_steps'] = model.n_steps
    learn_info['Training steps'] = learning_steps
    learn_info['Policy_kwargs'] = model.policy_kwargs
    learn_info['Policy'] = model.policy

    dict_csv(name=os.path.join(log_dir_extras, 'model'), dict=learn_info)
    return

if __name__ == "__main__":
    # Create log dir
    log_dir = 'robosuite/'
    log_dir_extras = os.path.join(log_dir, 'extras')
    log_dir_callback = os.path.join(log_dir, 'callback')
    os.makedirs(log_dir_callback, exist_ok=True)
    os.makedirs(log_dir_extras, exist_ok=True)

    control_param = dict(type='IMPEDANCE_POSE_Partial', input_max=1, input_min=-1,
                         output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                         output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], kp=700, damping_ratio=np.sqrt(2),
                         impedance_mode='fixed', kp_limits=[0, 100000], damping_ratio_limits=[0, 10],
                         position_limits=None, orientation_limits=None, uncouple_pos_ori=True, control_delta=True,
                         interpolation=None, ramp_ratio=0.2, control_dim=26, plotter=False, ori_method='rotation',
                         show_params=False)

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "PegInHoleSmall",
            robots="UR5e",  # use UR5e robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=False,  # Make sure we can render to the screen
            reward_shaping=True,
            ignore_done=False,
            horizon=500,
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
            controller_configs=control_param,
            r_reach_value=0.2,
            tanh_value=20.0,
            error_type='ring',
            control_spec=26,
            dist_error=0.0008
        )
    )
    eval_steps = 50
    learning_steps = 10_000
    # seed = 4
    seed = seed_initializer()
    mode = 'new_train'
    # mode = 'eval'
    # mode = 'continue_train'

    env = Monitor(env, log_dir_callback, allow_early_resets=True)
    # Create the callback: check every check_freq steps
    reward_callback = SaveOnBestTrainingRewardCallback(mean_eps=100, check_freq=200, log_dir=log_dir_callback)

    if mode == 'new_train':
        print('Training New Model')

        # new training
        policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU, net_arch=[32, 32])
        model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs,
                    tensorboard_log="./learning_log/ppo_tensorboard/", n_steps=10, seed=seed)

        model_info_collect(model=model)

        model.learn(total_timesteps=learning_steps, tb_log_name="learning", callback=reward_callback)
        print("------------ Done Training -------------")
        model.save('Benchmark_after_changes')

    if mode == 'eval':
        print('Evaluating Model')
        # evaluation
        model = PPO.load("./daniel_sim_results/daniel_original_benchmark/Daniel_n5_banchmark_single.zip", verbose=1,
                         env=env)

    if mode == 'continue_train':
        print('Training Continuation')
        model = PPO.load("./daniel_sim_results/daniel_original_benchmark/Daniel_n5_banchmark_single.zip",
                         tensorboard_log="./learning_log/ppo_tensorboard/", verbose=1, env=env)
        model.set_env(env)
        model.learn(total_timesteps=learning_steps, tb_log_name="learning", callback=reward_callback, reset_num_timesteps=False)
        print("------------ Done Retraining -------------")

    # evaluation
    mean_reward, std_reward, episode_success = evaluate(model, env, n_eval_episodes=eval_steps, render=False)
    print(
        f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f} \nsuccess rate: {episode_success / eval_steps * 100:.1f}")
