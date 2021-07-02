import or_gym
from or_gym.utils import create_env
import ray
from ray.rllib.agents import ppo
from ray import tune
import numpy as np
import matplotlib.pyplot as plt
import psutil
import os

# To save results
import shutil

CHECKPOINT_ROOT = "C:/Users/i.quadras.costa/Desktop/proj/or-gym/res/jj"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = "C:/Users/i.quadras.costa/ray_results"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)




env = or_gym.make('InvManagement-v2')

print('Env created')


def register_env(env_name, env_config={}):
    env = create_env(env_name)
    tune.register_env(env_name,
        lambda env_name: env(env_name,
            env_config=env_config))


# Environment and RL Configuration Settings
print('Environment and RL Configuration Settings...')
env_name = 'InvManagement-v2'
env_config = {}  # Change environment parameters here
rl_config = dict(
    env=env_name,
    num_workers=2,
    env_config=env_config,
    model=dict(
        vf_share_layers=False,
        fcnet_activation='elu',
        fcnet_hiddens=[256, 256]
    ),
    lr=1e-5
)

# Register environment
print('Registering environment...')
register_env(env_name, env_config)
print('Environment registered')

# Initialize Ray and Build Agent
print('Initializing Ray...')
ray.init()
print('Ray initialized')
print('Building agent...')
agent = ppo.PPOTrainer(env=env_name, config=rl_config)
print('Agent built')
results = []
#it was 500
for i in range(500):
    res = agent.train()
    results.append(res)
    #agent.save(CHECKPOINT_ROOT)
    if (i + 1) % 5 == 0:
        print('\rIter: {}\tReward: {:.2f}'.format(
            i + 1, res['episode_reward_mean']), end='')
ray.shutdown()


from matplotlib import gridspec

print('Printing results...')
# Unpack values from each iteration
rewards = np.hstack([i['hist_stats']['episode_reward']
                     for i in results])
pol_loss = [
    i['info']['learner']['default_policy']['policy_loss']
    for i in results]
vf_loss = [
    i['info']['learner']['default_policy']['vf_loss']
    for i in results]

p = 100
mean_rewards = np.array([np.mean(rewards[i - p:i + 1])
                         if i >= p else np.mean(rewards[:i + 1])
                         for i, _ in enumerate(rewards)])
std_rewards = np.array([np.std(rewards[i - p:i + 1])
                        if i >= p else np.std(rewards[:i + 1])
                        for i, _ in enumerate(rewards)])

print('GONNA PRINT')
fig = plt.figure(constrained_layout=True, figsize=(20, 10))
gs = fig.add_gridspec(2, 4)
ax0 = fig.add_subplot(gs[:, :-2])
ax0.fill_between(np.arange(len(mean_rewards)),
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 label='Standard Deviation', alpha=0.3)
ax0.plot(mean_rewards, label='Mean Rewards')
ax0.set_ylabel('Rewards')
ax0.set_xlabel('Episode')
ax0.set_title('Training Rewards')
ax0.legend()

ax1 = fig.add_subplot(gs[0, 2:])
ax1.plot(pol_loss)
ax1.set_ylabel('Loss')
ax1.set_xlabel('Iteration')
ax1.set_title('Policy Loss')

ax2 = fig.add_subplot(gs[1, 2:])
ax2.plot(vf_loss)
ax2.set_ylabel('Loss')
ax2.set_xlabel('Iteration')
ax2.set_title('Value Function Loss')

plt.show()

