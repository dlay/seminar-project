import os
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as td
from torch.utils.tensorboard import SummaryWriter

from agent import DDPGAgent
from env import Env
from eval import EvalSet, calc_hit_rate, calc_ndcg, run_eval
from noise import OUActionNoise

# Hyperparameters
BUFFER_SIZE = 100000  # replay buffer size
BATCH_SIZE = 128  # replay buffer size N
GAMMA = 0.9  # discount factor
TAU = 0.001  # target network update rate
LR_ACTOR = 1e-4  # actor network learning rate
LR_DECAY_ACTOR = 1e-5
LR_CRITIC = 1e-3  # critic network learning rate
LR_DECAY_CRITIC = 1e-4
SIGMA = 0.2  # Sigma for Ornstein-Uhlenbeck process
THETA = 0.1 # Theta for Ornstein-Uhlenbeck process
EMBED_DIM = 8 # vector size for user and item embeddings
ACTION_DIM = 8 # vector size for action embeddings
HIDDEN_DIM = 16 # size of hidden layers in actor and critic
STATE_SIZE = 5 # last interacted items represented in a state
T = 10 # number of recommendations for each episode/user
USE_NOISE = True
PRE_TRAINED_EMBEDS = False
PER_BUFFER = False
DATA_DIR = 'ml-1m'
LOG_DIR = 'logs'
SEED = 1337

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


#preprocess data
print("Start preprocessing data...")
data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'), 
                       sep='::', header=None, names=['user', 'item', 'rating', 'timestamp'], 
                       usecols=[0, 1, 2, 3], dtype={0: np.int32, 1: np.int32, 2: np.int8, 3: np.int32}, engine='python')
data = data[data['rating'] > 3]
user_count = data['user'].max() + 1
item_count = data['item'].max() + 1
train_data = data.sample(frac=0.8, random_state=SEED)
test_data = data.drop(train_data.index)
train_dict = {k:train_data[train_data['user'] == k]['item'].values.tolist() for k in set(train_data['user'])}
test_dict = {k:test_data[test_data['user'] == k]['item'].values.tolist() for k in set(test_data['user'])}
valid_train_users = np.array([k for k in set(train_data['user']) if len(train_dict[k]) >= 20])
np.random.shuffle(valid_train_users)

agent = DDPGAgent(EMBED_DIM, ACTION_DIM, HIDDEN_DIM, user_count, item_count,
                  STATE_SIZE, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU,
                  LR_ACTOR, LR_CRITIC, LR_DECAY_ACTOR, LR_DECAY_CRITIC,
                  USE_NOISE, SIGMA, THETA, PRE_TRAINED_EMBEDS, PER_BUFFER)
env = Env(train_dict, item_count, STATE_SIZE)

# evaluation during training on a single fixed user
test_env = Env(test_dict, item_count, STATE_SIZE)
test_user = test_data.iloc[-1]['user']
test_items = test_env.reset(test_user)
test_set = EvalSet([test_user], test_dict, item_count, STATE_SIZE)
test_loader = td.DataLoader(test_set, batch_size=100, shuffle=False)

# evaluation after training over whole eval set
# eval_set = EvalSet(test_users, user_items_dict, item_count, STATE_SIZE)
# eval_loader = td.DataLoader(eval_set, batch_size=100, shuffle=False)

print("preprocessing done.")

writer = SummaryWriter(log_dir=LOG_DIR)

step = 0
# Training loop
for episode, user in enumerate(valid_train_users):
    # Reset the environment for each episode
    items = env.reset(user)
    
    # Initialize stochastic process
    agent.noise_reset()

    episode_reward = []
    actor_losses = []
    critic_losses = []

    for t in range(T):
        # Choose action according to current policy + OUNoise
        action, action_emb = agent.get_action(user, items, env.avail_items)

        # Perform an action in the environment and observe the next state, reward, and done flag
        items_, reward, done = env.step(action)

        # Store the experience in the replay buffer
        agent.remember(user, items, action_emb, reward, items_, done)

        # Train the agent
        if agent.replay_buffer.mem_cntr >= agent.batch_size:
            actor_loss, critic_loss = agent.train(writer, step)
            actor_losses.append(actor_loss.detach().cpu().numpy().astype(float))
            critic_losses.append(critic_loss.detach().cpu().numpy().astype(float))

        # Update the current state
        items = items_

        # Accumulate the episode reward
        episode_reward.append(reward)
        step += 1

    hit_metric, ndcg_metric = run_eval(agent, test_env, test_loader)
    reward_metric = np.mean(episode_reward)
    writer.add_scalar('hit', hit_metric, episode)
    writer.add_scalar('ndcg', ndcg_metric, episode)
    writer.add_scalar('reward', reward_metric, episode)
    if (len(actor_losses) > 0):
        writer.add_scalar('actor loss', np.mean(actor_losses), episode)
        writer.add_scalar('critic loss', np.mean(critic_losses), episode)
    print("Episode:", episode, "Reward:", reward_metric, "Hit@10:", hit_metric, "nDCG@10:", ndcg_metric)


agent.save_model(os.path.join(DATA_DIR, 'model.pt'))

#eval
# print('Evaluating...')
# hit, ndcg = run_eval(agent, test_env, eval_loader)
# print(f"Evaluation Result: Hit@10: {hit}, nDCG@10: {ndcg}")