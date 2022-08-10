#!/usr/bin/env python3
import random
import numpy as np
import argparse
from DRL.evaluator import Evaluator
from . utils.util import *
from . utils.tensorboard import TensorBoard
import time
import config

exp = os.path.abspath('.').split('/')[-1]
writer = TensorBoard('../train_log/{}'.format(exp))
os.system('ln -sf ../train_log/{} ./log'.format(exp))
os.system('mkdir ./model')

def train(agent, env, evaluate):
    train_times = config.train['train_times']
    env_batch = config.train['env_batch']
    validate_interval = config.train['validate_interval']
    max_step = config.train['max_step']
    debug = config.train['debug']
    episode_train_times = config.train['episode_train_times']
    resume = config.train['resume']
    output = config.train['output']
    time_stamp = time.time()
    step = episode = episode_steps = 0
    tot_reward = 0.
    observation = None
    noise_factor = config.train['noise_factor']
    while step <= train_times:
        step += 1
        episode_steps += 1
        # reset if it is the start of episode
        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)    
        action = agent.select_action(observation, noise_factor=noise_factor)
        observation, reward, done, _ = env.step(action)
        agent.observe(reward, observation, done, step)
        if (episode_steps >= max_step and max_step):
            if step > config.train['warmup']:
                # [optional] evaluate
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    reward, dist = evaluate(env, agent.select_action, debug=debug)
                    if debug: prRed('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'.format(step - 1, np.mean(reward), np.mean(dist), np.var(dist)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                    writer.add_scalar('validate/var_dist', np.var(dist), step)
                    agent.save_model(output)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.
            tot_value_loss = 0.
            if step > config.train['warmup']:
                if step < 10000 * max_step:
                    lr = (3e-4, 1e-3)
                elif step < 20000 * max_step:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)
                for i in range(episode_train_times):
                    Q, value_loss = agent.update_policy(lr)
                    tot_Q += Q.data.cpu().numpy()
                    tot_value_loss += value_loss.data.cpu().numpy()
                writer.add_scalar('train/critic_lr', lr[0], step)
                writer.add_scalar('train/actor_lr', lr[1], step)
                writer.add_scalar('train/Q', tot_Q / episode_train_times, step)
                writer.add_scalar('train/critic_loss', tot_value_loss / episode_train_times, step)
            if debug: prBlack('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}' \
                .format(episode, step, train_time_interval, time.time()-time_stamp)) 
            time_stamp = time.time()
            # reset
            observation = None
            episode_steps = 0
            episode += 1
    
    
config.path['output'] = get_output_folder(config.path['output'], "Sculpt")
np.random.seed(config.train['seed'])
torch.manual_seed(config.train['seed'])
if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.train['seed'])
random.seed(config.train['seed'])
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
from DRL.ddpg import DDPG
from DRL.multi import fastenv
fenv = fastenv(config.train['max_step'], config.train['env_batch'], writer)
agent = DDPG(config.train['batch_size'], config.train['env_batch'], config.train['max_step'], \
                config.train['tau'], config.train['discount'], config.train['rmsize'], \
                writer, config.train['resume'], config.train['output'])
evaluate = Evaluator(args, writer)
print('observation_space', fenv.observation_space, 'action_space', fenv.action_space)
train(agent, fenv, evaluate)
