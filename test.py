from itertools import count
import random
from torch.autograd.grad_mode import F
from config_file import Config_file
import logging
from env.env import Env
from actor_agent_pytorch import PolicyGradient
import torch
import numpy as np
from torch.distributions import Bernoulli

# def env_init():
#     config_file = Config_file()
#     env = Env(config_file) 
#     env.seed(1)
#     n_states = env.observe
#     n_actions = env.action_space.n
#     return env,n_states,n_actions


if __name__ == '__main__':
        # logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    #                 level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                        filename='reward.log',
                        filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        #a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        #日志格式
                        )

    # a = torch.ones(1, 3)    
    # b = Bernoulli(a)
    # b = b.sample()
    # c = b.data.numpy().astype(int)[0]

    config_file = Config_file()
    rl_agent_flag = True
    if rl_agent_flag is False:
        while True:
            env = Env(config_file)
            env.reset()
            total_reward = 0
            while True:
                if len(env.wait_jobs) > 0:
                    # action = np.random.randint(len(env.wait_jobs))
                    job_nums = len(env.wait_jobs)
                    if False:
                        action = 10 if job_nums == 1 else 5
                    else:
                        if job_nums == 2:
                            temp = np.random.randint(1)
                            action = temp*10
                        else:
                            action = 10
                    next_state, reward, done = env.step(action, True)
                else:
                    next_state, reward, done = env.step(None, False)
                total_reward += reward
                
                if done:
                    break
            logging.debug("episode use_time: %f", -total_reward)
    
    else:
        # env, n_states, n_actions = env_init()

        env = Env(config_file)
        env.reset()
        n_states = env.observe()

        agent  = PolicyGradient(lr = 0.01)
        if True:
            agent.load_model('./my_model.pth')

        state_pool = [] # 存放每batch_size个episode的state序列
        action_pool = []
        reward_pool = [] 
        total_reward = 0

        for i_episode in range(120000):
            state = env.reset()
            ep_reward = 0
            for t in count():
                if len(env.wait_jobs) > 0:
                    action = agent.choose_action(state) # 根据当前环境state选择action
                    # next_state, reward, done = env.step([env.action_map[action], env.smartnic.bandwidth_free], True)
                    next_state, reward, done = env.step(action, True)

                    ep_reward += reward
                    agent.state_pool.append(state)
                    agent.action_pool.append(action)
                    agent.reward_pool.append(reward)
                    state = next_state
                    if done:
                        logging.debug('Episode:%d Reward:%f', i_episode, ep_reward)
                        print('Episode:',i_episode,' Reward:', ep_reward)
                        # logging.debug("episode use_time: %f", -ep_reward)
                        break
                else:
                    next_state, reward, done = env.step(None, False)
                    ep_reward += reward
                    state = env.observe()
            # if i_episode > 0 and i_episode % 5 == 0:
            #     agent.update(reward_pool,state_pool,action_pool)
            #     state_pool = [] # 每个episode的state
            #     action_pool = []
            #     reward_pool = []
            agent.update()
        
            if i_episode>1000 and i_episode%1000 == 1:
                # torch.save(agent.policy_net.state_dict(), 'my_model.pth')
                agent.save_model('./my_model.pth')
            

