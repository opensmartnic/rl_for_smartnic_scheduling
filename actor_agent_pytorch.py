from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
import logging
import os
import math
import numpy as np






# class PolicyNet(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(PolicyNet, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         return F.softmax(self.fc2(x), dim=-1)
    
#     def load(self, model_path):
#         if os.path.exists(model_path) :
#             logging.debug(f"loading model from {model_path}")
#             self = torch.load(model_path)

#             return True
#         else:
#             logging.debug(f"model files does not exist at {model_path}")
#             return False

#     def save(self, model_path):
#         logging.debug(f"save model to {model_path}")
#         torch.save(self, model_path)




# class PolicyGradient:
#     def __init__(self, state_dim = 10, hidden_dim = 16, action_dim = 11, lr = 0.01, gamma = 0.99,
#                  device = "cpu"):

#         self.feature_num = 5         
#         self.policy_net = PolicyNet(state_dim, hidden_dim,
#                                     action_dim).to(device)
#         self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
#                                           lr=lr)  # 使用Adam优化器
#         self.gamma = gamma  # 折扣因子
#         self.device = device
#         self.state_pool = [] # 存放每batch_size个episode的state序列
#         self.action_pool = []
#         self.reward_pool = []

#     def choose_action(self, state):  # 根据动作概率分布随机采样
#         # state = self.translate_state(state)
#         state = torch.tensor([state], dtype=torch.float).to(self.device)
#         probs = self.policy_net(state)
#         action_dist = torch.distributions.Categorical(probs)
#         action = action_dist.sample()

#         return action.item()

#     def update(self):
#         reward_list = self.reward_pool
#         state_list = self.state_pool
#         action_list = self.action_pool

#         G = 0
#         loss_all = 0
#         self.optimizer.zero_grad()
#         for i in reversed(range(len(reward_list))):  # 从最后一步算起
#             reward = reward_list[i]
#             state = torch.tensor([state_list[i]],
#                                   dtype=torch.float).to(self.device)
#             action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
#             log_prob = torch.log(self.policy_net(state).gather(1, action))
#             G = self.gamma * G + reward
#             loss = -log_prob * G  # 每一步的损失函数
#             loss_all += loss
#             loss.backward()  # 反向传播计算梯度
#         # logging.debug('average_loss:%f, all_loss:%f', loss_all/len(self.reward_pool), loss_all)
#         # print('average_loss:', loss_all/len(self.reward_pool), 'all_loss:', loss_all)
#         self.optimizer.step()  # 梯度下降















































class FCN(nn.Module):
    ''' 全连接网络'''
    
# -------------------------------------------------------------
    def __init__(self, input_size = 2*4, hidden_size = 16, output_size = 3):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x),dim=-1)
        # x = F.sigmoid(self.fc3(x))
        return x


    def load(self, model_path):
        if os.path.exists(model_path) :
            logging.debug(f"loading model from {model_path}")
            self = torch.load(model_path)

            return True
        else:
            logging.debug(f"model files does not exist at {model_path}")
            return False

    def save(self, model_path):
        logging.debug(f"save model to {model_path}")
        torch.save(self, model_path)




class PolicyGradient:
    def __init__(self, device='cpu', gamma=0.99, lr=0.01,feature_num = 5):
        self.gamma = gamma
        self.feature_num = feature_num
        self.policy_net = FCN(2*self.feature_num,32, 11)
        # self.policy_net = torch.load("my_model_11action_v2.pt")
        # self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(self.policy_net.parameters(),lr=lr)
        # self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=lr)
        self.optimizer = torch.optim.Adagrad(self.policy_net.parameters(), lr=0.02)
        
        
        self.state_pool = [] # 存放每batch_size个episode的state序列
        self.action_pool = []
        self.reward_pool = []
        self.eps = 1e-6
        self.device = device


    def load_model(self, model_path):
        if os.path.exists(model_path) :
            logging.debug(f"loading model from {model_path}")
            self.policy_net.load_state_dict(torch.load(model_path))
            return True
        else:
            logging.debug(f"model files does not exist at {model_path}")
            return False

    def save_model(self, model_path):
        logging.debug(f"save model to {model_path}")
        # torch.save(self, model_path)
        torch.save(self.policy_net.state_dict(), model_path)
        
    # 选择action，根据学到的P_{\theta}(a_t|s_t)进行采样
    def choose_action(self, state):

        # state = torch.reshape(state,(state.size(0),1))

        probs = self.policy_net(state)
        # action = probs
        action = torch.multinomial(probs, 1)

        return action.item()

        

    # 利用累积的episode路径进行gradient decent
    def update(self):
        loss_all = 0

        # Discount reward and normalize reward：根据公式6
        # 对 reward 进行累积和折扣处理
        running_add = 0
        for i in reversed(range(len(self.reward_pool))):
            if self.reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + self.reward_pool[i]
                self.reward_pool[i] = running_add
        reward_mean = np.mean(self.reward_pool)
        reward_std = np.std(self.reward_pool)

        # 对 reward 进行标准化处理，将 reward 值缩放到一个合适的范围，使得 policy gradient 的优化效果更好。
        for i in range(len(self.reward_pool)):
            self.reward_pool[i] = (self.reward_pool[i] - reward_mean) / reward_std
        
        # 计算 loss 并进行反向传播
        self.optimizer.zero_grad()
        for i in range(len(self.reward_pool)):
            state = self.state_pool[i]
            # action = torch.FloatTensor([self.action_pool[i]])
            action = self.action_pool[i]
            reward = self.reward_pool[i]

            
            # m = Bernoulli(probs)
            # 公式6的实现：本质上是个分类问题,负号是为了将梯度上升转为梯度下降
            # loss = -m.log_prob(action) * reward  # Negtive score function x reward
            # print(loss)
            # selected_job_prob = torch.sum(torch.multiply(probs,action), -1)
            
            probs = self.policy_net(state)
            selected_job_prob = probs[action]
            # log_prob = torch.log(selected_job_prob)
            loss = -torch.sum(torch.log(selected_job_prob + self.eps) * reward)
            loss_all += loss.item()
            loss.backward()

        logging.debug('average_loss:%f, all_loss:%f', loss_all/len(self.reward_pool), loss_all)
        print('average_loss:', loss_all/len(self.reward_pool), 'all_loss:', loss_all)
        # print('all_loss:',  loss_all)
        self.optimizer.step()


        self.state_pool = [] # 每个episode的state
        self.action_pool = []
        self.reward_pool = []
        
