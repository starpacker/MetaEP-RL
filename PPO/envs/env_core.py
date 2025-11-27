import numpy as np
import os
import random
import copy
from envs import package as pkg
import time,sys
import importlib.util
import pickle
import subprocess

flag_env = 6
if flag_env == 0:
    from envs.collect_data_2 import get_reward
    pra_txt = "C:\\data\\pra_all.txt"
    reward_txt = "C:\\data\\reward.txt"
elif flag_env == 1:
    from envs.collect_data import get_reward
    pra_txt = "C:\\data\\pra_all_2.txt"
    reward_txt = "C:\\data\\reward_2.txt"
elif flag_env == 6:
    from envs.multi_collect_data_2 import get_reward
    pra_txt = "C:\\data\\pra_all_6.txt"
    reward_txt = "C:\\data\\reward_6.txt"





def boundary_test(pra):
    c = pra[0]>=30 and pra[0]<=160 and pra[2]>=40 and pra[2]<=80 \
    and pra[3]>=80 and pra[3]<=340 and pra[0]+pra[3]<=370 \
    and pra[6]>=30 and pra[6]<=150 and pra[7]>=40 and pra[7]<=340 \
    and pra[6]+pra[7]<=340 and pra[9]>=30 and pra[9]<=100 \
    and pra[8]>=40 and pra[8] <=100 and pra[5]>=40 and pra[5]<=100 \
    and pra[4]>=40 and pra[4]<=100 and pra[1]>=40 and pra[1]<=100 \
    and pra[9]+pra[8]+pra[5]+pra[4]+pra[1]<=370
    return c

def normalize(obs1):
    obs = copy.copy(obs1)
    obs[0] = (obs[0]-30)/130.0   
    obs[1] = (obs[1]-40)/60.0   
    obs[2] = (obs[2]-40)/40.0   
    obs[3] = (obs[3]-80)/260.0  
    obs[4] = (obs[4]-40)/60.0   
    obs[5] = (obs[5]-40)/60.0  
    obs[6] = (obs[6]-30)/120.0 
    obs[7] = (obs[7]-40)/300.0
    obs[8] = (obs[8]-40)/60.0
    obs[9] = (obs[9]-30)/70.0   # new
    return obs

def round_to_nearest_multiple_of_four(value):
    # 将值四舍五入到最近的4的整倍数
    return np.round(value / 4) * 4

def denormalize_and_round(obs):
    # 假设 obs 是归一化后的观测值数组
    denormalized_obs = copy.copy(obs)  # 创建副本以避免修改原始数据
    # print("obs",obs)
    # 应用反归一化公式
    denormalized_obs[0] = obs[0] * 130.0 + 30
    denormalized_obs[1] = obs[1] * 60.0 + 40
    denormalized_obs[2] = obs[2] * 40.0 + 40
    denormalized_obs[3] = obs[3] * 260.0 + 80
    denormalized_obs[4] = obs[4] * 60.0 + 40
    denormalized_obs[5] = obs[5] * 60.0 + 40
    denormalized_obs[6] = obs[6] * 120.0 + 30
    denormalized_obs[7] = obs[7] * 300.0 + 40  # + 30
    denormalized_obs[8] = obs[8] * 60.0 + 40
    denormalized_obs[9] = obs[9] * 70.0 + 30   # new

    # 将每个参数四舍五入到4的整倍数
    for i in range(len(denormalized_obs)):
        denormalized_obs[i] = round_to_nearest_multiple_of_four(denormalized_obs[i])

    return denormalized_obs
def random_start():
    pra = [0,0,0,0,0,0,0,0,0,0]
    while True:
        # print(1)
        pra[0] = random.randint(8,40)*4    #[32,160]
        pra[1] = random.randint(10,25)*4   #[40,100]
        pra[2] = random.randint(10,20)*4   #[40,80]
        # pra[3] = random.randint(20,(340-pra[2])//4)*4  
        pra[3] = random.randint(20,(370-pra[0])//4)*4
        pra[4] = random.randint(10,25)*4
        pra[5] = random.randint(10,25)*4
        pra[6] = random.randint(8,37)*4
        pra[7] = random.randint(10,(340-pra[6])//4)*4
        pra[8] = random.randint(10,25)*4
        pra[9] = random.randint(8,37)*4
        if boundary_test(pra):
            return pra
        
def random_start_part():
    pra = [0,0,0,0]
    new_pra = [112,48,52,152,48,48,112,92,80,76]
    while True:
        # print(1)
        pra[0] = random.randint(8,40)*4    #[32,160]
        pra[1] = random.randint(10,25)*4   #[40,100]
        pra[2] = random.randint(10,20)*4   #[40,80]
        # pra[3] = random.randint(20,(340-pra[2])//4)*4  
        pra[3] = random.randint(20,(370-pra[0])//4)*4
        new_pra[0] = pra[0]
        new_pra[1] = pra[1]
        new_pra[2] = pra[2]
        new_pra[3] = pra[3]

        if boundary_test(new_pra):
            return pra



def round_to_nearest_multiple_of_four(value):
    # 将值四舍五入到最近的4的整倍数
    return np.round(value / 4) * 4

def denormalize_and_round_part(action):
    # 假设 action 是 shape=(9,) 的 numpy array
    denormalized_action = np.empty_like(action, dtype=np.float32)

    denormalized_action[0] = action[0] * 130.0 + 30
    denormalized_action[1] = action[1] * 60.0 + 40
    denormalized_action[2] = action[2] * 40.0 + 40
    denormalized_action[3] = action[3] * 260.0 + 80
    denormalized_action[4] = action[4] * 60.0 + 40
    denormalized_action[5] = action[5] * 60.0 + 40
    denormalized_action[6] = action[6] * 120.0 + 30
    denormalized_action[7] = action[7] * 300.0 + 40
    denormalized_action[8] = action[8] * 60.0 + 40

    # 四舍五入到 4 的整倍数
    denormalized_action = round_to_nearest_multiple_of_four(denormalized_action)

    # 在末尾添加固定值 76
    denormalized_action = np.append(denormalized_action, 76.0)

    return denormalized_action

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        # self.agent_num = 1
        self.obs_dim = 10  # 最后一个维度固定
        # self.action_dim = 10
        # self.obs_dim = 4  # 最后一个维度固定
        self.action_dim = 9
        self.agent_num = 9
        # self.agent_num = 10  # 设置智能体的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
        # self.obs_dim = 10  # 设置智能体的观测维度 # set the observation dimension of agents
        # self.action_dim = 78  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional
    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """

        sub_obs = random_start()
        # sub_obs = random_start_part()

        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_agent_obs.append(normalize(sub_obs))
        self.last_obs = sub_obs
        self.dones = [False for i in range(self.agent_num)]
        return sub_agent_obs
    def read_data(self):
        with open('my_class.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data
    def to_pra(self,actions):
        actions = np.argmax(actions,axis=1)
        pras = []
        for action in actions:
            b = (action+8)*4
            pras.append(b)
            
        return np.array(pras)
    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2维的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        # print(actions)
        # if self.agent_num == 1:
        actions = actions[0]

        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        pras = []

        # pras_int = np.array(denormalize_and_round(actions)).astype(int).T.squeeze()
        pras_int = np.array(denormalize_and_round_part(actions)).astype(int).T.squeeze()

        self.last_obs = pras_int
        
        if not boundary_test(pras_int):
            pras_int = np.zeros(10)
            np.savetxt(pra_txt,pras_int)

            get_reward()
            reward = np.loadtxt(reward_txt)
            for i in range(self.agent_num):

                # sub_agent_obs.append(pras_int)
                sub_agent_obs.append(self.last_obs)

                sub_agent_reward.append(reward)
                sub_agent_done.append(False)
                sub_agent_info.append({})
            return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
        
        np.savetxt(pra_txt,pras_int)
        # print(pras_int)

        get_reward()
        reward = np.loadtxt(reward_txt)
        # reward = 0.0

        for i in range(self.agent_num):

            # sub_agent_obs.append(pras_int)
            sub_agent_obs.append(self.last_obs)

            sub_agent_reward.append(reward)
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
    
