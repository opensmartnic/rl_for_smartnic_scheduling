import time
import logging
from env.job_generator import *
from env.wall_time import WallTime
from env.smartnic import Smartnic
from env.timeline import Timeline
from env.reward_calculator import RewardCalculator
from utils import *
from action_map import *
import torch

class Env():
    def __init__(self, config_file):
        self.smartnic = Smartnic(config_file)

        # global timer
        self.wall_time = WallTime()

        # uses priority queue
        self.timeline = Timeline()

        # for computing reward at each step
        self.reward_calculator = RewardCalculator()

        self.max_time = config_file.max_time

        self.communication_delay = config_file.communication_delay

        self.feature_num = 5
        
        

    def calculate_reward(self):
        return self.reward_calculator.get_reward(self.arrived_jobs, self.wall_time.curr_time)

    def observe(self):
        self.action_map = compute_act_map(self.wait_jobs) 
        obs = self.wait_jobs, self.arrived_jobs, self.smartnic.bandwidth_free, self.action_map
        return self.translate_state(obs)
    

    def translate_state(self, obs):
        wait_jobs, arrived_jobs, bandwidth_free, action_map = obs

        # job_inputs = np.zeros([len(wait_jobs), 1])
        feature_num = self.feature_num
        job_inputs = torch.zeros(2*feature_num)
        # return job_inputs
        # job_inputs = np.zeros(2*feature_num)

        job_idx = 0
        for job_idx in range(len(action_map)):
            job = action_map[job_idx]
        # for job in arrived_jobs:
            # val = job.arrival_time + job.model_size + 0.2*(job.iterations - job.prsent_iterations)
            job_inputs[job_idx*feature_num] = self.wall_time.curr_time - job.prsent_it_start_time
            job_inputs[job_idx*feature_num + 1] = job.model_size
            job_inputs[job_idx*feature_num + 2] = job.iterations - job.prsent_iterations
            job_inputs[job_idx*feature_num + 3] = 1 if job.present_stage == 2 else 0
            job_inputs[job_idx*feature_num + 4] = (job.encrypt_flag + 1)/5
            job_idx += 1
        
        if len(action_map) == 1:
            job_inputs[1*feature_num] = -1
            job_inputs[1*feature_num + 1] = -1
            job_inputs[1*feature_num + 2] = -1
            job_inputs[1*feature_num + 3] = -1
            job_inputs[1*feature_num + 4] = -1

        arrival_time_thresholds = [0,40 ,80, 120, 160,200,240,280,320,360,400]
        model_size_thresholds = [0,1, 100, 200, 300]
        iteration_thresholds = [0,1, 5, 10, 15]

        def discretize_value(value, thresholds):
            for i in range(len(thresholds)-1):
                if value == -1:
                    return -1
                if value >= thresholds[i] and value < thresholds[i+1]:
                    return i
            return len(thresholds) - 1

        for i in range(len(action_map)):
            job_inputs[i*feature_num] = discretize_value(job_inputs[i*feature_num], arrival_time_thresholds)
            job_inputs[i*feature_num + 1] = discretize_value(job_inputs[i*feature_num + 1], model_size_thresholds)
            job_inputs[i*feature_num + 2] = discretize_value(job_inputs[i*feature_num + 2], iteration_thresholds)

        return job_inputs
    def calculate_stage_time(self, job):
        if job.present_stage == 0:
            return (job.model_size / job.gpu_source)
        elif job.present_stage == 1:
            return job.encrypt_flag*(job.model_size / job.encrypt_source)
        else:
            if job.bandwidth == 0:
                return 9999999
            return (job.untransmitted_size / job.bandwidth) + self.communication_delay

    def allocation_rule(self, action):
        # job = action[0]
        # self.smartnic.bandwidth_free -= action[1]
        # job.bandwidth = action[1]

        # time = self.calculate_stage_time(job)
        # self.timeline.push(time + self.wall_time.curr_time, job)
        # self.wait_jobs.remove(job)
        
        # 直接分配给两个任务多少带宽
        # for i, job in enumerate(self.wait_jobs):
        #     # if(len(self.wait_jobs) == 1):
        #     #     job.bandwidth = 1
        #     # else:
        #     job.bandwidth = action[i].item()
        #     # job.bandwidth = 0.5
        #     job.transmit_time = self.wall_time.curr_time
        #     time = self.calculate_stage_time(job)
        #     self.timeline.push(time + self.wall_time.curr_time, job)
        #     self.smartnic.bandwidth_free = 0
        # self.wait_jobs.clear()


        # 十一个动作空间，分别为第一个作业分配的带宽数量0,0.1，0.2....1.0
        for i, job in enumerate(self.wait_jobs):
            # if(len(self.wait_jobs) == 1):
            #     job.bandwidth = 1
            # else:
            if i ==0:
                job.bandwidth = (action)*0.1
            else:
                job.bandwidth = 1 - (action)*0.1
            # job.bandwidth = 0.5
            job.transmit_time = self.wall_time.curr_time
            time = self.calculate_stage_time(job)
            self.timeline.push(time + self.wall_time.curr_time, job)
            self.smartnic.bandwidth_free -= job.bandwidth
        self.wait_jobs.clear()
            

            
    def join_new_job(self):
        for job in self.jobs:
            if job.arrived is False:
                job.arrived = True
                time = self.calculate_stage_time(job)
                job.start_time = self.wall_time.curr_time
                job.arrival_time = self.wall_time.curr_time
                self.arrived_jobs.add(job)
                self.timeline.push(time + self.wall_time.curr_time, job)  
                return   
        
    
    def get_env_state(self):
        pass

    def step(self, action, is_allocated = False):
        done = False


        # if self.smartnic.bandwidth_free == 0:
        #      # all commitments are made, now schedule free executors
        if is_allocated:
            self.allocation_rule(action)
        
        while (len(self.timeline) > 0 and self.smartnic.bandwidth_free == 0) \
            or (len(self.timeline) > 0 and len(self.wait_jobs) == 0):
            new_time, job = self.timeline.pop()
            self.wall_time.update_time(new_time)
            if job.arrived:
                if job.present_stage == 2:
                    if job.prsent_iterations == job.iterations:
                        # 最后一次迭代
                        job.completed = True
                        job.completion_time = new_time
                        # self.smartnic.bandwidth_free += job.bandwidth 
                        job.bandwidth = 0
                        self.arrived_jobs.remove(job)
                        self.finished_jobs.add(job)
                        self.join_new_job()
                        print("job run time:",self.wall_time.curr_time - job.start_time)
                        
                    else:
                        job.prsent_iterations += 1
                        job.prsent_it_start_time = self.wall_time.curr_time
                        job.bandwidth = 0
                        job.present_stage = (job.present_stage + 1)%job.stage_num
                        time = self.calculate_stage_time(job)
                        self.timeline.push(time + self.wall_time.curr_time, job)

                    # 从阶段2结束释放带宽资源，重新分配
                    # self.smartnic.bandwidth_free += job.bandwidth
                    self.smartnic.reset() 
                    for job in self.arrived_jobs:
                        if job.present_stage == 2:
                            job.untransmitted_size -= (new_time - job.transmit_time)*job.bandwidth 
                            job.bandwidth = 0
                            self.timeline.remove(job)
                            self.wait_jobs.add(job)



                elif job.present_stage == 1:
                    job.present_stage = (job.present_stage + 1)%job.stage_num
                    job.untransmitted_size = job.model_size
                    self.wait_jobs.add(job)

                    self.smartnic.reset() 
                    for job in self.arrived_jobs:
                        if job.present_stage == 2:
                            job.untransmitted_size -= (new_time - job.transmit_time)*job.bandwidth 
                            job.bandwidth = 0
                            self.timeline.remove(job)
                            self.wait_jobs.add(job)
                    
                elif job.present_stage == 0:
                    job.present_stage = (job.present_stage + 1)%job.stage_num
                    time = self.calculate_stage_time(job)
                    self.timeline.push(time + self.wall_time.curr_time, job)

            else:
                job.arrived = True
                job.present_stage = 0
                job.start_time = self.wall_time.curr_time
                time = self.calculate_stage_time(job)
                self.arrived_jobs.add(job)
                self.timeline.push(time + self.wall_time.curr_time, job)


        done = (len(self.timeline) == 0 and len(self.wait_jobs) == 0) or \
               (self.wall_time.curr_time >= self.max_time)
        # if len(self.timeline) == 0:
        #     a = 1

        next_state = self.observe()

        reward = self.calculate_reward()

        return next_state, reward, done
    
    

    def reset(self):

        self.wall_time.reset()
        self.timeline.reset()
        self.reward_calculator.reset()
        self.smartnic.reset()
        self.finished_jobs = OrderedSet()
        self.jobs = OrderedSet()
        self.wait_jobs = OrderedSet()
        self.arrived_jobs = OrderedSet()
        self.jobs = load_random_job()
        for job in self.jobs:
            if job.arrived:
                time = self.calculate_stage_time(job)
                job.start_time = self.wall_time.curr_time
                self.arrived_jobs.add(job)
                self.timeline.push(time + job.arrival_time, job)
            # else:
            #     time = job.arrival_time
            
        return self.observe()

        



