import numpy as np
from param import *

class RewardCalculator(object):
    def __init__(self):
        self.jobs = set()
        self.prev_time = 0

    def get_reward(self, jobs, curr_time):
        reward = 0

        # add new job into the store of jobs
        # for job in jobs:
        #     self.jobs.add(job)

        # now for all jobs (may have completed)
        # compute the elapsed time
        if False:
            for job in list(self.jobs):
                if job.arrived:
                    reward -= (min(job.completion_time, curr_time) - max(job.start_time, self.prev_time))
                    # / \
                    # args.reward_scale 

                # if the job is done, remove it from the list
                if job.completed:
                    self.jobs.remove(job)

        else:
            reward -= (curr_time - self.prev_time) 
        
        for job in jobs:
            if curr_time - job.prsent_it_start_time  > 100:
                for i in range(int((curr_time - job.prsent_it_start_time)/100)):
                    reward -= (curr_time - self.prev_time)/10


        self.prev_time = curr_time

        return reward

    def reset(self):
        self.jobs.clear()
        self.prev_time = 0
