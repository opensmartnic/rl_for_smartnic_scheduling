import random
import math
from utils import OrderedSet
import numpy as np


class Job:
    def __init__(self):
        self.stage_num = 3
        self.present_stage = 0
        self.iterations = 0
        self.prsent_iterations = 0
        self.prsent_it_start_time = 0
        self.bp_ratio = 50 
        self.fp_ratio = 50

        self.gpu_source = 1
        self.encrypt_source = 5

        # train_job's model size
        self.model_size = 0

        # 
        self.untransmitted_size = 0
        self.transmit_time = 0
        # job is arrived
        self.arrived = False

        self.arrival_time = -1

        # job is completed
        self.completed = False

        # job start ime
        self.start_time = None

        # job completion time
        self.completion_time = np.inf

        # job allocated bandwidth
        self.bandwidth = 0

        self.encrypt_flag = 0;


def generator_job():
    job = Job()
    job.model_size = int(random.randint(150,200))
    job.iterations = int(random.randint(100,200))
    return job


# def load_random_job():

#     jobs = OrderedSet()
#     t = 0 
#     for i in range(10):
#         job = generator_job()
#         job.arrived = True
#         jobs.add(job)
#     for i in range(90):
#         t += int(random.randint(0,19))
#         job = generator_job()
#         job.arrived = False
#         job.arrival_time = t
#         jobs.add(job)
#     return jobs

def load_random_job():
    jobs = OrderedSet()

    job1 = generator_job()
    job1.arrived = True
    job1.arrival_time = 0
    # job1.model_size = int(random.randint(200,200))
    # job1.iterations = int(random.randint(200,200))

    jobs.add(job1)

    job2 = generator_job()
    job2.arrived = True
    job2.arrival_time = int(random.randint(0,200))
    # job2.arrival_time = 0
    # job2.model_size = int(random.randint(200,200))
    # job2.iterations = int(random.randint(150,150))
    job2.encrypt_flag = 1
    jobs.add(job2)

    for i in range(8):
        job = generator_job()
        job.arrived = False
        job.encrypt_flag = int(random.randint(0,1))
        jobs.add(job)

    return jobs




