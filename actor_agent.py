import numpy as np
import bisect
from utils import *


class ActorAgent():
    def __init__(self):
        pass

    def actor_network(self):
        pass


    def apply_gradients(self, gradients, lr_rate):
        pass

    def define_params_op(self):
        # define operations for setting network parameters
        return None

    def gcn_forward(self, node_inputs, summ_mats):
        return None

    def get_params(self):
        return self.sess.run(self.params)

    def save_model(self, file_path):
        self.saver.save(self.sess, file_path)

    def get_gradients(self):

        return None

    def predict(self):
        return None

    def set_params(self, input_params):
        pass

    def translate_state(self, obs):
        """
        Translate the observation to matrix form
        """
        pass

    def get_valid_masks(self):
        pass

    def invoke_model(self, obs):
        pass

    def get_action(self, obs):

       pass
