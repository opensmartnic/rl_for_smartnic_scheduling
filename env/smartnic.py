


class Smartnic():
    def __init__(self, config_file):
        self.bandwidth = config_file.bandwidth
        self.bandwidth_free = config_file.bandwidth

    def reset(self):
        self.bandwidth_free =self.bandwidth