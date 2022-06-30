import torch
import os


class Checkpoint:
    def __init__(self, root_dir='./checkpoints', checkpoint_period=2):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        self.root_dir = root_dir
        self.checkpoint_period = checkpoint_period

    def save_model(self, state_dict, epoch_cnt):
        if epoch_cnt % self.checkpoint_period == 0:
            torch.save(state_dict, os.path.join(self.root_dir, 'model-{}.pth'.format(epoch_cnt)))

    def load_model(self, model_name):
        return torch.load(os.path.join(self.root_dir, model_name))