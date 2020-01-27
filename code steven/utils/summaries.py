import os
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from utils.functions import decode_seg_map_sequence
import cv2
import torch

class TensorboardSummary(object):
    def __init__(self, directory, args):
        self.directory = directory
        self.args = args

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, image, target, output, global_step):

        grid_image = make_grid(image[:3, (0, 1, 2)].clone().cpu().data, 3, normalize=True)
        #grid_image = make_grid(image[:3, :].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)

        grid_image = make_grid(decode_seg_map_sequence(target[:3], self.args), 3, normalize=True)
        writer.add_image('target', grid_image, global_step)

        grid_image = make_grid(decode_seg_map_sequence(output[:3], self.args), 3, normalize=True)
        writer.add_image('output', grid_image, global_step)
