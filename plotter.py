from tensorboardX import SummaryWriter
import torch

class TensorboardPlotter(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir) # log가 저장될 경로

    def loss_plot(self, tag, type, scalar_value, global_step):
        self.writer.add_scalar(tag + '/' + type, torch.tensor(scalar_value), global_step)

    def overlap_plot(self, tag, tag_scalar_dict, global_step):
        self.writer.add_scalars(tag, tag_scalar_dict, global_step)
    
    def img_plot(self, title, img, global_step):
        self.writer.add_images(title, img, global_step)

    def text_plot(self, tag, text, global_step):
        self.writer.add_text(tag, text, global_step)

    def close(self):
        self.writer.close()