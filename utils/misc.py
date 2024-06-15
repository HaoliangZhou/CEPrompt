import torch
import numpy as np
import random
import os
import codecs
import logging
import builtins
import datetime
import torch.distributed as dist



class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def compute_ACC(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_log(args, record_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(record_path, 'recording.log')

    fh = logging.FileHandler(log_path, mode='w') 
    fh.setLevel(logging.DEBUG)  
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def convert_models_to_half(model):
    for p in model.parameters():
        p.data = p.data.half()
        if p.grad:
            p.grad.data = p.grad.data.half()


def write_description_to_folder(file_name, config):
    with codecs.open(file_name, 'w') as desc_f:
        desc_f.write("- Training Parameters: \n")
        for key, value in config.__dict__.items():
            desc_f.write("  - {}: {}\n".format(key, value))
