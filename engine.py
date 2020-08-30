import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from utils import CumulativeMovingAverage
import config

def train_fn(annotations, model, optimizer, device, scheduler=None):
    model.train()
    loss_avg = CumulativeMovingAverage()
    annotations = tqdm(annotations, total=len(annotations))
    annotations.set_description('train')
    for a in annotations:
        loss, new_edus = model(a.edus, True, a.raw)
        loss_avg.add(loss.item())
        annotations.set_postfix_str(f'loss={loss_avg:.4f}')
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.zero_grad()

def eval_fn(annotations, model, device):
    model.eval()
    pred_trees = []
    gold_trees = []
    with torch.no_grad():
        annotations = tqdm(annotations, total=len(annotations))
        annotations.set_description('devel')
        for a in annotations:
            pred, gold = model(a.edus, False, a.raw)
            # pred_trees.append(tree)
            # gold_trees.append(a.dis)
    return pred, gold
