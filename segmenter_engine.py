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
        # print("annotation in train: ", a)
        loss, new_edus = model(a.edus, "train", a)
        loss_avg.add(loss.item())
        annotations.set_postfix_str(f'loss={loss_avg:.4f}')
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.zero_grad()

def eval_fn(annotations, model, device):
    model.eval()
    predictions = []
    gold_labels = []
    with torch.no_grad():
        annotations = tqdm(annotations, total=len(annotations))
        annotations.set_description('devel')
        for a in annotations:
            pred, gold = model(a.edus, "eval", a)
            predictions.extend(pred)
            gold_labels.extend(gold)
    return predictions, gold_labels

def run_fn(annotations, model, device):
    model.eval()
    new_annotations = []
    with torch.no_grad():
        annotations = tqdm(annotations, total=len(annotations))
        annotations.set_description('run')
        for a in annotations:
            # print("annotation in run: ", a)
            pred, _ = model(a.edus, "run", a)
            # print("pred: ", pred)
            new_annotations.append(pred)
    # print("new ann 1: ", new_annotations[0])
    return new_annotations