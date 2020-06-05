#!/usr/bin/env python

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report
from transformers import AdamW, get_linear_schedule_with_warmup
from model import DiscoBertModel
from rst import load_annotations, iter_spans_only
from utils import prf1
import config
import engine

def optimizer_parameters(model):
    no_decay = ['bias', 'LayerNorm']
    named_params = list(model.named_parameters())
    return [
        {'params': [p for n,p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n,p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

def eval_trees(pred_trees, gold_trees):
    all_pred_spans = [[f'{x}' for x in iter_spans_only(t.get_nonterminals())] for t in pred_trees]
    all_gold_spans = [[f'{x}' for x in iter_spans_only(t.get_nonterminals())] for t in gold_trees]
    scores = [prf1(pred, gold) for pred, gold in zip(all_pred_spans, all_gold_spans)]
    scores = np.array(scores).mean(axis=0).tolist()
    return scores

def main():
    device = torch.device('cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu')
    model = DiscoBertModel()
    model.to(device)

    train_ds = list(load_annotations(config.TRAIN_PATH))
    valid_ds = list(load_annotations(config.VALID_PATH))

    num_training_steps = int(len(train_ds) * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters(model), lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    best_f1 = 0
    for epoch in range(config.EPOCHS):
        if epoch > 0: print()
        print(f'epoch: {epoch+1}/{config.EPOCHS}')
        engine.train_fn(train_ds, model, optimizer, device, scheduler)
        pred_trees, gold_trees = engine.eval_fn(valid_ds, model, device)
        p, r, f1 = eval_trees(pred_trees, gold_trees)
        print(f'P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        if f1 > best_f1:
            model.save(config.MODEL_PATH)
            best_f1 = f1

if __name__ == '__main__':
    main()