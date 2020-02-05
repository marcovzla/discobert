import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from model import DiscoBertModel
from rst import load_annotations, iter_spans_only


# load some parameters
num_epochs = 20
learning_rate = 1e-3
train_dir = ""
device = torch.device('cpu')

SAVED_MODELS_PATH = '/Users/bsharp/data/discobert/RST/models/debug'
RST_CORPUS_PATH = '/Users/bsharp/data/discobert/RST/data/RSTtrees-WSJ-main-1.0/TRAINING/training_subset'
RST_VAL_CORPUS_PATH = '/Users/bsharp/data/discobert/RST/data/RSTtrees-WSJ-main-1.0/TRAINING/validation'


# TODO: replace?? tokenizer based on what Enrique said OR pass all EDUs at once
# TODO: finish the validation code -- ability to eval just tree, tree + direction, tree + dir + label
# TODO: add the classifiers for label and direction


def train():
    discobert = DiscoBertModel().to(device)

    # setup the optimizer, loss, etc
    optimizer = Adam(params=discobert.parameters(), lr=learning_rate)

    # for each epoch
    for epoch_i in range(num_epochs):
        for annotation in tqdm(list(load_annotations(RST_CORPUS_PATH))):
            discobert.zero_grad()
            loss, pred_tree = discobert(annotation.edus, annotation.dis)
            loss.backward()
            optimizer.step()

        # save model
        model_dir = os.path.join(SAVED_MODELS_PATH, f'discobert_{epoch_i}')
        discobert.save_pretrained(model_dir)
    return model_dir

def predict(model_dir):
    discobert = DiscoBertModel.from_pretrained(model_dir)

    all_gold_nodes = []
    all_pred_nodes = []
    for annotation in tqdm(load_annotations(RST_VAL_CORPUS_PATH)):
        pred_tree = discobert(annotation.edus)
        all_gold_nodes.extend(annotation.dis.get_nonterminals())
        all_pred_nodes.extend(pred_tree.get_nonterminals())

        all_gold_spans = list(iter_spans_only(all_gold_nodes))
        all_pred_spans = list(iter_spans_only(all_pred_nodes))

    p, r, f1 = eval(all_gold_spans, all_pred_spans) # TODO confirm
    print(f'P:{p}\tR:{r}\tF1:{f1}')

def eval(gold, pred):
    TP, FP, FN = 0, 0, 0
    for g in gold:
        if g in pred:
            TP += 1
        else:
            FN += 1

    for p in pred:
        if p not in gold:
            FP += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return precision, recall, f1


if __name__=='__main__':

    last_model = train()
    predict(last_model)