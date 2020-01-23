import torch
from torch.optim import Adam
from tqdm import tqdm
from model import DiscoBertModel
from rst import load_annotations
from sklearn.metrics import precision_recall_fscore_support


# load some parameters
num_epochs = 20
learning_rate = 1e-3
train_dir = ""
device = torch.device('cpu')

RST_CORPUS_PATH = 'RST/data/RSTtrees-WSJ-main-1.0/TRAINING/'


def train():
    discobert = DiscoBertModel().to(device)

    # setup the optimizer, loss, etc
    optimizer = Adam(params=discobert.parameters(), lr=learning_rate)

    # for each epoch
    for epoch_i in range(num_epochs):
        for annotation in tqdm(load_annotations(RST_CORPUS_PATH)):
            discobert.zero_grad()
            loss, pred_tree = discobert(annotation.edus, annotation.dis)
            loss.backward()
            optimizer.step()

        # save model
        model_file = f'discobert_{epoch_i}'
        torch.save(discobert, model_file)

def predict(model_file):
    discobert = torch.load(model_file)

    all_gold_nodes = []
    all_pred_nodes = []
    for annotation in tqdm(load_annotations(RST_TEST_CORPUS_PATH)):
        pred_tree = discobert(annotation.edus)
        all_gold_nodes.extend(annotation.dis.get_nonterminals())
        all_pred_nodes.extend(pred_tree.get_nonterminals())

    p, r, f1, _ = precision_recall_fscore_support(all_gold_nodes, all_pred_nodes, average='micro') # TODO confirm
    print(f'P:{p}\tR:{r}\tF1:{f1}')

