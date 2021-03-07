import torch
from torch import nn
from transformers import *
from rst import TreeNode
from transition_system import TransitionSystem
from treelstm import TreeLstm
# from bilstm import BiLSTM
import config
import torch.nn.functional as F
from utils import make_embedding_matrix, make_index2word, load_glove
import numpy as np
import re
inf = float('inf')

def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

class DiscoBertModelGlove2ClassStackOnly(nn.Module):
    
    def __init__(self, word2index):
        super().__init__()
        # config
        self.dropout = config.DROPOUT
        self.bert_path = config.BERT_PATH
        self.id_to_action = config.ID_TO_ACTION
        self.action_to_id = config.ACTION_TO_ID
        self.id_to_direction = config.ID_TO_DIRECTION
        self.direction_to_id = config.DIRECTION_TO_ID
        self.id_to_label = config.ID_TO_LABEL
        self.label_to_id = config.LABEL_TO_ID
        self.hidden_size = config.HIDDEN_SIZE
        self.relation_label_hidden_size = config.RELATION_LABEL_HIDDEN_SIZE
        self.direction_hidden_size = config.DIRECTION_HIDDEN_SIZE
        self.include_relation_embedding = config.INCLUDE_RELATION_EMBEDDING
        self.include_direction_embedding = config.INCLUDE_DIRECTION_EMBEDDING
        # init model
        self.tokenizer = config.TOKENIZER

        self.glove = load_glove(config.GLOVE_PATH)
        self.word2index = word2index
        self.index2word = make_index2word(word2index)
        self.embedding_matrix = make_embedding_matrix(self.index2word, self.glove)
        self.word_embedding = nn.Embedding(
                num_embeddings=len(self.word2index.keys()),
                embedding_dim=config.EMBEDDING_SIZE
            )
        self.word_embedding.load_state_dict({'weight': self.embedding_matrix})

        self.bilstm = nn.LSTM(input_size=config.EMBEDDING_SIZE, hidden_size=100, num_layers=3, batch_first=True, bidirectional=True) 
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.bert_drop = nn.Dropout(self.dropout)
        # self.project = nn.Linear(self.encoder.config.hidden_size, self.hidden_size)
        self.missing_node = nn.Parameter(torch.rand(self.hidden_size, dtype=torch.float))
        self.action_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_action))
        self.label_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_label))
        # self.merge_layer = nn.Linear(2 * self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.treelstm = TreeLstm(self.hidden_size // 2, self.include_relation_embedding, self.include_direction_embedding, self.relation_label_hidden_size, self.direction_hidden_size)
        self.relu = nn.ReLU()
        if self.include_relation_embedding:
            self.relation_embeddings = nn.Embedding(len(self.id_to_label), self.relation_label_hidden_size)
        if self.include_direction_embedding:
            self.direction_embedding = nn.Embedding(len(self.id_to_direction), self.direction_hidden_size)
        
    @property
    def device(self):
        return self.missing_node.device

    @classmethod
    def load(cls, path, word2index):
        model = cls(word2index)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def save(self, path):
        torch.save(self.state_dict(), path)

    def merge_embeddings(self, embed_1, embed_2, relation_embedding):
        # return torch.max(embed_1, embed_2)
        # return self.relu(self.merge_layer(torch.cat((embed_1, embed_2))))
        # print("emb1: ", embed_1.shape, "\n", embed_1)
        return self.treelstm(embed_1.unsqueeze(dim=0), embed_2.unsqueeze(dim=0), relation_embedding).squeeze(dim=0)

    def make_features(self, parser, incl_buffer):
        """Gets a parser and returns an embedding that represents its current state.
        The state is represented by the concatenation of the embeddings corresponding
        to the two nodes on the top of the stack and the next node in the buffer.
        """
        s1 = self.missing_node if len(parser.stack) < 2 else parser.stack[-2].embedding
        s0 = self.missing_node if len(parser.stack) < 1 else parser.stack[-1].embedding
        if incl_buffer:
            b = self.missing_node if len(parser.buffer) < 1 else parser.buffer[0].embedding
        else:
            b = self.missing_node
        return torch.cat([s1, s0, b])

    def best_legal_action(self, actions, scores):
        """Gets a list of legal actions w.r.t the current state of the parser
        and the predicted scores for all possible actions. Returns the legal action
        with the highest score."""
        if len(actions) == 1:
            # only one legal action available
            return self.action_to_id[actions[0]]
        elif len(actions) == scores.shape[1]:
            # all actions are legal
            return torch.argmax(scores)
        else:
            # some actions are illegal, beware
            action_ids = [self.action_to_id[a] for a in actions]
            mask = torch.ones_like(scores) * -inf
            mask[:, action_ids] = 0
            masked_scores = scores + mask
            return torch.argmax(masked_scores)

    # adapted from pat
    def prepare(self, edus, word2index):
        x = [torch.tensor([word2index[word] if word in word2index else word2index["<unk>"] for word in edu]).to(self.device) for edu in edus]
        x_length = np.array([len(edu) for edu in x])
        padded_x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        return padded_x, x_length

    # adapted from pat
    def edus2padded_sequences(self, edus, tokenizer):

        w = [[word for word in tokenizer(edu)] for edu in edus]
        w, x_lengths = self.prepare(w, self.word2index)
        
        return w, x_lengths

    # my version of padding
    def indexAndPad(self, token_sequences, max_length):
        encodings_padded_ids = []
        for edu in token_sequences:
            ids = [self.word2index[word] if word in self.word2index.keys() else self.word2index['<unk>'] for word in edu]
            if len(ids) < max_length:
                ids.extend([0] * (max_length - len(ids) ))
            encodings_padded_ids.append(ids)
        return encodings_padded_ids

    def average_without_padding(self, padded_input, input_lengths):
        # for every padded input, take the mean of the token vectors that are not pads---we know that from the corresponding input lengths
        # stack the resulting mean vectors
        avg_without_padding = torch.stack([torch.mean(padded_input[i][:input_lengths[i]], dim=0) for i in range(padded_input.shape[0])])
        return avg_without_padding

    def replace_connectives(self, string, connectives):
        for conn in connectives:
            string = re.sub(r'\b(' + conn + r'|' + conn.capitalize()  + r')\b', "[UNK]", string)
        return string

    def forward(self, edus, gold_tree=None):

        if config.NO_CONNECTIVES:
            connectives = config.CONNECTIVES
            new_edus = [self.replace_connectives(edu, connectives) for edu in edus]
            w, x_length = self.edus2padded_sequences(new_edus, self.tokenizer)

        else:
            w, x_length = self.edus2padded_sequences(edus, self.tokenizer)
        we = self.word_embedding(w)
        packed = torch.nn.utils.rnn.pack_padded_sequence(we, x_length, batch_first=True, enforce_sorted=False)
        # print(packed)
        after_bilstm, _ = self.bilstm(packed)
        sequence_output, _ = torch.nn.utils.rnn.pad_packed_sequence(after_bilstm, batch_first=True)

        #todo: set pad to zeros
        # unk to avg
        # the output of bilstm, needs to be pooled in some way and pads shouldnt be included 
        
        # version 2:
        # # tokenize edus
        # encodings_variable_size_tokens = [self.tokenizer(edu) for edu in edus]
        # lengths = [len(enc) for enc in encodings_variable_size_tokens]         
        # max_length = max(lengths)
        # encodings_padded_ids = self.indexAndPad(encodings_variable_size_tokens, max_length)
        # encoding_glove_vectors = self.word_embedding(torch.LongTensor(encodings_padded_ids).to(self.device))
        # sequence_output, _ = self.bilstm(encoding_glove_vectors) # the first part of the tuple is 'output', second is a tuple with hidden state and cell state (https://pytorch.org/docs/master/generated/torch.nn.LSTM.html)
        
        # print(sequence_output)
        # print(sequence_output.shape)
        # enc_edus = torch.mean(sequence_output, dim=1)
        enc_edus = self.average_without_padding(sequence_output, x_length)
        enc_edus = self.bert_drop(enc_edus) 
        # print(sequence_output.shape)
        # print("enc edus shape: ", enc_edus.shape)

        #TODO: try larger batches
        #TODO: get someone else's implementation
        #TODO: try with attention
        #TODO: try other glove/ other emb
        #TODO: double-check the averaging 
        #TODO: have both dev and test with the 5 models from 5 diff random seeds - add code for that

        # enc_edus = self.project(enc_edus) 

        # make treenodes
        buffer = []
        for i in range(enc_edus.shape[0]):
            buffer.append(TreeNode(leaf=i, embedding=enc_edus[i]))

        # initialize automata
        parser = TransitionSystem(buffer)

        losses = []

        while not parser.is_done():
            state_features = self.make_features(parser, True)
            # legal actions for current parser
            legal_actions = parser.all_legal_actions()
            # predict next action, label, and direction
            action_scores = self.action_classifier(state_features).unsqueeze(dim=0)
            # label_scores = self.label_classifier(state_features).unsqueeze(dim=0)
            # # direction_scores = self.direction_classifier(state_features).unsqueeze(dim=0)
            
            if self.id_to_action[self.best_legal_action(legal_actions, action_scores)].startswith("reduce"):

                state_features_for_labels = self.make_features(parser, False) #make a new set of features without the buffer
                label_scores = self.label_classifier(state_features_for_labels).unsqueeze(dim=0)
            else:
                label_scores = self.label_classifier(state_features).unsqueeze(dim=0)
            
            # are we training?
            if gold_tree is not None:
                gold_step = parser.gold_step(gold_tree)
                # unpack step
                gold_action = torch.tensor([self.action_to_id[gold_step.action]], dtype=torch.long).to(self.device)
                gold_label = torch.tensor([self.label_to_id[gold_step.label]], dtype=torch.long).to(self.device)
                # gold_direction = torch.tensor([self.direction_to_id[gold_step.direction]], dtype=torch.long).to(self.device)
                # calculate loss
                loss_on_actions = loss_fn(action_scores, gold_action)
                loss_on_labels = loss_fn(label_scores, gold_label)
                # loss_on_direction = loss_fn(direction_scores, gold_direction)
                loss = loss_on_actions + loss_on_labels #+ loss_on_direction 
                # store loss for later
                losses.append(loss)
                # teacher forcing
                next_action = gold_action
                next_label = gold_label
                # next_direction = gold_direction
            else:
                next_action = self.best_legal_action(legal_actions, action_scores)
                # next_label = label_scores.argmax().unsqueeze(0) #unsqueeze because after softmax the output tensor is tensor(int) instead of tensor([int]) (different from next_label in training)
                if self.id_to_action[next_action].startswith("reduce"):
                    next_label = label_scores.argmax().unsqueeze(0) #unsqueeze because after softmax the output tensor is tensor(int) instead of tensor([int]) (different from next_label in training)
                else:
                    torch.tensor(0, dtype=torch.long).to(self.device)
                
                # next_direction = direction_scores.argmax().unsqueeze(0)
            
            if self.include_relation_embedding:
                rel_emb = self.relation_embeddings(next_label)
                # if self.include_direction_embedding:
                #     dir_emb = self.direction_embedding(next_direction)
                #     rel_dir_emb = torch.cat((rel_emb, dir_emb), dim=1)
                # else:
                #     rel_dir_emb = rel_emb
            else:
                rel_dir_emb = None  
            
            action=self.id_to_action[next_action]
            parser.take_action(
                action=action,
                label=self.id_to_label[next_label] if action.startswith("reduce") else "None",
                direction=None,
                reduce_fn=self.merge_embeddings,
                rel_embedding = rel_dir_emb
            )

        # returns the TreeNode for the tree root
        predicted_tree = parser.get_result()
        outputs = (predicted_tree,)

        # are we training?
        if gold_tree is not None:
            loss = sum(losses) / len(losses)
            outputs = (loss,) + outputs

        return outputs