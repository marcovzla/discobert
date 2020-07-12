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

inf = float('inf')

def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

class DiscoBertModel(nn.Module):
    
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
        self.encoding = config.ENCODING
        if self.encoding == 'bert':
            self.encoder = BertModel.from_pretrained(self.bert_path)
        elif self.encoding == 'roberta':
            self.encoder = RobertaModel.from_pretrained(self.bert_path)
        elif self.encoding == 'glove':
            self.glove = load_glove(config.GLOVE_PATH)
            self.word2index = word2index
            self.index2word = make_index2word(word2index)
            self.embedding_matrix = make_embedding_matrix(self.index2word, self.glove)
            self.word_embedding = nn.Embedding(
                num_embeddings=len(self.word2index.keys()),
                embedding_dim=config.EMBEDDING_SIZE
            )
            self.word_embedding.load_state_dict({'weight': self.embedding_matrix})

        #     self.glove = nn.Embedding.from_pretrained("/home/alexeeva/data/glove/vectors.txt")
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        if config.BERT_LESS == True:
            self.attn1 = nn.Linear(200, 100) #todo: will depend on the output of bilstm? same as self.project
        else:
            self.attn1 = nn.Linear(self.encoder.config.hidden_size, 100)
        self.attn2 = nn.Linear(100, 1)
        self.betweenAttention = nn.Tanh()
        self.bert_drop = nn.Dropout(self.dropout)
        if config.BERT_LESS == True:
            self.project = nn.Linear(200, self.hidden_size) #below, I just made this optional for bert
        else:
            self.project = nn.Linear(self.encoder.config.hidden_size, self.hidden_size)
        self.missing_node = nn.Parameter(torch.rand(self.hidden_size, dtype=torch.float))
        self.action_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_action))
        self.label_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_label))
        self.direction_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_direction))
        # self.merge_layer = nn.Linear(2 * self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.treelstm = TreeLstm(self.hidden_size // 2, self.include_relation_embedding, self.include_direction_embedding, self.relation_label_hidden_size, self.direction_hidden_size)
        self.bilstm = nn.LSTM(config.EMBEDDING_SIZE, 100, bidirectional=True)
        self.relu = nn.ReLU()
        if self.include_relation_embedding:
            self.relation_embeddings = nn.Embedding(len(self.id_to_label), self.relation_label_hidden_size)
        if self.include_direction_embedding:
            self.direction_embedding = nn.Embedding(len(self.id_to_direction), self.direction_hidden_size)
        
    @property
    def device(self):
        return self.missing_node.device

    @classmethod
    def load(cls, path):
        model = cls()
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

    def make_features(self, parser):
        """Gets a parser and returns an embedding that represents its current state.
        The state is represented by the concatenation of the embeddings corresponding
        to the two nodes on the top of the stack and the next node in the buffer.
        """
        s1 = self.missing_node if len(parser.stack) < 2 else parser.stack[-2].embedding
        s0 = self.missing_node if len(parser.stack) < 1 else parser.stack[-1].embedding
        b = self.missing_node if len(parser.buffer) < 1 else parser.buffer[0].embedding
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

    # def average_embeddings(self, edu):
    #     emb_sum = torch.zeros(1, config.EMBEDDING_SIZE)
    #     for word in edu:
    #         if word in self.word2index.keys():
    #             print("word in vocab:", word)
                
    #             emb = self.word_embedding(self.word2index[word])
    #             print(emb)
    #             if word == "furloughs":
    #                 print("word and index: ", word, " ", self.word2index[word])
    #                 print("emb: ", emb)
                
    #             # print("emb: ", emb)
    #         else:
    #             # print("word not in vocab")
    #             emb = self.embedding_matrix[1] #unk emb 
    #         emb_sum += emb
    #     return emb_sum/len(edu) 
    # 
    # def pad_encodings(self, edus):
    #     max_length = max([len(edu) for edu in edus])
            
    #     return max_length

    def forward(self, edus, gold_tree=None):

        # print("emb matrix shape: ", self.embedding_matrix.shape)

        if self.encoding == "bert":
            # tokenize edus
            encodings = self.tokenizer.encode_batch(edus)
            ids = torch.tensor([e.ids for e in encodings], dtype=torch.long).to(self.device)
            attention_mask = torch.tensor([e.attention_mask for e in encodings], dtype=torch.long).to(self.device)
            token_type_ids = torch.tensor([e.type_ids for e in encodings], dtype=torch.long).to(self.device)

            # encode edus
            sequence_output, pooled_output = self.encoder( #sequence_output: [edu, tok, emb]
                ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        elif self.encoding == "roberta":
            # tokenize edus 
            batched_encodings = self.tokenizer(edus, padding=True, return_attention_mask=True, return_tensors='pt').to(self.device) #add special tokens is true by default
            ids = batched_encodings['input_ids']
            attention_mask = batched_encodings['attention_mask']
            # encode edus
            sequence_output, pooled_output = self.encoder(ids, attention_mask)

        elif self.encoding == "glove":
            # tokenize edus
            encodings_variable_size_tokens = [self.tokenizer(edu) for edu in edus]
            encodings_variable_size_ids = []

            # for enc in encodings_variable_size:
            #     print(enc, " ", len(enc))
            # padded_encodings = self.pad_encodings(encodings_variable_size)
            # print("pad enc: ", padded_encodings)
            lengths = [len(enc) for enc in encodings_variable_size]
            encodings = torch.nn.utils.rnn.pack_padded_sequence(encodings_variable_size, lengths)
            print("encodings: ", encodings)
            # encoding_glove_vectors = torch.stack([self.average_embeddings(edu) for edu in encodings]).to(self.device)
            encoding_glove_vectors = self.word_embedding(encodings)
            print(encoding_glove_vectors)
            # print("encodings shape: ", encoding_glove_vectors.shape)
            sequence_output, pooled_output = self.bilstm(encoding_glove_vectors)
            # print("output: ", output[0])
            # sequence_output = output
            # print("seq output shape: ", sequence_output.shape)

            # for edu in encodings:
            #     avgeraged_emb = self.average_embeddings(edu)


        # whether or not drop the classification token in bert-like models
        if config.DROP_CLS == True:
            sequence_output = sequence_output[:, 1:, :] 
            attention_mask = attention_mask[:, 1:]
        
        if config.USE_ATTENTION == True:
            after1stAttn = self.attn1(sequence_output)
            nonLinearity = self.betweenAttention(after1stAttn)
            after2ndAttn = self.attn2(nonLinearity)
            attention_mask = attention_mask.unsqueeze(dim=-1)
            masked_att = after2ndAttn * attention_mask
            attn_weights = F.softmax(masked_att, dim=1)
            attn_applied =  sequence_output * attn_weights #[9, 17, 768] [9, 17, 1] 
            summed_up = torch.sum(attn_applied, dim=1)
            enc_edus = self.bert_drop(summed_up)

        else:
            enc_edus = self.bert_drop(sequence_output[:,0,:])

        if config.BERT_LESS == False:
            enc_edus = self.project(enc_edus) 

        # make treenodes
        buffer = []
        for i in range(enc_edus.shape[0]):
            buffer.append(TreeNode(leaf=i, embedding=enc_edus[i]))

        # initialize automata
        parser = TransitionSystem(buffer)

        losses = []

        while not parser.is_done():
            state_features = self.make_features(parser)
            # legal actions for current parser
            legal_actions = parser.all_legal_actions()
            # predict next action, label, and direction
            action_scores = self.action_classifier(state_features).unsqueeze(dim=0)
            label_scores = self.label_classifier(state_features).unsqueeze(dim=0)
            direction_scores = self.direction_classifier(state_features).unsqueeze(dim=0)
            # are we training?
            if gold_tree is not None:
                gold_step = parser.gold_step(gold_tree)
                # unpack step
                gold_action = torch.tensor([self.action_to_id[gold_step.action]], dtype=torch.long).to(self.device)
                gold_label = torch.tensor([self.label_to_id[gold_step.label]], dtype=torch.long).to(self.device)
                gold_direction = torch.tensor([self.direction_to_id[gold_step.direction]], dtype=torch.long).to(self.device)
                # calculate loss
                loss_on_actions = loss_fn(action_scores, gold_action)
                loss_on_labels = loss_fn(label_scores, gold_label)
                loss_on_direction = loss_fn(direction_scores, gold_direction)
                loss = loss_on_actions + loss_on_labels + loss_on_direction 
                # store loss for later
                losses.append(loss)
                # teacher forcing
                next_action = gold_action
                next_label = gold_label
                next_direction = gold_direction
            else:
                next_action = self.best_legal_action(legal_actions, action_scores)
                next_label = label_scores.argmax().unsqueeze(0) #unsqueeze because after softmax the output tensor is tensor(int) instead of tensor([int]) (different from next_label in training)
                next_direction = direction_scores.argmax().unsqueeze(0)
            
            if self.include_relation_embedding:
                rel_emb = self.relation_embeddings(next_label)
                if self.include_direction_embedding:
                    dir_emb = self.direction_embedding(next_direction)
                    rel_dir_emb = torch.cat((rel_emb, dir_emb), dim=1)
                else:
                    rel_dir_emb = rel_emb
            else:
                rel_dir_emb = None  

            parser.take_action(
                action=self.id_to_action[next_action],
                label=self.id_to_label[next_label],
                direction=self.id_to_direction[next_direction],
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