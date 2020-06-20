import torch
from torch import nn
from transformers import *
from rst import TreeNode
from transition_system import TransitionSystem
from treelstm import TreeLstm
import config
import torch.nn.functional as F


inf = float('inf')

def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

class DiscoBertModel(nn.Module):
    
    def __init__(self):
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
        self.hidden_size = 200
        # init model
        self.tokenizer = config.TOKENIZER
        self.bert = BertModel.from_pretrained(self.bert_path)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        self.attn1 = nn.Linear(self.bert.config.hidden_size, 100)
        self.attn2 = nn.Linear(100, 1)
        self.betweenAttention = nn.ReLU()
        self.bert_drop = nn.Dropout(self.dropout)
        self.project = nn.Linear(self.bert.config.hidden_size, self.hidden_size)
        self.missing_node = nn.Parameter(torch.rand(self.hidden_size, dtype=torch.float))
        self.action_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_action))
        self.label_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_label))
        self.direction_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_direction))
        # self.merge_layer = nn.Linear(2 * self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.treelstm = TreeLstm(self.hidden_size // 2)
        self.relu = nn.ReLU()

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

    def merge_embeddings(self, embed_1, embed_2):
        # return torch.max(embed_1, embed_2)
        # return self.relu(self.merge_layer(torch.cat((embed_1, embed_2))))
        return self.treelstm(embed_1.unsqueeze(dim=0), embed_2.unsqueeze(dim=0)).squeeze(dim=0)

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
        elif len(actions) == scores.shape[0]:
            # all actions are legal
            return torch.argmax(scores)
        else:
            # some actions are illegal, beware
            action_ids = [self.action_to_id[a] for a in actions]
            mask = torch.ones_like(scores) * -inf
            mask[action_ids] = 0
            masked_scores = scores + mask
            return torch.argmax(masked_scores)

    def forward(self, edus, gold_tree=None):

        # tokenize edus
        encodings = self.tokenizer.encode_batch(edus)
        ids = torch.tensor([e.ids for e in encodings], dtype=torch.long).to(self.device)
        # print("ids shape: ", ids.shape)
        attention_mask = torch.tensor([e.attention_mask for e in encodings], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([e.type_ids for e in encodings], dtype=torch.long).to(self.device)

        # encode edus
        sequence_output, pooled_output = self.bert( #first token represents the full sentence in seq output; in pooled it's different; maybe drop that first token? maybe not. hm.
            ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # print("mask shape: ", attention_mask.shape)


        #todo: try with these settings:
        sequence_output = sequence_output[:, 1:, :] 
        attention_mask = attention_mask[:, 1:]
        


        #TODO:
        #make a results table with different settings (eg dropping one, adding linearity, etc)

        #try non linearity between attn layers
        #try removing first token 
        #attend on edus around
        #Becky: need to focus on S - predict reduce left/right 
        #check if trees are binarized 
        #visualize the tree - ask becky to send the ref to codebase
        # marco: try different flavors of bert? 
        # becky: weighting components of the loss, ; if loss only be S, does S improve; how high does S get if loss is only that? if S gets high, maybe do curriculum learning. curr learning can be based on doc length or it can be based on loss, e.g., (see below where losses are)
        # if architecture is promising, then can try switching out bert trained with the most appropriate task (sent prediction)
        # check if eval is overly punitive on R!!!!
        # but could also have a paper with bert not really helping
        # if i give you the gold structure, does bert help with relation classifier
        # analysis of what components bert helps with or does not help with

        #what needs to be done:

        #encode a sequence of edus 

        # one opetion to get a a vector for each edu is by pooling the token vectors, so for every doc, we get 40 edu vectors 
        # such that the token embeddings are not pooled, but weighted


        
        #what we have now is sequence output:
        # for every doc:

        # [
        #   [edu1 
        #       [tok1_emb], 
        #       [tok2_emb]
        #   ], 
        #   [edu2 
        #       [tok1_emb], 
        #       [tok2_emb]
        #   ]
        # ]

        
        # we want to find attention weights for each token and use those weights to create a vector for every edu where token embeddings will be multiplied by their weights? 

        #pass input through two att layers and softmax the result?

        # mult input by those weights? 


        # print(sequence_output)
        # print('sequence', sequence_output.shape) #eg [40, 38, 768]
        # # print("seq output: ", sequence_output)
        # print('pooled', pooled_output.shape) #eg [40, 768] 
        # enc_edus = self.bert_drop(pooled_output)
        
        # sequence_output = sequence_output.view(1, 1, -1)
        
        after1stAttn = self.attn1(sequence_output)
        #Marco: add non-linearity between the two attentions, otherwise this could be a single layer - becky also was suggesting trying 1 layer at some point

        # print("after 1st attn: ", after1stAttn)

        # print('after 1st attn: ', after1stAttn.shape)

        # nonLinearity = self.betweenAttention(after1stAttn)

        # after2ndAttn = self.attn2(nonLinearity)
        after2ndAttn = self.attn2(after1stAttn)
        # print("after 2nd att: ", after2ndAttn)
        # print('after 2nd attn: ', after2ndAttn.shape)

        # print("first of after second attention: ", after2ndAttn[0])

        # print("after att shape: ", after2ndAttn.shape)
        attention_mask = attention_mask.unsqueeze(dim=-1)
        masked_att = after2ndAttn * attention_mask
        attn_weights = F.softmax(masked_att, dim=1)
        # print("att weights shape: ", attn_weights.shape)
        # print("att weights: ", attn_weights)
        # print("one of the att weights: ", attn_weights[0])

        attn_applied =  sequence_output * attn_weights #[9, 17, 768] [9, 17, 1]  

        #first dimension is normally the batch

        summed_up = torch.sum(attn_applied, dim=1)

        # print("summed_up: ", summed_up.shape)
        # output = torch.cat((enc_edus[0], attn_applied[0]), 0)
        # # score = torch.tahn(after1stAttn + after2ndAttn)
        
        # # attn_applied = torch.mul(attn_weights, enc_edus)#.view(1,1,-1))#.view(sequence_output.shape[0], sequence_output.shape[1], sequence_output.shape[2])
        # # print("shape attn applied: ", attn_applied.shape)
        # enc_edus = self.attn_combine(output)

        # enc_edus = self.attn(enc_edus)
        # print('enc_edus after attn: ', enc_edus.shape)
        # enc_edus = torch.nn.Softmax(self.attn_combine(enc_edus))
        # print('enc_edus after attn combine: ', enc_edus.shape)
        # attn_applied = attn_applied.view(sequence_output.shape[0], sequence_output.shape[1], sequence_output.shape[2])
        
        # print('enc educ after drop ', enc_edus.shape)
        enc_edus = self.bert_drop(summed_up)
        # print("enc edus after dropout: ", enc_edus.shape)
        enc_edus = self.project(enc_edus) #try without this
        # print('enc educ after project ', enc_edus.shape)

        # enc_edus = self.attn(sequence_output[:,0,:])
        # print('enc_edus after attn: ', enc_edus.shape)
        # enc_edus = self.attn_combine(enc_edus)
        # print('enc_edus after attn combine: ', enc_edus.shape)
        # enc_edus = self.bert_drop(enc_edus)
        # print('enc_edus after bert dropout: ', enc_edus.shape)
        # enc_edus = self.project(enc_edus)
        # print('enc_edus after project: ', enc_edus.shape)


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
                loss_1 = loss_fn(action_scores, gold_action)
                loss_2 = loss_fn(label_scores, gold_label)
                loss_3 = loss_fn(direction_scores, gold_direction)
                loss = loss_1 + loss_2 + loss_3 #todo: rename :) 
                #todo: (from becky) change to just loss1 
                #can make a scheduler to make sure losses are
                #weighted different at different stages of learning 
                #train the model with initial params 0.9 * loss1 + 0.5 loss2 + 0.5 loss 3 -> freeze (not pretty)
                #if keep only loss 1, dont compute loss2 and 3
                #if S improves with just loss1, then there's hope for this architecture
                # store loss for later
                losses.append(loss)
                # teacher forcing
                next_action = gold_action
                next_label = gold_label
                next_direction = gold_direction
            else:
                next_action = self.best_legal_action(legal_actions, action_scores)
                next_label = label_scores.argmax()
                next_direction = direction_scores.argmax()

            # take the next parser step
            parser.take_action(
                action=self.id_to_action[next_action],
                label=self.id_to_label[next_label],
                direction=self.id_to_direction[next_direction],
                reduce_fn=self.merge_embeddings,
            )

        # returns the TreeNode for the tree root
        predicted_tree = parser.get_result()
        outputs = (predicted_tree,)

        # are we training?
        if gold_tree is not None:
            loss = sum(losses) / len(losses)
            outputs = (loss,) + outputs

        return outputs