import torch
from torch import nn
from transformers import *
from rst import TreeNode
from transition_system import TransitionSystem
from treelstm import TreeLstm
import config

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
    
    def legal_action_scores(self, actions, scores):
        """Gets a list of legal actions w.r.t the current state of the parser
        and the predicted scores for all possible actions. Returns only the scores for the legal actions."""

        # some actions are illegal, beware
        action_ids = [self.action_to_id[a] for a in actions]
        mask = torch.ones_like(scores) * -inf
        for i in action_ids:
            mask[0][i] = 0

        masked_scores = scores + mask
        return masked_scores

    def forward(self, edus, gold_tree=None):

        # tokenize edus
        encodings = self.tokenizer.encode_batch(edus)
        ids = torch.tensor([e.ids for e in encodings], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([e.attention_mask for e in encodings], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([e.type_ids for e in encodings], dtype=torch.long).to(self.device)

        # encode edus
        sequence_output, pooled_output = self.bert(
            ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # print('sequence', sequence_output.shape)
        # print('pooled', pooled_output.shape)
        # enc_edus = self.bert_drop(pooled_output)
        enc_edus = self.bert_drop(sequence_output[:,0,:])
        enc_edus = self.project(enc_edus)

        # make treenodes
        buffer = []
        for i in range(enc_edus.shape[0]):
            buffer.append(TreeNode(leaf=i, embedding=enc_edus[i]))

        # initialize automata
        parser = TransitionSystem(buffer)

        
        #diverge train and eval here
        if gold_tree is not None:

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

                gold_step = parser.gold_step(gold_tree)
                # unpack step
                gold_action = torch.tensor([self.action_to_id[gold_step.action]], dtype=torch.long).to(self.device)
                gold_label = torch.tensor([self.label_to_id[gold_step.label]], dtype=torch.long).to(self.device)
                gold_direction = torch.tensor([self.direction_to_id[gold_step.direction]], dtype=torch.long).to(self.device)
                # calculate loss
                loss_1 = loss_fn(action_scores, gold_action)
                loss_2 = loss_fn(label_scores, gold_label)
                loss_3 = loss_fn(direction_scores, gold_direction)
                loss = loss_1 + loss_2 + loss_3
                # store loss for later
                losses.append(loss)
                # teacher forcing
                next_action = gold_action
                next_label = gold_label
                next_direction = gold_direction
                parser.take_action(
                action=self.id_to_action[next_action],
                label=self.id_to_label[next_label],
                direction=self.id_to_direction[next_direction],
                reduce_fn=self.merge_embeddings,
            )



        else:

            # for every highest scored combo of action/label/dir, take an action and store the parser and cur score (make sure to update parser score)
            #OR
            # for every combo of action/label/dir (maybe also choose top k?), calc scores with previous parsers, then take action and store the parser and cur score
            parsers_done = []
            parsers = [[list(), 0.0]] #for us, it's parsers (?)
            # walk over each step in sequence
            while len(parsers_done) < 5:
            # while not parser.is_done():
                all_candidates = list() #here will be all parser candidates (previous parser updated with current steps)
		        # expand each current candidate
                for i in range(len(parsers)):
                    # for every previously found sequence, get the seq and its score (we'll update them with new scores)
                    parser, score = parsers[i] #this is one previous parser
                    #we want to see several ways of how we can update it

                    
                    state_features = self.make_features(parser)
                    # legal actions for current parser
                    legal_actions = parser.all_legal_actions()
                    # predict next action, label, and direction
                    action_scores = self.action_classifier(state_features).unsqueeze(dim=0)
                    legal_action_scores = self.legal_action_scores(legal_actions, action_scores)
                    label_scores = self.label_classifier(state_features).unsqueeze(dim=0)
                    direction_scores = self.direction_classifier(state_features).unsqueeze(dim=0)
     
                    # if len(legal_actions) == 1:
                    #     print('legal actions: ', legal_actions)
                    #     print("1: ", action_scores)
                    #     print("2: ", legal_action_scores)
                    # print("label scores: ", label_scores)
                    # print("direction scores: ", direction_scores)

           
                    # next_action = self.best_legal_action(legal_actions, action_scores)
                    # next_label = label_scores.argmax()
                    # next_direction = direction_scores.argmax()

                    #get top k action/label/dir combos and their scores
                    top_k_combos, combo_score = some_new_method_for_getting_top_combos(legal_action_scores, label_scores, direction_scores)
                    #would the combo score be mult or sum? or something else?

                    for combo in top_k_combos:
                        # take the next parser step
                        parser.take_action(
                            action=self.id_to_action[next_action],
                            label=self.id_to_label[next_label],
                            direction=self.id_to_direction[next_direction],
                            reduce_fn=self.merge_embeddings,
                        )

                    all_candidates.append(parser, score + combo) #this is the new parser after the action has been taken with the score updated

                #now we have several parse/score candidates
                #get top scoring parsers (remember to incorporate previous score)
                #should we normalize at every step?
            
                #out of all the top scoring parsers, check if any are done? 
                if parser.is_done:
                    parsers_done.append(parser)


        #out of done parsers, choose the highest prob one

        #here, get the results from the highest scoring of the done parsers
        
        # returns the TreeNode for the tree root
        predicted_tree = parser.get_result()
        outputs = (predicted_tree,)

        # are we training?
        if gold_tree is not None:
            loss = sum(losses) / len(losses)
            outputs = (loss,) + outputs

        return outputs