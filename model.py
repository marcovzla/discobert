import torch
from torch import nn
from transformers import *
from rst import TreeNode
from transition_system import TransitionSystem
from treelstm import TreeLstm
from copy import deepcopy
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
        self.beam_size = config.BEAM_SIZE
        self.hidden_size = config.HIDDEN_SIZE
        self.relation_label_hidden_size = config.RELATION_LABEL_HIDDEN_SIZE
        self.direction_hidden_size = config.DIRECTION_HIDDEN_SIZE
        self.include_relation_embedding = config.INCLUDE_RELATION_EMBEDDING
        self.include_direction_embedding = config.INCLUDE_DIRECTION_EMBEDDING
        # init model
        self.tokenizer = config.TOKENIZER
        self.bert = BertModel.from_pretrained(self.bert_path)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        self.attn1 = nn.Linear(self.bert.config.hidden_size, 100)
        self.attn2 = nn.Linear(100, 1)
        self.betweenAttention = nn.Tanh()
        self.bert_drop = nn.Dropout(self.dropout)
        self.project = nn.Linear(self.bert.config.hidden_size, self.hidden_size)
        self.missing_node = nn.Parameter(torch.rand(self.hidden_size, dtype=torch.float))
        self.action_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_action))
        self.label_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_label))
        self.direction_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_direction))
        # self.merge_layer = nn.Linear(2 * self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.treelstm = TreeLstm(self.hidden_size // 2, self.include_relation_embedding, self.include_direction_embedding, self.relation_label_hidden_size, self.direction_hidden_size)
        self.relu = nn.ReLU()
        if self.include_relation_embedding:
            self.relation_embeddings = nn.Embedding(len(self.id_to_label), self.relation_label_hidden_size)
        if self.include_direction_embedding:
            self.direction_embedding = nn.Embedding(len(self.id_to_direction), self.direction_hidden_size)
        self.logSoftmax = nn.LogSoftmax(dim=0)

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
        if len(parser.stack) < 1:
            s0 = self.missing_node 
        else:
            tens = parser.stack[-1]
            s0 = tens.embedding
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
    
    def legal_action_scores(self, actions, scores):
        """Gets a list of legal actions w.r.t the current state of the parser
        and the predicted scores for all possible actions. Returns only the scores for the legal actions."""
        # some actions are illegal, beware
        action_ids = [self.action_to_id[a] for a in actions]
        mask = torch.ones_like(scores) * -inf
        mask[action_ids] = 0
        masked_scores = scores + mask
        return masked_scores

    def getScoreCombinations(self, action_scores, label_scores, direction_scores):
        combos = []
        num_of_actions = len(action_scores)
        num_of_labels = len(label_scores)
        num_of_directions = len(direction_scores)
        summable_action_scores = (torch.cat([action_scores.squeeze(0)[i].repeat(num_of_labels * num_of_directions) for i in range(action_scores.squeeze(0).shape[0])])).view(num_of_actions, num_of_labels, num_of_directions)
        label_scores_half = torch.cat([label_scores.squeeze(0)[i].repeat(num_of_directions) for i in range(label_scores.squeeze(0).shape[0])])
        summable_label_scores = torch.cat([label_scores_half, label_scores_half]).view(num_of_actions, num_of_labels, num_of_directions)
        summable_direction_scores = direction_scores.squeeze(0).repeat(num_of_actions * num_of_labels).view(num_of_actions, num_of_labels, num_of_directions)

        all_scores = summable_action_scores + summable_label_scores + summable_direction_scores
        return all_scores   

   
    #based on https://discuss.pytorch.org/t/return-indexes-of-top-k-maximum-values-in-a-matrix/54698
    def top_k_in_3d_matrix(self, three_d_torch_tensor, k):
        D, H, W = three_d_torch_tensor.shape
        #1D tensor that topk can be used on to get overall max instead of per dimension
        three_d_torch_tensor = three_d_torch_tensor.view(-1)
        #get the indices of top k values
        _, indices = three_d_torch_tensor.topk(k, sorted=True)
        #get a tensor containing the position of each max value in the original vector, e.g., 
        # tensor([[0, 2, 3],
        #         [1, 1, 2]])
        # means k = 2, with the first max value at position three_d_torch_tensor[0][2][3] and the second max value at position [1][1][2]
        topk_positions = torch.cat(((indices // (W*H)).unsqueeze(1), (indices % (W * H) // W).unsqueeze(1), (indices % (W * H)  % W).unsqueeze(1)), dim=1)
        return topk_positions


    def forward(self, train, edus, gold_tree=None):
        
        # tokenize edus
        encodings = self.tokenizer.encode_batch(edus)
        ids = torch.tensor([e.ids for e in encodings], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([e.attention_mask for e in encodings], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([e.type_ids for e in encodings], dtype=torch.long).to(self.device)

        # encode edus
        sequence_output, pooled_output = self.bert( #sequence_output: [edu, tok, emb]
            ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

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

        enc_edus = self.project(enc_edus) 

        # make treenodes
        buffer = []
        for i in range(enc_edus.shape[0]):
            buffer.append(TreeNode(leaf=i, embedding=enc_edus[i]))

        # initialize automata
        parser = TransitionSystem(buffer, None)
       
        if train==False:

            gold_sequence = parser.gold_path(gold_tree)
            
        losses = []    
        
        #diverge train and eval here
        if train==True:         
            while not parser.is_done():
                state_features = self.make_features(parser)
                # legal actions for current parser
                legal_actions = parser.all_legal_actions()
                # predict next action, label, and direction
                action_scores = self.action_classifier(state_features).unsqueeze(dim=0)
                label_scores = self.label_classifier(state_features).unsqueeze(dim=0)
                direction_scores = self.direction_classifier(state_features).unsqueeze(dim=0)

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

                #TODO: add this during eval as well! Omitting this now to not overcrowd the already crowded code draft
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


        else:

            
            # storage for done parsers
            parsers_done = []
            parsers = [[parser, 0.0, 1, "init_step"]]  #[[parser, score:Float, stepsTaken:Int, [seq of transition scores that led to this]]] 
            
            # go through each current parser
            # continue until we have beam_size done parsers to choose the best one from or until there are no parsers left to expand
            while len(parsers_done) < self.beam_size and len(parsers) > 0: 
               
                all_candidates = list() # this will store all parser candidates (i.e. all the parsers that are the result of applying current steps to the parsers produced during previous steps)
		        
                # expand each parser produced and stored during last step and store all resulting parser candidates in all_candidates (to later pick top k to append to parsers for next step)
                for i in range(len(parsers)):
                    
                    # for every parser from last step, get the parser and its score (we'll update them with new transition scores after taking an action)
                    parser, score, steps, tr_scores_str = parsers[i] #this is one previously stored parser (either the first one initialized or a parser we got by taking an action during previous step)

                    state_features = self.make_features(parser)
                    # legal actions for current parser
                    legal_actions = parser.all_legal_actions()
                    
                    # get next action, label, and direction scores
                    action_scores = self.action_classifier(state_features)
                    # we will only need the scores for legal actions, so the illegal actions will be set to 'inf'
                    legal_action_scores = self.logSoftmax(self.legal_action_scores(legal_actions, action_scores))
                    label_scores_no_log = self.label_classifier(state_features)
                    label_scores = self.logSoftmax(label_scores_no_log)
                    direction_scores_no_log = self.direction_classifier(state_features)
                    direction_scores = self.logSoftmax(direction_scores_no_log)
                    
                    combo_scores = self.getScoreCombinations(legal_action_scores, label_scores, direction_scores) #returns a matrix of shape [len(action_scores), len(label_scores), len(direction_scores)]
                    
                    #we want to get top k parsers that highest scored action-label-direction combinations will produce
                    top_k_combos = self.top_k_in_3d_matrix(combo_scores, 4) #todo: beam size or some other # of score combos? # tensor of shape [beam_size, 3], with 3 being the three classifier classes (action, label, dir)
                    # for every top action/label/direction combo, get its score from the combo_scores matrix
                    for combo in top_k_combos:
                        action_idx = combo[0].item()
                        label_idx = combo[1].item()
                        dir_idx = combo[2].item()
                        combo_score = combo_scores[action_idx][label_idx][dir_idx]
                        parser_cand = TransitionSystem(parser.buffer, parser.stack) 
                        # get the action/label/direction from the combo
                        action=self.id_to_action[action_idx]
                        label=self.id_to_label[label_idx]
                        direction=self.id_to_direction[dir_idx]

                        # take the corresponding step
                        parser_cand.take_action(
                                    action=action,
                                    label=label,
                                    direction=direction,
                                    reduce_fn=self.merge_embeddings,
                                )
                        
                        # append the resulting parser to all candidates
                        # update the score (previous score for this parser + new transition score, i.e., the combo score)
                        # update the step (steps will be used for normalizing the score for done parsers at the end)
                        # the rest is added for debugging (can be deleted once the code is solidified)
                        all_candidates.append([parser_cand, score + combo_score, steps + 1, tr_scores_str + "+" + str(combo_score) + "_act:" + str(action_idx) + "-label:" + str(label_idx) + "-dir:" + str(dir_idx)]) #this is the new parser after the action has been taken with the score updated

                              
                # now, for this step, we have several parser/score candidates
                # get top k scoring parsers
                
                #sort candidates by score
                sorted_candidates = sorted(all_candidates, key = lambda x: x[1], reverse=True)
                #now top k candidates are the new parsers (discarding the parsers from last steps because those have been expanded during this step)
                parsers = sorted_candidates[:self.beam_size]

                # check if any of the parsers are done and remove the ones that are
                to_remove = []
                for parser in parsers:
                    if parser[0].is_done(): 
                        parsers_done.append(parser)
                        to_remove.append(parser)


                for parser in to_remove:
                    parsers.remove(parser) 
                    
            # after we have reached a set number of done parsers OR there are no more parsers to expand,
            # normalize the done parsers (divide the score for the parser by the number of steps that have been taken)
            normalized_parsers = []
            for parser in parsers_done:
                score = float(parser[1])/parser[2] # parser[0] is the parser itself, parser[1] is its score, parser[2] is the # of steps
                normalized_parsers.append((parser[0], score)) # at this point, it's safe to just store the parser and its normalized score

            parser = max(normalized_parsers, key = lambda x: x[1])[0]

        predicted_tree = parser.get_result()
        predicted_step_sequence = parser.gold_path(predicted_tree)

        outputs = (predicted_tree,)

        # are we training?
        if train:
            loss = sum(losses) / len(losses)
            outputs = (loss,) + outputs

        return outputs