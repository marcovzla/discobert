import torch
from torch import nn
from transformers import *
from rst import TreeNode
from transition_system import TransitionSystem
from treelstm import TreeLstm
from copy import deepcopy
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
        self.beam_size = config.BEAM_SIZE
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
        self.rel_embeddings = nn.Embedding(len(config.ID_TO_LABEL), 50)

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
        # print("parser inside make features: ", parser)
        # print("parser stack: ", parser.stack)

        s1 = self.missing_node if len(parser.stack) < 2 else parser.stack[-2].embedding
        # print("s1: ", s1)
        # print("parser stack: ", parser.stack)
        
        if len(parser.stack) < 1:
            # print("len parser stack < 1")
            s0 = self.missing_node 
        else:
            # print("len parser stack not less than 1")
            tens = parser.stack[-1]
            # print("tens: ", tens.span)
            # print("tens emb: ", tens.embedding)
            print(tens.children)
            s0 = tens.embedding
        # print("s0: ", s0)
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
        print("scores  in legal action scores: ", scores)
        # some actions are illegal, beware
        action_ids = [self.action_to_id[a] for a in actions]
        mask = torch.ones_like(scores) * -inf
        for i in action_ids:
            mask[0][i] = 0

        masked_scores = scores + mask
        return masked_scores

    def getTopNCombos(self, legal_action_scores, label_scores, direction_scores, beam_size):

        # log_action_scores = torch.log(nn.Softmax(legal_action_scores))
        # log_label_scores = torch.log(nn.Softmax(label_scores))
        # log_dir_scores = torch.log(nn.Softmax(direction_scores))

        combos = []
        
        # print("leg act scores: ", legal_action_scores)
        for i in range(len(legal_action_scores.squeeze(dim=0))):
            
            if legal_action_scores[0][i] != -inf:
                # print("legal acr score: ", legal_action_scores[0][i])
                # print("not inf")
                # print("label scores: ", label_scores)
                for j in range(len(label_scores.squeeze(dim=0))):
                    # print("label score: ", label_scores[0][j])
                    # print("dir scores: ", direction_scores)
                    for k in range(len(direction_scores.squeeze(dim=0))):
                        # print("dir score: ", direction_scores[0][k])
                        log_act_score = torch.log(legal_action_scores[0][i])
                        # print("log act score: ", log_act_score)
                        # print("log label score: ", torch.log(label_scores[0][j]).item())
                        # print("log dir score: ", torch.log(direction_scores[0][k]).item())
                        combo_score = legal_action_scores[0][i].item() * label_scores[0][j].item() * direction_scores[0][k].item()
                        # combo_score = torch.log(legal_action_scores[0][i]).item() + torch.log(label_scores[0][j]).item() + torch.log(direction_scores[0][k]).item()
                        # print('combo score: ', combo_score)
                        combos.append(((i,j,k), combo_score))
        # print("sorted combos: ", sorted(combos, key = lambda x: x[1], reverse=True))
        sorted_combos = sorted(combos, key = lambda x: x[1], reverse=True)
        return sorted_combos[:beam_size]

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

        for b in buffer:
            print("b: ", b)

        # initialize automata
        parser = TransitionSystem(buffer)
        # print("parser just initialized: ", parser)

        print("parser initialized buffer 0th el and -1th element: ", parser.buffer[0], " ", parser.buffer[-1])
            
        
        #diverge train and eval here
        if gold_tree is not None:

            losses = []

            while not parser.is_done():
                # print("parser in train: ", parser)
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

                # print("action in train: ", self.id_to_action[next_action])
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
            parsers = [[parser, 0.0, 1]] #for us, it's parsers (?) [[parser, score:Float, stepsTaken:Int]] and the parser is not gonna be an actual list---we are not storing a sequence
            
            print("parser buffer 0th el and -1th element: ", parser.buffer[0], " ", parser.buffer[-1])
            
            # walk over each step in sequence 
            while len(parsers_done) < 5:
            # while not parser.is_done():
                # print("parsers in dev: ", parsers)
                # if len(parsers[0][0].stack) > 0:
                #     print("here: ", parsers[0][0].stack[0], " ", parsers[0][0].stack[0].embedding, " ", len(parsers[0][0].stack))
                # else:
                #     print("buffer here: ", parsers[0][0].buffer[0], " ", parsers[0][0].buffer[0].embedding)
                all_candidates = list() #here will be all parser candidates (previous parser updated with current steps)
		        # expand each current candidate
                for i in range(len(parsers)):
                    # for every previously found sequence, get the seq and its score (we'll update them with new scores)
                    parser, score, steps = parsers[i] #this is one previous parser
                    #we want to see several ways of how we can update it
                    # print("parser in dev: ", parser)
                    print("parser inside parsers loop buffer 0th el and -1th element: ", parser.buffer[0], " ", parser.buffer[-1])
            
                    
                    state_features = self.make_features(parser)
                    # legal actions for current parser
                    legal_actions = parser.all_legal_actions()
                    # predict next action, label, and direction
                    action_scores = self.action_classifier(state_features).unsqueeze(dim=0)
                    # print("action scores: ", action_scores)
                    legal_action_scores = self.legal_action_scores(legal_actions, action_scores)
                    # print("legal action scores: ", legal_action_scores)
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

                    #
                    #get softmax scores for all of those classifier scores, convert scores to log probabilities, and then sum up
                    #be careful: keep track of how many steps i took--> bc need to normalize at the end (normalizing during doesnt make sense bc by then, they have taken the same# of steps)
                    top_k_combos = self.getTopNCombos(legal_action_scores, label_scores, direction_scores, self.beam_size) #returns tuples of ((indices of action/label/direction), score for the combo)
                    #make sure choosing new top parsers from ALL previous ones, not keeping n from each previous parser

                    for combo in top_k_combos:
                        print(combo)
                        next_action = combo[0]
                        # print("next act from combo ", next_action)
                        print("parser before deepcopy buffer 0th el and -1th element: ", parser.buffer[0], " ", parser.buffer[0].embedding, " ", parser.buffer[-1])
            
                        # parser_cand = deepcopy(parser)
                        parser_cand = TransitionSystem.fill_out(parser, parser.buffer, parser.stack)

                        print("parser cand buffer 0th el and -1th element: ", parser_cand.buffer[0].embedding, " ", parser_cand.buffer[-1])
                        # take the next parser step
                        #clone current parser and apply the step to the clone(copy) ---use deepcopy, so that we don't apply all the candidate steps to the same parser
                        #the steps should apply to the copies of the previous parser

                        action=self.id_to_action[next_action[0]]
                        print("action: ", action)
                        label=self.id_to_label[next_action[1]]
                        print("label: ", label)
                        direction=self.id_to_direction[next_action[2]]
                        print("direction: ", direction)

                        
                        


                        print("parser cand buffer before action: ", parser_cand.buffer[0].embedding)
                        parser_cand.take_action(
                            action=self.id_to_action[next_action[0]],
                            label=self.id_to_label[next_action[1]],
                            direction=self.id_to_direction[next_action[2]],
                            reduce_fn=self.merge_embeddings,
                        )

                        # print("parser cand buffer 0th el and -1th element after action: ", parser_cand.buffer[0], " ", parser_cand.buffer[-1])
                        
                        if len(parser_cand.stack) > 0:
                            print("parser cand stack after action: ", parser_cand.stack)

                        all_candidates.append([parser_cand, score + combo[1], steps + 1]) #this is the new parser after the action has been taken with the score updated

                #now we have several parse/score candidates
                #get top scoring parsers (remember to incorporate previous score)
                #should we normalize at every step? - no
                # print("all candidates: ", all_candidates)

                #sort candidates by score

                sorted_candidates = sorted(all_candidates, key = lambda x: x[1], reverse=True)

                #now top n candidates are the new parsers

                parsers = sorted_candidates[:self.beam_size]

                # print("parsers: ", parsers, " ", len(parsers))
                for parser in parsers:
                    # print("one parser: ", parser)
                    # print("par stack: ", parser[0].stack[0].embedding)
                    # print("parser stack: ", parser[0].stack)
                    # print("parser buffer: ", len(parser[0].buffer))
                    if len(parser[0].buffer) == 0 and len(parser[0].stack) == 1:
                        # print("parser done")
                        parsers_done.append(parser)
                        parsers.remove(parser)

                #out of all the top scoring parsers, check if any are done? make sure this is in the right place
                # if parser.is_done:
                #     parsers_done.append((parser, )


        #out of done parsers (normalize scores by length of corresponding sequence---the number of actions), choose the highest prob one (NORMALIZE)

        #here, get the results from the highest scoring of the done parsers

        #likely to cause bugs - if some finish, need to decrease the size of beam

            #normalize scores
            # for parser in parsers:
            #     print("parser: ", parser)
            #     score = float(parser[1])/parser[2] 
        # returns the TreeNode for the tree root


        if gold_tree is None:
            # print("parsers done: ", parsers_done)
            #normalize done parsers

            normalized_parsers = []
            for parser in parsers_done:
                score = float(parser[1])/parser[2] 
                normalized_parsers.append(parser[0], score)

            parser = sorted(normalized_parsers, key = lambda x: x[1], reverse=True)[0]


        predicted_tree = parser.get_result()
        outputs = (predicted_tree,)

        # are we training?
        if gold_tree is not None:
            loss = sum(losses) / len(losses)
            outputs = (loss,) + outputs

        return outputs