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
            # print(tens.children)
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
        # print("scores  in legal action scores: ", scores)
        # some actions are illegal, beware
        action_ids = [self.action_to_id[a] for a in actions]
        mask = torch.ones_like(scores) * -inf
        mask[action_ids] = 0

        masked_scores = scores + mask
        return masked_scores

    def getScoreCombinations(self, action_scores, label_scores, direction_scores):
        combos = []
        #todo: update shapes from hardcoded to shapes of args
        summable_action_scores = torch.cat([action_scores.squeeze(0)[i].repeat(19*3) for i in range(action_scores.squeeze(0).shape[0])]) 
        print("summable action scores: ", summable_action_scores)
        summable_label_scores = label_scores.squeeze(dim=0).repeat(2*3)
        print("summable_label_scores: ", summable_label_scores)
        direction_scores_summable_with_labels = torch.cat([direction_scores.squeeze(dim=0)[i].repeat(19) for i in range(direction_scores.squeeze(dim=0).shape[0])])
        summable_direction_scores = torch.cat([direction_scores_summable_with_labels, direction_scores_summable_with_labels]) 
        print("summable direction scores: ", summable_direction_scores)

        all_scores = (summable_action_scores + summable_label_scores + summable_direction_scores).view(2,19,3)
        print("all scores: ", all_scores)
        return all_scores

    

    def forward(self, edus, gold_tree=None):
        # print("ONE TREE")
        # tokenize edus
        # print("beam: ", self.beam_size)
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

        # for b in buffer:
        #     print("b: ", b)

        # initialize automata
        parser = TransitionSystem(buffer, None)
        # print("parser just initialized: ", parser)

        # print("parser initialized buffer 0th el and -1th element: ", parser.buffer[0], " ", parser.buffer[-1])
            
        
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
            
            # print("parser buffer 0th el and -1th element: ", parser.buffer[0], " ", parser.buffer[-1])
            
            # walk over each step in sequence 
            # for i in range(0, 20):
            while len(parsers_done) < self.beam_size:
               
                all_candidates = list() #here will be all parser candidates (previous parser updated with current steps)
		        # expand each current parser and store all candidates in all_candidates (to later pick top k to append to parsers for next step)
                
                for i in range(len(parsers)):
                    
                    # for every parser from last step, get the parser and its score (we'll update them with new scores)
                    parser, score, steps = parsers[i] #this is one previous parser
                    #we want to get top k parsers that highest scored action-label-direction combinations will produce
                    
                    state_features = self.make_features(parser)
                    # legal actions for current parser
                    legal_actions = parser.all_legal_actions()
                    
                    # get next action, label, and direction scores
                    # we will want to get top k action/label/dir combos and their scores to decide which steps to take

                    
                    # marco: get softmax scores for all of those classifier scores, convert scores to log probabilities, and then sum up
                    #be careful: keep track of how many steps i took--> bc need to normalize at the end (normalizing during doesnt make sense bc by then, they have taken the same# of steps)
                    

                    logSoftmax = nn.LogSoftmax(dim=0)
                    
                    action_scores = self.action_classifier(state_features)
                    # logSoftmaxedActionScores = logSoftmax(action_scores)
                    # print("action scores: ", action_scores)
                    # print("log softmaxed scores: ", logSoftmaxedActionScores)

                    legal_action_scores = logSoftmax(self.legal_action_scores(legal_actions, action_scores))
                    print("legal act scores: ", legal_action_scores)
                    label_scores = logSoftmax(self.label_classifier(state_features))
                    print("label scores: ", label_scores)
                    direction_scores = logSoftmax(self.direction_classifier(state_features))
                    print("dir scores: ", direction_scores)
     
                    
                    combo_scores = self.getScoreCombinations(legal_action_scores, label_scores, direction_scores) #returns tuples of ((indices of action/label/direction), score for the combo)
                    # print(combo_scores)
                    #print("combo scores shape: ", combo_scores.shape)
                    #make sure choosing new top parsers from ALL previous ones, not keeping n from each previous parser

                    action_ids = [self.action_to_id[a] for a in legal_actions]
                    #todo: read on pass by value/reference
                    for i in action_ids:
                        for j in range(len(label_scores)):
                            for k in range(len(direction_scores)):
                                print(i, " ", j, " ", k)
                                combo_score = combo_scores[i][j][k]
                                print("combo score: ", combo_score)
                                # print("parser before deepcopy buffer 0th el and -1th element: ", parser.buffer[0], " ", parser.buffer[0].embedding, " ", parser.buffer[-1])
                                #print("parser before deepcopy, bufffer and stack len: ", len(parser.buffer), " ", len(parser.stack)) 
                                # parser_cand = deepcopy(parser)
                                # new_buffer = list(parser.buffer)
                                # new_stack = list(parser.stack)
                                parser_cand = TransitionSystem(parser.buffer, parser.stack) #need to send a copy, grab them as list
                                #print("parser cand: ", parser_cand)
                                #print("parser cand bufffer and stack len before action: ", len(parser_cand.buffer), " ", len(parser_cand.stack))
                                # print("parser cand buffer 0th el and -1th element: ", parser_cand.buffer[0].embedding, " ", parser_cand.buffer[-1])
                                # take the next parser step
                                #clone current parser and apply the step to the clone(copy) ---use deepcopy, so that we don't apply all the candidate steps to the same parser
                                #the steps should apply to the copies of the previous parser

                                action=self.id_to_action[i]
                                print("action: ", action)
                                label=self.id_to_label[j]
                                print("label: ", label)
                                direction=self.id_to_direction[k]
                                print("direction: ", direction)

                                
                                


                                # print("parser cand buffer before action: ", parser_cand.buffer[0].embedding)
                                parser_cand.take_action(
                                    action=action,
                                    label=label,
                                    direction=direction,
                                    reduce_fn=self.merge_embeddings,
                                )

                                #print("parser cand bufffer and stack len after action: ", len(parser_cand.buffer), " ", len(parser_cand.stack))
                                # print("parser cand buffer 0th el and -1th element after action: ", parser_cand.buffer[0], " ", parser_cand.buffer[-1])
                                
                                # if len(parser_cand.stack) > 0:
                                #     print("parser cand stack after action: ", parser_cand.stack)

                                all_candidates.append([parser_cand, score + combo_score, steps + 1]) #this is the new parser after the action has been taken with the score updated

                #now we have several parse/score candidates
                #get top scoring parsers (remember to incorporate previous score)
                #should we normalize at every step? - no
                #print("all candidates: ", all_candidates)

                #sort candidates by score

                sorted_candidates = sorted(all_candidates, key = lambda x: x[1], reverse=True)
                print("sorted candidates: ", sorted_candidates)
                #now top n candidates are the new parsers

                parsers = sorted_candidates[:self.beam_size]

                print("sorted parsers: ", parsers)

                # print("parsers: ", parsers, " ", len(parsers))
                for parser in parsers:
                    # print("one parser: ", parser)
                    # print("par stack: ", parser[0].stack[0].embedding)
                    # print("parser stack: ", parser[0].stack)
                    # print("parser buffer: ", len(parser[0].buffer))
                    if len(parser[0].buffer) == 0 and len(parser[0].stack) == 1: #maybe this will work now with is_done
                        print("parser done")
                        parsers_done.append(parser)
                        parsers.remove(parser)
                        # self.beam_size = self.beam_size - 1 - let's not do this

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


           
            #print("parsers done before normalization: ", parsers_done)
            #normalize done parsers

            normalized_parsers = []
            
            for parser in parsers_done:
                score = float(parser[1])/parser[2] 
                normalized_parsers.append((parser[0], score))

            #print("normalized parsers: ", normalized_parsers)
            parser = sorted(normalized_parsers, key = lambda x: x[1], reverse=True)[0][0] #call max, not sort - easier to decipher when reading
            #print(parser)


        predicted_tree = parser.get_result()
        outputs = (predicted_tree,)

        # are we training?
        if gold_tree is not None:
            loss = sum(losses) / len(losses)
            outputs = (loss,) + outputs

        return outputs