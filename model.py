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
        summable_action_scores = (torch.cat([action_scores.squeeze(0)[i].repeat(19*3) for i in range(action_scores.squeeze(0).shape[0])])).view(2,19,3)
        print("summable action scores: ", summable_action_scores)
        label_scores_half = torch.cat([label_scores.squeeze(0)[i].repeat(3) for i in range(label_scores.squeeze(0).shape[0])])
        summable_label_scores = torch.cat([label_scores_half, label_scores_half]).view(2,19,3)
        print("summable_label_scores: ", summable_label_scores)
        summable_direction_scores = direction_scores.squeeze(0).repeat(2*19).view(2,19,3)
        print("summable direction scores: ", summable_direction_scores)

        all_scores = summable_action_scores + summable_label_scores + summable_direction_scores
        #print("all scores: ", all_scores)
        return all_scores

    

    def forward(self, train, edus, gold_tree=None):
        # print("ONE TREE")
        # tokenize edus
        # print("beam: ", self.beam_size)

        
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

        # for b in buffer:
        #     print("b: ", b)

        # initialize automata
        parser = TransitionSystem(buffer, None)
        # print("parser just initialized: ", parser)

        if train==False:

            gold_sequence = parser.gold_path(gold_tree)
            print("Steps:")
            for step in gold_sequence:
                print(step)
            # nltked_gold_tree = gold_tree.to_nltk
            # print(nltked_gold_tree)
        

        # print("parser initialized buffer 0th el and -1th element: ", parser.buffer[0], " ", parser.buffer[-1])
        losses = []    
        
        #diverge train and eval here
        if train==True:

            

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

            # for every highest scored combo of action/label/dir, take an action and store the parser and cur score (make sure to update parser score)
            #OR
            # for every combo of action/label/dir (maybe also choose top k?), calc scores with previous parsers, then take action and store the parser and cur score
            parsers_done = []
            parsers = [[parser, 0.0, 1, "init_step"]] #for us, it's parsers (?) [[parser, score:Float, stepsTaken:Int, [seq of transition scores that led to this]]] and the parser is not gonna be an actual list---we are not storing a sequence
            
            # print("parser buffer 0th el and -1th element: ", parser.buffer[0], " ", parser.buffer[-1])
            
            # walk over each step in sequence 
            # for i in range(0, 20):
            while len(parsers_done) < self.beam_size:
               
                all_candidates = list() #here will be all parser candidates (previous parser updated with current steps)
		        # expand each current parser and store all candidates in all_candidates (to later pick top k to append to parsers for next step)
                
                for i in range(len(parsers)):
                    
                    # for every parser from last step, get the parser and its score (we'll update them with new scores)
                    print("cur parser: ", parsers[i])
                    
                    parser, score, steps, tr_scores_str = parsers[i] #this is one previous parser

                    print("tr scores: ", tr_scores_str , type(tr_scores_str))
                    #we want to get top k parsers that highest scored action-label-direction combinations will produce
                    
                    print("-------------------\nStarting a parser, step:", steps)
                    print("-------------------")
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
                    print("action scores: \n", action_scores)
                    # print("log softmaxed scores: ", logSoftmaxedActionScores)

                    legal_action_scores = logSoftmax(self.legal_action_scores(legal_actions, action_scores))
                    print("legal act scores logged: \n", legal_action_scores)

                    label_scores_no_log = self.label_classifier(state_features)
                    print("label scores: \n", label_scores_no_log)
                    label_scores = logSoftmax(label_scores_no_log)
                    print("label scores logged: \n", label_scores)
                    direction_scores_no_log = self.direction_classifier(state_features)
                    print("direction scores: \n", direction_scores_no_log)
                    direction_scores = logSoftmax(direction_scores_no_log)
                    print("dir scores logged: \n", direction_scores)
     
                    
                    combo_scores = self.getScoreCombinations(legal_action_scores, label_scores, direction_scores) #returns tuples of ((indices of action/label/direction), score for the combo)
                    print(combo_scores)
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
                                #print("action: ", action)
                                label=self.id_to_label[j]
                                #print("label: ", label)
                                direction=self.id_to_direction[k]
                                #print("direction: ", direction)

                                
                                


                                # print("parser cand buffer before action: ", parser_cand.buffer[0].embedding)
                                parser_cand.take_action(
                                    action=action,
                                    label=label,
                                    direction=direction,
                                    reduce_fn=self.merge_embeddings,
                                )

                                #print("parser cand bufffer and stack len after action: ", len(parser_cand.buffer), " ", len(parser_cand.stack))
                                # print("parser cand buffer 0th el and -1th element after action: ", parser_cand.buffer[0], " ", parser_cand.buffer[-1])
                                print("parser cand and score: ", parser_cand, " ", score + combo_score)
                                # if len(parser_cand.stack) > 0:
                                #     print("parser cand stack after action: ", parser_cand.stack)
                                
                                all_candidates.append([parser_cand, score + combo_score, steps + 1, tr_scores_str + "+" + str(combo_score) + "_act:" + str(i) + "-label:" + str(j) + "-dir:" + str(k)]) #this is the new parser after the action has been taken with the score updated

                #now we have several parse/score candidates
                #get top scoring parsers (remember to incorporate previous score)
                #should we normalize at every step? - no
                # print("all candidates: ", all_candidates)

                #sort candidates by score

                sorted_candidates = sorted(all_candidates, key = lambda x: x[1], reverse=True)
                print("------------\nsorted candidates: \n-------------")
                for cand in sorted_candidates:
                    print(cand)
                print("+++++++++")
                #now top n candidates are the new parsers

                parsers = sorted_candidates[:self.beam_size]

                
                # print("parsers: ", parsers, " ", len(parsers))
                print("sorted parsers:")
                for parser in parsers:
                    print("one parser: ", parser)
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
            parser_with_scores = sorted(normalized_parsers, key = lambda x: x[1], reverse=True)[0] #call max, not sort - easier to decipher when reading
            parser = parser_with_scores[0] 
            print("final parser (with scores): ", parser_with_scores)


        predicted_tree = parser.get_result()
        predicted_step_sequence = parser.gold_path(predicted_tree)


        if train == False:
            print("predicted tree: ", predicted_step_sequence)
            for step in predicted_step_sequence: #make sure it's okay to use this method on the predicted tree
                print("step: ", step)
        outputs = (predicted_tree,)
        # print("outputs: ", outputs)

        # are we training?
        if train:
            loss = sum(losses) / len(losses)
            outputs = (loss,) + outputs

        return outputs