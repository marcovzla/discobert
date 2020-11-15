import torch
from torch import nn
from transformers import *
from rst import TreeNode, iter_nuclearity_spans
from transition_system import TransitionSystem
from treelstm import TreeLstm
import config
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

inf = float('inf')

def reduceForEasyFirst(child1, child2, label=None, direction=None, reduce_fn=None, rel_embedding=None):
    lhs = child1
    rhs = child2
    emb = None if reduce_fn is None else reduce_fn(lhs.embedding, rhs.embedding, rel_embedding) 
    # print(emb)

    node = TreeNode(children=[lhs, rhs], label=label, direction=direction, embedding=emb)
    node.calc_span()
    return node

def loss_fn_for_labels(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

def loss_fn(outputs, targets):
    return nn.MultiMarginLoss()(outputs, targets)

# def loss_fn_on_labels(outputs, targets, label_weights_tensor):
#     return nn.CrossEntropyLoss(weight=label_weights_tensor)(outputs, targets)

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
        self.hidden_size = config.HIDDEN_SIZE
        self.relation_label_hidden_size = config.RELATION_LABEL_HIDDEN_SIZE
        self.direction_hidden_size = config.DIRECTION_HIDDEN_SIZE
        self.include_relation_embedding = config.INCLUDE_RELATION_EMBEDDING
        self.include_direction_embedding = config.INCLUDE_DIRECTION_EMBEDDING
        # init model
        self.tokenizer = config.TOKENIZER
        self.encoding = config.ENCODING
        if self.encoding == 'bert' or self.encoding == 'bert-large':
            self.tokenizer = config.TOKENIZER
            self.encoder = BertModel.from_pretrained(self.bert_path)
        elif self.encoding == 'roberta':
            self.tokenizer = config.TOKENIZER
            self.encoder = RobertaModel.from_pretrained(self.bert_path)
        elif self.encoding == 'openai-gpt':
            self.tokenizer = config.TOKENIZER
            self.encoder = OpenAIGPTModel.from_pretrained(self.bert_path)
        elif self.encoding == 'gpt2':
            self.tokenizer = config.TOKENIZER
            self.encoder = GPT2Model.from_pretrained(self.bert_path)
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        elif self.encoding == 'xlnet':
            self.tokenizer = config.TOKENIZER
            # self.encoder = SentenceTransformer("roberta-base-nli-stsb-mean-tokens")
            self.encoder = XLNetModel.from_pretrained(self.bert_path)
        elif self.encoding == 'distilbert':
            self.tokenizer = config.TOKENIZER
            self.encoder = DistilBertModel.from_pretrained(self.bert_path)
        elif self.encoding == 'albert':
            self.tokenizer = config.TOKENIZER
            self.encoder = AlbertModel.from_pretrained(self.bert_path)
        elif self.encoding == 'ctrl':
            self.tokenizer = config.TOKENIZER
            self.encoder = CTRLModel.from_pretrained(self.bert_path)
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        if config.USE_ATTENTION:
            self.attn1 = nn.Linear(self.encoder.config.hidden_size, 100)
            self.attn2 = nn.Linear(100, 1)
        self.betweenAttention = nn.Tanh()
        self.bert_drop = nn.Dropout(self.dropout)
        self.project = nn.Linear(self.encoder.config.hidden_size, self.hidden_size)
        self.missing_node = nn.Parameter(torch.rand(self.hidden_size, dtype=torch.float))
        self.separate_action_and_dir_classifiers = config.SEPARATE_ACTION_AND_DIRECTION_CLASSIFIERS
        self.action_classifier = nn.Linear(4 * self.hidden_size, len(self.id_to_action))
        self.label_classifier = nn.Linear(4 * self.hidden_size, len(self.id_to_label))
        if self.separate_action_and_dir_classifiers==True:
            self.direction_classifier = nn.Linear(4 * self.hidden_size, len(self.id_to_direction))
        # self.merge_layer = nn.Linear(2 * self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.treelstm = TreeLstm(self.hidden_size // 2, self.include_relation_embedding, self.include_direction_embedding, self.relation_label_hidden_size, self.direction_hidden_size)
        # self.relu = nn.ReLU()
        # self.lstm = nn.LSTM(self.encoder.config.hidden_size, 200, bidirectional=True, batch_first=True)
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

    def make_features(self, parser, incl_buffer):
        """Gets a parser and returns an embedding that represents its current state.
        The state is represented by the concatenation of the embeddings corresponding
        to the two nodes on the top of the stack and the next node in the buffer.
        """
        buffer_size = len(parser.buffer) / (len(parser.buffer) + len(parser.stack))
        # print(buffer_size)
        # buffer_feature = torch.from_numpy(np.array(buffer_size)).to(self.device)
        buffer_feature = torch.as_tensor(buffer_size).to(self.device)
        # if len(parser.stack) > 0:
        #     print("parser: ", parser.stack[0].embedding, parser.stack[0].embedding.shape)
        # else:
        #     print("parser: ", parser.buffer[0].embedding, parser.buffer[0].embedding.shape)

        # print("buffer feature type: ", buffer_feature.type)
        # print("buffer feature: ", buffer_feature.shape)
        
        s2 = self.missing_node if len(parser.stack) < 3 else parser.stack[-3].embedding
        s1 = self.missing_node if len(parser.stack) < 2 else parser.stack[-2].embedding
        s0 = self.missing_node if len(parser.stack) < 1 else parser.stack[-1].embedding
        if incl_buffer:
            # print("LEN BUFFER: ", len(parser.buffer))
            b0 = self.missing_node if len(parser.buffer) < 1 else parser.buffer[0].embedding
            b1 = self.missing_node if len(parser.buffer) < 2 else parser.buffer[1].embedding
            b2 = self.missing_node if len(parser.buffer) < 3 else parser.buffer[2].embedding

            
        else:
            b0 = self.missing_node
            b1 = self.missing_node
            b2 = self.missing_node
        return torch.cat([s2, s1, s0, b0, b1, b2, buffer_feature.unsqueeze(dim=0)])

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

    def forward(self, edus, gold_tree=None, annotation=None, class_weights=None):
        print("DOC START")

        print(gold_tree)
        if gold_tree is not None:
            print("HERE")
            all_gold_spans = [f'{x}' for x in iter_nuclearity_spans(gold_tree.get_nonterminals())] 
            print(all_gold_spans)
        # BERT model returns both sequence and pooled output
        if self.encoding == "bert" or self.encoding == "bert-large":
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

            
        # the other models we test, do not have a pooled output
        else:
            # tokenize edus 
            batched_encodings = self.tokenizer(edus, padding=True, return_attention_mask=True, return_tensors='pt').to(self.device) #add special tokens is true by default
            ids = batched_encodings['input_ids']
            attention_mask = batched_encodings['attention_mask']
            # encode edus
            sequence_output = self.encoder(ids, attention_mask=attention_mask, output_hidden_states=True)[0]


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
            # gpt1 and gpt2 have not been trained with a cls (the beginning of sequence/classification token),
            # so to get the representation of the edu, take the mean of the token embeddings
            if self.encoding == "openai-gpt" or self.encoding == "gpt2":
                enc_edus = self.bert_drop(torch.mean(sequence_output, dim=1))
            else:
                enc_edus = self.bert_drop(sequence_output[:,0,:])
                # enc_edus = self.bert_drop(sequence_output[:,:4,:])
                # print("=>", enc_edus.shape)
                # enc_edus = torch.cat((enc_edus[:, 0, :], enc_edus[:, 1, :], enc_edus[:, 2, :], enc_edus[:, 3, :]), dim=1)
                # print("here", enc_edus.shape)
                # enc_edus = self.bert_drop(sequence_output)

        # enc_edus, _ = self.lstm(enc_edus)
        enc_edus = self.project(enc_edus) 
        # enc_edus, _ = self.lstm(enc_edus.unsqueeze(dim=0))
        # enc_edus_shape = enc_edus.shape
        # batch, seq_len, hidden_size = enc_edus.shape
        # zeros = torch.zeros((batch, seq_len, hidden_size//2)).to(self.device)

        # enc_edus = enc_edus.view(batch, seq_len, 2, hidden_size//2)
        # print("viewed: ", enc_edus.shape)
        # enc_edus = torch.mean(enc_edus, dim=2)
        # enc_edus = torch.cat((enc_edus, zeros), dim=2)
        # enc_edus = self.bert_drop(enc_edus[:,0,:])
        # enc_edus = torch.mean(enc_edus, dim=1)
        # enc_edus = enc_edus.squeeze(dim=0)



        # make treenodes
        buffer = []
        for i in range(enc_edus.shape[0]):
            buffer.append(TreeNode(leaf=i, embedding=enc_edus[i], text=edus[i]))

        # initialize automata
        parser = TransitionSystem(buffer)

        losses = []
        all_current_nodes = parser.stack + parser.buffer
        
        
        # get scores for every possible reduction (that is the score for reducing each pair of adjacent nodes)
        while len(all_current_nodes) > 1:
            print("len all current nodes: ", len(all_current_nodes))
            print("ALL CURRENT NODES: ")
            for node in all_current_nodes:
                print("->", node)
                print("->>", f'{node.span}::{node.direction}')

            # during training, we'll be getting both max gold score ...
            gold_max_score = 0
            gold_max_node = None # id of the first node in reduce
            gold_max_action = None # action for the max score we've seen so far
            gold_max_label = None

            if gold_tree is not None:
                # ... and max predicted score (with max gold and max predicted can---and ideally should---be the same)
                max_score = 0
                max_node = None # id of the first node in reduce
                max_action = None # action for the max score we've seen so far
                max_label = None
                second_best_predicted_score = 0

            legal_actions = config.ID_TO_ACTION #ReduceL, ReduceR
            
            # calculate the scores 
            for i, node1 in enumerate(all_current_nodes):
                print(f'i: {i}')
                
                if not i >= len(all_current_nodes) - 1:
                    # incl two more nodes for features
                    node2 = all_current_nodes[i + 1].embedding
                    node0 = self.missing_node if i == 0 else all_current_nodes[i - 1].embedding
                    node3 = self.missing_node if i + 2 >= len(all_current_nodes) - 1 else all_current_nodes[i + 2].embedding
                    state_features = torch.cat([node0, node1.embedding, node2, node3])

                    #hinge loss 
                    #the predicted and the next highest 
                    #use "is valid"
                    #add the label loss
                    

                    
                    action_scores = self.action_classifier(state_features).unsqueeze(dim=0)
                    
                    print("act scores: ", action_scores)
                    curr_score = torch.max(action_scores).item()
                    print("curr score: ", curr_score)
                    curr_best_action = torch.argmax(action_scores).item()
                    print("cur best action: ", curr_best_action)
                    

                    label_scores = self.label_classifier(state_features).unsqueeze(dim=0)
                    next_label = label_scores.argmax().unsqueeze(0)
                    label = self.id_to_label[next_label]
                    print(f"label: {label}")
                    # since there are only two action and the best one is already gonna be considered as a candidate for reduction,
                    # we only need to check if the min score action is a good gold candidate
                    if gold_tree is not None:
                        second_best_score = torch.min(action_scores).item()
                        if second_best_score > gold_max_score:
                            second_best_action = torch.argmin(action_scores).item()
                            if second_best_action == "reduceL":
                                direction = "RightToLeft"
                            else:
                                direction = "LeftToRight"
                            potential_gold_node = reduceForEasyFirst(node1, all_current_nodes[i + 1], label=None, direction=direction, reduce_fn=None) 
                            if f'{potential_gold_node.span}::{potential_gold_node.direction}' in all_gold_spans:
                                gold_max_score = second_best_score
                                gold_max_node = i
                                gold_max_action = second_best_action
                                gold_max_label = label

                    
                    if curr_score > max_score:
                        second_best_predicted_score = max_score #update second best score
                        max_score = curr_score
                        max_node = i
                        max_action = legal_actions[curr_best_action]
                        max_label = label
                        print("max action: ", max_action)
                        print("max score: ", max_score)
                        print("max nodes: ", max_node)
                else:
                    # when reach the point when there is no i + 1
                    print("BREAK")
                    break
                        
            print("REACHED HERE")

            


            # check if max score reduction is in gold
            if max_action == "reduceL":
                direction = "RightToLeft"
            else:
                direction = "LeftToRight"
            new_node = reduceForEasyFirst(all_current_nodes[max_node], all_current_nodes[max_node + 1], label=max_label, direction=direction, reduce_fn=self.merge_embeddings)  
            new_node_as_string = f'{new_node.span}::{new_node.direction}'

            if gold_tree is not None:
                if new_node_as_string in all_gold_spans: 
                    new_current = []
                    new_current.extend(all_current_nodes[:max_node])
                    new_current.append(new_node_as_string)
                    new_current.extend(all_current_nodes[max_node+2:])
                else:

                    if gold_max_action == "reduceL":
                        direction = "RightToLeft"
                    else:
                        direction = "LeftToRight"
                    
                    new_node = reduceForEasyFirst(all_current_nodes[gold_max_node], all_current_nodes[gold_max_node + 1], label=gold_max_label, direction=direction, reduce_fn=self.merge_embeddings) 
                    new_current = []
                    new_current.extend(all_current_nodes[:max_node])
                    new_current.append(new_node)
                    new_current.extend(all_current_nodes[max_node+2:])
            else:
                new_current = []
                new_current.extend(all_current_nodes[:max_node])
                new_current.append(new_node_as_string)
                new_current.extend(all_current_nodes[max_node+2:])


            all_current_nodes = new_current
            top_two_predicted_scores = torch.FloatTensor([max_score, second_best_predicted_score]).to(self.device)
            action_loss = loss_fn(top_two_predicted_scores, torch.LongTensor([gold_max_score]).to(self.device))
            # loss_on_labels = loss_fn_for_labels(label_scores, gold_label)
            losses.append(action_loss)
        print(all_current_nodes[0].to_nltk())
        predicted_tree = all_current_nodes[0]

        outputs = (predicted_tree,)

        # are we training?
        if gold_tree is not None:
            loss = sum(losses) / len(losses)
            outputs = (loss,) + outputs

        return outputs





        
            



        #     new_current = []
            
        #     # print("new current: ", new_current)
        #     for i, node1 in enumerate(all_current_nodes):
        #         print(f"is i = max node?: {i} = {max_node}")
        #         if i == max_node:
                    
                    
        #             if max_action == "reduceL":
        #                 new_node = reduceForEasyFirst(node1, all_current_nodes[i + 1], label=label, direction="RightToLeft", reduce_fn=self.merge_embeddings) 
        #             else:
        #                 new_node = reduceForEasyFirst(node1, all_current_nodes[i + 1], label=label, direction="LeftToRight", reduce_fn=self.merge_embeddings) 
                    
        #             print("HERE!!: ", new_node.span, new_node.direction)
        #             new_node_as_string = f'{new_node.span}::{new_node.direction}'
        #             if new_node_as_string in all_gold_spans:
        #                 print("new node in gold")
        #                 new_current.append(new_node)
        #                 # for node in all_current_nodes[i+2:]:
        #             #     new_current.append(node)
        #                 all_current_nodes = new_current
        #             else:
        #                 print("new node not in gold: ", new_node_as_string, all_gold_spans)
        #                 break
                    
        #             # print(f"after reduce: ", new_node.to_nltk())
        #             break


        #             # new node
        #             # add this node
        #             # add the remaining nodes i + 2 (?)
        #             # break
        #         else:
        #             new_current.append(node1)
        #     print("new new current len: ", len(new_current))

        # print(all_current_nodes[0].to_nltk())


                


            



                    


        #     # # the boolean in 'make_features' is whether or not to include the buffer node as a feature
        #     # state_features = self.make_features(parser, True)
        #     # # legal actions for current parser
        #     # legal_actions = parser.all_legal_actions()
        #     # # predict next action, label, and, if predicting actions and directions separately, direction based on the stack and the buffer
        #     # action_scores = self.action_classifier(state_features).unsqueeze(dim=0)
        #     if gold_tree is not None:
        #         label_weights_tensor = torch.Tensor.float(torch.from_numpy(class_weights).to(self.device))
        #     # make a new set of features without the buffer for label classifier for any of the reduce actions
        #     if self.id_to_action[self.best_legal_action(legal_actions, action_scores)].startswith("reduce"):
        #         state_features_for_labels = self.make_features(parser, False) 
        #         label_scores = self.label_classifier(state_features_for_labels).unsqueeze(dim=0)
        #         # label_scores = torch.mul(self.label_classifier(state_features_for_labels).unsqueeze(dim=0), label_weights_tensor)
        #     # for shift, use the stack + buffer features for label classifier
        #     else:
        #         label_scores = self.label_classifier(state_features).unsqueeze(dim=0)
        #         # label_scores = torch.mul(self.label_classifier(state_features).unsqueeze(dim=0), label_weights_tensor)
            
        #     # in the three classifier version, direction is predicted separately from the action
        #     if self.separate_action_and_dir_classifiers==True:
        #         direction_scores = self.direction_classifier(state_features).unsqueeze(dim=0)
        #     # are we training?
        #     if gold_tree is not None:
        #         gold_step = parser.gold_step(gold_tree)
        #         # unpack step
        #         gold_action = torch.tensor([self.action_to_id[gold_step.action]], dtype=torch.long).to(self.device)
        #         gold_label = torch.tensor([self.label_to_id[gold_step.label]], dtype=torch.long).to(self.device)
        #         if self.separate_action_and_dir_classifiers==True:
        #             gold_direction = torch.tensor([self.direction_to_id[gold_step.direction]], dtype=torch.long).to(self.device)
        #         # calculate loss
        #         loss_on_actions = loss_fn(action_scores, gold_action)
        #         loss_on_labels = loss_fn(label_scores, gold_label) 
        #         action_for_labels = self.best_legal_action(legal_actions, action_scores)
        #         if self.id_to_action[action_for_labels].startswith("reduce"): 
        #             loss_on_labels = loss_fn_on_labels(label_scores, gold_label, label_weights_tensor) 
        #         else:
        #             loss_on_labels = 0
        #         if self.separate_action_and_dir_classifiers==True:
        #             loss_on_direction = loss_fn(direction_scores, gold_direction)
        #             loss = loss_on_actions + loss_on_labels + loss_on_direction
        #         else:

        #             loss = loss_on_actions + loss_on_labels
        #         #     loss = loss_on_actions + 2 * loss_on_labels 
                           
        #         # store loss for later
        #         losses.append(loss)
        #         # teacher forcing
        #         next_action = gold_action
        #         next_label = gold_label
        #         if self.separate_action_and_dir_classifiers==True:
        #             next_direction = gold_direction
        #     else:
        #         next_action = self.best_legal_action(legal_actions, action_scores)
        #         # predict the label for any of the reduce actions
        #         if self.id_to_action[next_action].startswith("reduce"):
        #             next_label = label_scores.argmax().unsqueeze(0) #unsqueeze because after softmax the output tensor is tensor(int) instead of tensor([int]) (different from next_label in training)
        #         # there is no label to predict for shift
        #         else:
        #             next_label = torch.tensor(0, dtype=torch.long).to(self.device)          
        #         if self.separate_action_and_dir_classifiers==True:
        #             next_direction = direction_scores.argmax().unsqueeze(0)
                
            
        #     if self.include_relation_embedding:
        #         rel_emb = self.relation_embeddings(next_label)
        #         if self.include_direction_embedding:
        #             dir_emb = self.direction_embedding(next_direction)
        #             rel_dir_emb = torch.cat((rel_emb, dir_emb), dim=1)
        #         else:
        #             rel_dir_emb = rel_emb
        #     else:
        #         rel_dir_emb = None  

            
        #     action=self.id_to_action[next_action]
        #     # take the step
        #     parser.take_action(
        #         action=action,
        #         label=self.id_to_label[next_label] if action.startswith("reduce") else "None", # no label for shift 
        #         direction=self.id_to_direction[next_direction] if self.separate_action_and_dir_classifiers==True else None,
        #         reduce_fn=self.merge_embeddings,
        #         rel_embedding = rel_dir_emb
        #     )

        # # returns the TreeNode for the tree root
        # predicted_tree = parser.get_result()
        # if config.PRINT_TREES == True:
        #     # print(annotation)
        #     print("Document::" + str(annotation.docid) + "::" + str(predicted_tree.to_nltk()) + "\n")
        # outputs = (predicted_tree,)

        # # are we training?
        # if gold_tree is not None:
        #     loss = sum(losses) / len(losses)
        #     outputs = (loss,) + outputs

        # return outputs