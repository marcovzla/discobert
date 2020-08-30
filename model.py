import torch
from torch import nn
from transformers import *
from rst import TreeNode
from transition_system import TransitionSystem
from treelstm import TreeLstm
import config
import torch.nn.functional as F
from collections import namedtuple
from nltk.tokenize import sent_tokenize
import nltk

Annotation = namedtuple('Annotation', 'docid raw dis edus')

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
        self.segmentor_tag_to_id = config.SEGMENT_CLASS_TO_ID
        self.id_to_segmentor_tag = config.ID_TO_SEGMENT_CLASSES
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
        self.missing_node_for_boundaries = nn.Parameter(torch.rand(self.encoder.config.hidden_size, dtype=torch.float))
        self.separate_action_and_dir_classifiers = config.SEPARATE_ACTION_AND_DIRECTION_CLASSIFIERS
        
        
        self.segment_classifier = nn.Linear(768, 2)
        self.action_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_action))
        self.label_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_label))
        if self.separate_action_and_dir_classifiers==True:
            self.direction_classifier = nn.Linear(3 * self.hidden_size, len(self.id_to_direction))
        # self.merge_layer = nn.Linear(2 * self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.treelstm = TreeLstm(self.hidden_size // 2, self.include_relation_embedding, self.include_direction_embedding, self.relation_label_hidden_size, self.direction_hidden_size)
        # self.relu = nn.ReLU()
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

    def forward(self, edus, train, raw):

        # print(raw)
        sentences = sent_tokenize(raw)
        # print(sentences)
        # if train == False:
        #     print("EDUS: ", edus)

        #TODOS:
        # tokenize sent: 
        # MAYBE, can get gold from edus without cls token but get predictions based on sentences (raw)? will there be a mismatch?
        # ATTN: Braud calculated score exclusing first token of a sentence? or a document? prob doc. 
        # encode sent by sent
        # concat docs from the sent encodings
        # predictions will have sent boundaries 
        # check if i am not evaluating the right thing---maybe i am feeding the gold data somewhere during eval
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
        # todo: how to detokenize into edus to produce annotations? maybe what's discussed here: https://github.com/huggingface/transformers/issues/36
        # print("edus: ", edus)
        # ===================================================================================
        # =======This is what I should do with raw input, but not if I want to evaluate======
        # ===================================================================================
        # for running the segmenter without eval, can just do raw; need some flag somewhere
        # print("raw: ", raw)
        # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # tokenized_raw = tokenizer(raw.replace("\n", " "))
        # # ids = torch.tensor([e.input_ids for e in tokenized_raw], dtype=torch.long).to(self.device)
        # # attention_mask = torch.tensor([e.attention_mask for e in tokenized_raw], dtype=torch.long).to(self.device)
        # # token_type_ids = torch.tensor([e.token_type_ids for e in tokenized_raw], dtype=torch.long).to(self.device)
        # print("raw ids: ", torch.tensor(tokenized_raw.input_ids, dtype=torch.long).to(self.device).unsqueeze(dim=0).shape)
        # seq_output_from_raw, pooled_from_raw = self.encoder( #sequence_output: [edu, tok, emb]
        #         torch.tensor(tokenized_raw.input_ids, dtype=torch.long).to(self.device).unsqueeze(dim=0),
        #         attention_mask=torch.tensor(tokenized_raw.attention_mask, dtype=torch.long).to(self.device).unsqueeze(dim=0),
        #         token_type_ids=torch.tensor(tokenized_raw.token_type_ids, dtype=torch.long).to(self.device).unsqueeze(dim=0),
        #     )
        # ===================================================================================
        # print("tokenized raw: ", tokenized_raw)
        # print("encoded raw: ", seq_output_from_raw.shape)
        

        tokens_to_classify = []
        tokens_to_classify_bertified = []
        for sent in sentences:
            tokenized_sent_not_tensored = tokenizer(sent)
            tokenized_sent = tokenizer(sent, return_tensors='pt')
            sent_tokens = tokenizer.convert_ids_to_tokens(tokenized_sent_not_tensored.input_ids)
            # print("bert tokenized sent tokens: ",  tokenizer.convert_ids_to_tokens(tokenized_sent_not_tensored.input_ids))
            # print("bert tokenized send:", tokenized_sent_not_tensored)
            ids = torch.tensor(tokenized_sent.input_ids, dtype=torch.long).to(self.device)
            # print("ids: ", ids.shape)
            attention_mask = torch.tensor(tokenized_sent.attention_mask, dtype=torch.long).to(self.device)
            token_type_ids = torch.tensor(tokenized_sent.token_type_ids, dtype=torch.long).to(self.device)
            encoded_sent, pooled_output = self.encoder(ids, attention_mask, token_type_ids)
            # print('ecnoded sent: ', encoded_sent.shape)
            for i in range(encoded_sent.squeeze(dim=0).shape[0]):
                # print("tok sent input ids: ", tokenized_sent.input_ids[0])
                # print("cur token id: ", tokenized_sent.input_ids[0][i])
                # print("token in sent: ", sent_tokens[i])
                if sent_tokens[i] != '[SEP]' and sent_tokens[i] != '[CLS]':
                    tokens_to_classify.append(sent_tokens[i])
                    token_encoding = encoded_sent.squeeze(dim=0)[i].unsqueeze(dim=0)
                    # print("token encoding: ", token_encoding.shape)
                    tokens_to_classify_bertified.append(token_encoding)

        tokens_to_classify_bertified = torch.cat(tokens_to_classify_bertified, dim=0)
        # print("shape tokens to classify bertified: ", tokens_to_classify_bertified.shape)
        #EUREKA: can encode however it encodes but the gold and the predictions will only look at non-special symbols (problem is unk words)
        # looks like gold is created correctly
        ids = []
        attention_mask = []
        token_type_ids = []
        gold_tags = []
        gold_as_tokens = []

        for i in range(len(edus)):
            # print(edus[i])
            tokenized_edu = tokenizer(edus[i])
            ids.extend(tokenized_edu.input_ids[1:])
            attention_mask.extend(tokenized_edu.attention_mask[1:])
            token_type_ids.extend(tokenized_edu.token_type_ids[1:])
            
            for j in range(0, len(tokenized_edu.input_ids)): #have to skip cls---it's too much of a giveaway
                token_id = tokenized_edu.input_ids[j]
                # print("token id: ", token_id)
                token = tokenizer.convert_ids_to_tokens([token_id])
                # print("token: ", token)
                
                if token[0] != "[SEP]" and token[0] != "[CLS]":
                    # print("tok added to gold: ", token[0])
                    gold_as_tokens.append(token[0])
                    if j == 1:
                        gold_tags.append("B")   
                        # print("gold B")
                    else:
                        gold_tags.append("O")
                        # print("gold O")
        # print("len tok to classify: ", len(tokens_to_classify))
        # print("len tok to classify bertified: ", len(tokens_to_classify_bertified))
        # print("len gold tags: ", len(gold_tags))  
        # print("len gold as tokens: ", len(gold_as_tokens))
        
        # for i, j in enumerate(tokens_to_classify):
        #     print(gold_as_tokens[i], " ", gold_tags[i], tokens_to_classify[i])
        # ids = torch.tensor(ids, dtype=torch.long).to(self.device).unsqueeze(dim=0)
        # attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device).unsqueeze(dim=0)
        # token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(self.device).unsqueeze(dim=0)
        # print(ids.shape)
        # print(attention_mask.shape)
        # print(token_type_ids.shape)
        # print(ids)
        # print(attention_mask)
        # print(token_type_ids)
        # print(gold_tags)
        # print("len ids: ", ids.shape)
        # print("gold len: ", len(gold_tags))

        # seq_output_from_tokenized_edus, pooled_from_tokenized_edus = self.encoder( #sequence_output: [edu, tok, emb]
        #         ids,
        #         attention_mask=attention_mask,
        #         token_type_ids=token_type_ids,
        #     )

        # get gold data: for every 
        #     ids = torch.tensor([e.ids for e in encodings], dtype=torch.long).to(self.device)
        #     print("ids: ", ids.shape)
        #     attention_mask = torch.tensor([e.attention_mask for e in encodings], dtype=torch.long).to(self.device)
        #     token_type_ids = torch.tensor([e.type_ids for e in encodings], dtype=torch.long).to(self.device)

        
        # BERT model returns both sequence and pooled output
        # if self.encoding == "bert":
        #     # tokenize edus
        #     encodings = self.tokenizer.encode_batch(edus)
            # ids = torch.tensor([e.ids for e in encodings], dtype=torch.long).to(self.device)
            # print("ids: ", ids.shape)
            # attention_mask = torch.tensor([e.attention_mask for e in encodings], dtype=torch.long).to(self.device)
            # token_type_ids = torch.tensor([e.type_ids for e in encodings], dtype=torch.long).to(self.device)

        #     # encode edus
        #     sequence_output, pooled_output = self.encoder( #sequence_output: [edu, tok, emb]
        #         ids,
        #         attention_mask=attention_mask,
        #         token_type_ids=token_type_ids,
        #     )

            
        # # the other models we test, do not have a pooled output
        # else:
        #     # tokenize edus 
        #     batched_encodings = self.tokenizer(edus, padding=True, return_attention_mask=True, return_tensors='pt').to(self.device) #add special tokens is true by default
        #     ids = batched_encodings['input_ids']
        #     attention_mask = batched_encodings['attention_mask']
        #     # encode edus
        #     sequence_output = self.encoder(ids, attention_mask=attention_mask, output_hidden_states=True)[0]


        # if config.DROP_CLS == True:
        #     sequence_output = sequence_output[:, 1:, :] 
        #     attention_mask = attention_mask[:, 1:]
        
        
        # if config.USE_ATTENTION == True:
        #     after1stAttn = self.attn1(sequence_output)
        #     nonLinearity = self.betweenAttention(after1stAttn)
        #     after2ndAttn = self.attn2(nonLinearity)
        #     attention_mask = attention_mask.unsqueeze(dim=-1)
        #     masked_att = after2ndAttn * attention_mask
        #     attn_weights = F.softmax(masked_att, dim=1)
        #     attn_applied =  sequence_output * attn_weights #[9, 17, 768] [9, 17, 1] 
        #     summed_up = torch.sum(attn_applied, dim=1)
        #     enc_edus = self.bert_drop(summed_up)

        # else:
        #     # gpt1 and gpt2 have not been trained with a cls (the beginning of sequence/classification token),
        #     # so to get the representation of the edu, take the mean of the token embeddings
        #     if self.encoding == "openai-gpt" or self.encoding == "gpt2":
        #         enc_edus = self.bert_drop(torch.mean(sequence_output, dim=1))
        #     else:
        #         enc_edus = self.bert_drop(sequence_output[:,0,:])
            

        # enc_edus = self.project(enc_edus) 
        # encoded_docs = seq_output_from_raw.squeeze(dim=0)
        # encoded_docs = seq_output_from_tokenized_edus.squeeze(dim=0)
        # print("encoded docs shape: ", encoded_docs.shape)
        # make treenodes
        # buffer = []
        # for i in range(enc_edus.shape[0]):
        #     buffer.append(TreeNode(leaf=i, embedding=enc_edus[i]))

        # # initialize automata
        # parser = TransitionSystem(buffer)
        predictions = []
        new_edus = []
        losses = []
        potential_edu = []
        
        for i in range(tokens_to_classify_bertified.shape[0]):
            # print("shape of encoded docs: ", encoded_docs.shape)
            
            # print("enc word: ", encoded_do cs[i].shape)
            # encoding = self.make_boundary_features(i, encoded_docs)

            prediction_scores = self.segment_classifier(tokens_to_classify_bertified[i])
            # print("pred: ", prediction_scores)
            # print("max: ", torch.argmax(prediction_scores))
            predicted_tag = self.id_to_segmentor_tag[torch.argmax(prediction_scores)]
            # if train == False:
            #     gold_pred = gold_tags[i]
            #     print("token, prediction, gold: ", predicted_tag, " ", tokenizer.convert_ids_to_tokens([ids.squeeze(dim=0)[i]]), " ", gold_pred)
            
            # are we training?
            if train == True:
                # print("pred tag: ", predicted_tag)
                gold_pred = torch.tensor([self.segmentor_tag_to_id[gold_tags[i]]], dtype=torch.long).to(self.device)
                # print("gold pred: ", gold_pred)
                # predictions.append(gold_tags[i])
                # if self.id_to_segmentor_tag[gold_pred] != predicted_tag:
                #     print("WRONG PRED: ", self.id_to_segmentor_tag[gold_pred], " ", predicted_tag, " ", tokenizer.convert_ids_to_tokens([ids.squeeze(dim=0)[i]]))
                # else:
                #     print("CORRECT PRED: ", self.id_to_segmentor_tag[gold_pred], " ", predicted_tag, " ", tokenizer.convert_ids_to_tokens([ids.squeeze(dim=0)[i]]))
                predictions.append(predicted_tag)
                
                if gold_tags[i] == "B" or i == tokens_to_classify_bertified.shape[0] - 1:
                    # print("pot edu: ", potential_edu)
                    potential_edu_str = " ".join(potential_edu).replace(" ##", "") #todo: replace [SEP] with nothing; what's gonna happen to UNKs? we can't really get those back like this, can we? need to add some other dict to track them?
                    
                    # print("pot edu str: ", potential_edu_str)
                    if len(potential_edu_str) > 0:
                        new_edus.append(potential_edu_str)

                    # #TODO: what does this achieve?
                    # token_id = ids.squeeze(dim=0)[i]
                    # # print("token id: ", token_id)
                    # token = tokenizer.convert_ids_to_tokens([token_id])
                    
                    # print("token: ", token)
                    potential_edu = [] 
                elif gold_tags[i] == "O":
                    # print("ids: ", ids)
                    # print("i: ", i)
                    # token_id = ids.squeeze(dim=0)[i]
                    # # print("token id: ", token_id)
                    # token = tokenizer.convert_ids_to_tokens([token_id])
                    token = tokens_to_classify[i]
                    
                    # print("token: ", token)
                    potential_edu.append(token)
                # while not parser.is_done():
                # the boolean in 'make_features' is whether or not to include the buffer node as a feature
                # state_features = self.make_features(parser, True)
                # # legal actions for current parser
                # legal_actions = parser.all_legal_actions()
                # # predict next action, label, and, if predicting actions and directions separately, direction based on the stack and the buffer
                # action_scores = self.action_classifier(state_features).unsqueeze(dim=0)
                # make a new set of features without the buffer for label classifier for any of the reduce actions
                # if self.id_to_action[self.best_legal_action(legal_actions, action_scores)].startswith("reduce"):
                #     state_features_for_labels = self.make_features(parser, False) 
                #     label_scores = self.label_classifier(state_features_for_labels).unsqueeze(dim=0)
                # # for shift, use the stack + buffer features for label classifier
                # else:
                #     label_scores = self.label_classifier(state_features).unsqueeze(dim=0)
                
                # in the three classifier version, direction is predicted separately from the action
                # if self.separate_action_and_dir_classifiers==True:
                #     direction_scores = self.direction_classifier(state_features).unsqueeze(dim=0)
                
                # gold = if 
                # if gold_tree is not None:
                #     gold_step = parser.gold_step(gold_tree)
                #     # unpack step
                #     gold_action = torch.tensor([self.action_to_id[gold_step.action]], dtype=torch.long).to(self.device)
                #     gold_label = torch.tensor([self.label_to_id[gold_step.label]], dtype=torch.long).to(self.device)
                #     if self.separate_action_and_dir_classifiers==True:
                #         gold_direction = torch.tensor([self.direction_to_id[gold_step.direction]], dtype=torch.long).to(self.device)
                    # calculate loss
                loss_on_tags = loss_fn(prediction_scores.unsqueeze(dim=0), gold_pred)
                # print("loss on tags: ", loss_on_tags)
                    # loss_on_labels = loss_fn(label_scores, gold_label) 
                    # if self.separate_action_and_dir_classifiers==True:
                    #     loss_on_direction = loss_fn(direction_scores, gold_direction)
                    #     loss = loss_on_actions + loss_on_labels + loss_on_direction
                    # else:
                    #     loss = loss_on_actions + loss_on_labels 
                            
                    # # store loss for later
                losses.append(loss_on_tags)
                # teacher forcing
            #     next_action = gold_action
            #     next_label = gold_label
            #     if self.separate_action_and_dir_classifiers==True:
            #         next_direction = gold_direction
            else: #in this case, no else?
                predictions.append(predicted_tag)
            #     next_action = self.best_legal_action(legal_actions, action_scores)
            #     # predict the label for any of the reduce actions
            #     if self.id_to_action[next_action].startswith("reduce"):
            #         next_label = label_scores.argmax().unsqueeze(0) #unsqueeze because after softmax the output tensor is tensor(int) instead of tensor([int]) (different from next_label in training)
            #     # there is no label to predict for shift
            #     else:
            #         next_label = torch.tensor(0, dtype=torch.long).to(self.device)          
            #     if self.separate_action_and_dir_classifiers==True:
            #         next_direction = direction_scores.argmax().unsqueeze(0)
                
            
            # if self.include_relation_embedding:
            #     rel_emb = self.relation_embeddings(next_label)
            #     if self.include_direction_embedding:
            #         dir_emb = self.direction_embedding(next_direction)
            #         rel_dir_emb = torch.cat((rel_emb, dir_emb), dim=1)
            #     else:
            #         rel_dir_emb = rel_emb
            # else:
            #     rel_dir_emb = None  

            
            # action=self.id_to_action[next_action]
            # # take the step
            # parser.take_action(
            #     action=action,
            #     label=self.id_to_label[next_label] if action.startswith("reduce") else "None", # no label for shift 
            #     direction=self.id_to_direction[next_direction] if self.separate_action_and_dir_classifiers==True else None,
            #     reduce_fn=self.merge_embeddings,
            #     rel_embedding = rel_dir_emb
            # )

        # returns the TreeNode for the tree root
        # print("NEW EDUS: ", new_edus)
        
        #todo: make an annotation:
        # annotation = Annotation(docid=doc_id, raw=raw, dis=None, edus=new_edus)
        # print("annotation: ", annotation)
        # new_edus = []
        # for 
        # outputs = (predicted_tree,)

        # print("preds: ", predictions)
        # print("golds: ", gold_tags)
        # are we training?
        if train == True:
            loss = sum(losses) / len(losses)
            return loss, new_edus
        else:
            return predictions, gold_tags

        # print("outputs: ", outputs)
        # return outputs