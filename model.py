import torch
from torch import nn
from transformers import *
from transition_system import EDU, TransitionSystem

inf = float('inf')

class DiscoBertModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # todo: put bert selection in the config?
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # todo: check what hidden size
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # TODO tensor to represent missing node
        self.bert_embedding_size = ??? # FIXME
        self.missing_node = nn.init.normal_(torch.empty(self.bert_embedding_size))


        # vocabularies
        self.id_to_action = ['shift', 'reduce']
        self.action_to_id = {action:id for id,action in enumerate(self.id_to_action)}

        self.id_to_direction = ['leftToRight', 'rightToLeft', 'None']
        self.direction_to_id = {direction:id for id,direction in enumerate(self.id_to_direction)}

        self.id_to_relation = ['elaboration', 'contrast'] # fixme: add rest/real
        self.relation_to_id = {relation:id for id,relation in enumerate(self.id_to_relation)}

        self.init_weights()

    def make_features(self, parser):
        s1 = self.missing_node if len(parser.stack) < 2 else parser.stack[-2].embedding
        s0 = self.missing_node if len(parser.stack) < 1 else parser.stack[-1].embedding
        b = self.missing_node if len(parser.buffer) < 1 else parser.buffer[0].embedding
        features = torch.cat([s1, s0, b])
        return features


    def merge_embeddings(self, embed_1, embed_2):
        # for now, add
        return embed_1 + embed_2

    def best_action(self, actions, logits):
        if len(actions) == 1:
            return self.action2id[actions[0]]
        elif len(actions) == logits.shape[0]: # FIXME is batch first or after?
            return torch.argmax(logits)
        action_ids = [self.action2id[a] for a in actions]
        mask = torch.ones_like(logits) * -inf
        mask[action_ids] = 0
        masked_logits = logits + mask
        return torch.argmax(masked_logits)

    # forward is for whole document
    def forward(self, edus, gold_tree=None):
        # tokenize
        buffer = []
        for i,edu in enumerate(edus):
            token_ids = self.tokenizer.encode(edu, add_special_tokens=True, return_tensors='pt')
            embedding = self.bert(token_ids)[1]
            buffer.append(EDU(edu, i, embedding))

        # initialize the automata
        parser = TransitionSystem(buffer)

        if gold_tree is not None:
            gold_spans = gold_tree.gold_spans()
            loss_fct = nn.CrossEntropyLoss()
            losses = []

        while not parser.is_done():
            state_features = self.make_features(parser)
            logits = self.classifier(state_features)
            legal_actions = parser.legal_actions()
            pred_action = self.best_action(legal_actions, logits)
            if gold_tree is not None:
                gold_actions = parser.all_correct_actions(gold_spans)
                gold_action = self.best_action(gold_actions, logits)
                loss = loss_fct(logits.view(-1, self.num_labels), gold_action)
                losses.append(loss)
                # teacher forcing ?
                parser.take_action(self.id_to_action[gold_action], self.merge_embeddings) # merge_embeddings is only used for REDUCE action
            else:
                parser.take_action(self.id_to_action[pred_action], self.merge_embeddings) # merge_embeddings is only used for REDUCE action

        predicted_tree = parser.get_result()
        outputs = (predicted_tree,)

        if gold_tree is not None:
            loss = sum(losses) / len(losses)
            outputs = (loss,) + outputs
            # todo: precision, recall, f1 of predicted_tree

        return outputs
