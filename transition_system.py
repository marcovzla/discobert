class Node:
    def __init__(self, lhs, rhs, embedding=None):
        self.lhs = lhs
        self.rhs = rhs
        self.embedding = embedding
        self.span = range(lhs.span.start, rhs.span.stop)

    def spans(self):
        ss = [self.span]
        ss.extend(self.lhs.spans())
        ss.extend(self.rhs.spans())
        return ss

class EDU:
    def __init__(self, text, idx, embedding=None):
        self.text = text
        self.span = range(idx, idx+1)
        self.embedding = embedding

    def spans(self):
        return [self.span]

class TransitionSystem:

    def __init__(self, edus=None):
        self.reset()
        self.buffer.extend(edus)

    def reset(self):
        self.buffer = []
        self.stack = []

    def is_done(self):
        return len(self.buffer) == 0 and len(self.stack) == 1

    def get_result(self):
        if self.is_done():
            return self.stack[0]

    def shift(self):
        node = self.buffer.pop(0)
        self.stack.append(node)

    def reduce(self, reduce_fn=None):
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        emb = None if reduce_fn is None else reduce_fn(lhs.embedding, rhs.embedding)
        node = Node(lhs, rhs, emb)
        self.stack.append(node)

    def all_actions(self):
        return ['shift', 'reduce']

    def legal_actions(self):
        actions = []
        if self.can_shift():
            actions.append('shift')
        if self.can_reduce():
            actions.append('reduce')
        return actions

    def can_shift(self):
        return len(self.buffer) >= 1

    def can_reduce(self):
        return len(self.stack) >= 2

    def all_correct_actions(self, gold_spans):
        correct_actions = []
        if self.can_reduce():
            rhs = self.stack[-1]
            lhs = self.stack[-2]
            new_span = range(lhs.span.start, rhs.span.stop)
            if new_span in gold_spans:
                correct_actions.append('reduce')
            elif self.can_shift():
                correct_actions.append('shift')
            else:
                raise Exception("There is no correct action given the current state of the parser.")
        elif self.can_shift():
            correct_actions.append('shift')
        return correct_actions

    def take_action(self, action, reduce_fn=None):
        if action == 'shift':
            self.shift()
        elif action == 'reduce':
            self.reduce(reduce_fn)
