class Node:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

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

    def reduce(self):
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        node = Node(lhs, rhs)
        self.stack.append(node)

    def all_actions(self):
        return ['shift', 'reduce']

    def valid_actions(self):
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
