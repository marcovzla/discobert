class CumulativeMovingAverage:
    # https://en.wikipedia.org/wiki/Moving_average#Cumulative_moving_average

    def __init__(self):
        self.n = 0
        self.avg = 0.0

    def __format__(self, format_spec):
        return format(self.avg, format_spec)

    def add(self, x):
        self.avg = float(x) + self.n * self.avg
        self.n += 1
        self.avg /= self.n

def prf1(pred, gold):
    tp, fp, fn = 0, 0, 0
    for g in gold:
        if g in pred:
            tp += 1
        else:
            fn += 1
    for p in pred:
        if p not in gold:
            fp += 1
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * ((p * r) / (p + r))
    return p, r, f1