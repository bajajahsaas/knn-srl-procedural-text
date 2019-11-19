import torch
class EarlyStopping:
    def __init__( self, patience, model_path, minmax = 'min'):
        self.patience = patience
        self.model_path = model_path
        self.minmax = minmax
        self.best = None
        self.history = []
        self.early_stop = False

    def save(self, model):
        torch.save(model.state_dict(), self.model_path)

    def step(self, value, model):
        if self.best == None or \
            (self.minmax == 'min' and value < self.best) or \
                (self.minmax == 'max' and value > self.best):
            self.best = value
            print('Saving model')
            self.save(model)
        if len(self.history) < self.patience:
            self.history = self.history + [value]
            return

        self.history = self.history[1:] + [value]

        if len(self.history) < self.patience:
            return

        if self.minmax == 'min':
            historybest = min(self.history)
            if historybest > self.best:
                self.early_stop = True
        else:
            historybest = max(self.history)
            if historybest < self.best:
                self.early_stop = True
                

