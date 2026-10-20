import torch
import copy

class EarlyStopping:
    def __init__(self, patience = 5, min_delta = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_mf1 = 0.0
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, mf1, model):
        if self.best_model_wts is None:
            self.best_mf1 = mf1
            self.model = copy.deepcopy(model.state_dict())

        elif mf1 < self.best_mf1 - self.min_delta:
            self.counter += 1 
            print(f"EarlyStopping counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_mf1 = mf1
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_wts)