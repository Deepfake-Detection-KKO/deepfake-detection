# Code borrowed from: https://www.geeksforgeeks.org/deep-learning/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.epoch = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss # Invert loss so higher score is better
        self.epoch += 1

        if self.best_score is None: # Save model weights from 1st epoch
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_epoch = self.epoch
        elif score < self.best_score + self.delta: # Increment counter and set early stop if counter exceeds patience
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else: # Save model state if score is better by more than delta
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
            self.best_epoch = self.epoch

    def load_best_model(self, model):
        print(f'Model weights from epoch {self.best_epoch}')
        model.load_state_dict(self.best_model_state)
        return self.best_epoch