import os
import torch
import glob

class YelpCheckpointer:
    def __init__(self, model, save_dir, optimizer=None, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.iteration = -1

    def resume_or_load(self, weight, resume):
        if resume:
            prev_weight = list.sort(glob.glob(os.path.join(self.save_dir, '*.pt')))
            if len(prev_weight):
                weight = prev_weight[-1]

        if weight is not None:
            checkpoint = torch.load(weight)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if resume:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_dict'])
                self.iteration = checkpoint['iteration']
            else:
                self.model.eval()

    def save_model(self, curr_iter):
        self.iteration = curr_iter
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_dict': self.scheduler.state_dict(),
            'iteration': self.iteration
        }, os.path.join(self.save_dir, 'model_'+ "{:4d}".format(curr_iter) + '.pt'))
    
    def has_checkpoint(self):
        return self.iteration != -1

class YelpPeriodicCheckpointer(YelpCheckpointer):
    def __init__(self, checkpointer, period, max_iter=0):
        super().__init__(checkpointer.model, checkpointer.save_dir, checkpointer.optimizer, checkpointer.scheduler)
        self.period = period
        self.max_iter = max_iter

    def step(self, curr_iter):
        if curr_iter <= self.max_iter and curr_iter % self.period == 0:
            self.save_model(curr_iter)