
import eagerpy as ep
import torch

from mmengine.runner.AdversarialAttack.foolbox.types import Bounds
from mmengine.model import is_model_wrapper

class Normalize():
    def __init__(self):
        self.data_max = None
        self.data_min = None

    def normalize(self, tensor):
        # normalize to [0, 1]
        self.data_max = tensor.max()
        self.data_min = tensor.min()
        return (tensor - self.data_min) / (self.data_max - self.data_min)

    def unnormalize(self, tensor):
        return tensor * (self.data_max - self.data_min) + self.data_min

class ModelWrapper():
    def __init__(self, my_model, bounds, normalizer):
        self.train_mode = my_model.training
        if is_model_wrapper(my_model):
            my_model = my_model.module
        self.my_model = my_model
        self.bounds = Bounds(*bounds)
        self.normalizer = normalizer
        self.data_samples = None
        self.inputs_stacked = None

    def reset_train_eval(self):
        if self.train_mode:
            self.my_model.train()
        else:
            self.my_model.eval()

    def train(self):
        self.train_mode = self.my_model.training
        self.my_model.train()

    def eval(self):
        self.train_mode = self.my_model.training
        self.my_model.eval()

    def freeze_or_unfreeze(self, mode):
        assert mode in ['freeze', 'unfreeze']
        if mode == 'freeze':
            self.param_rgs = []
            for param in self.my_model.parameters():
                self.param_rgs.append(param.requires_grad)
                param.requires_grad = False
        else:
            for rg, param in zip(self.param_rgs, self.my_model.parameters()):
                param.requires_grad = rg
            self.param_rgs = []

    def before_run(self, data_samples):
        self.data_samples = data_samples
        self.eval()
        self.freeze_or_unfreeze('freeze')

    def after_run(self):
        self.freeze_or_unfreeze('unfreeze')
        self.reset_train_eval()

    @torch.enable_grad()
    def loss(self, inputs):
        # get loss
        inputs = inputs.raw # cast type from eagerpy.Tensor to torch.Tensor
        inputs.requires_grad = True
        inputs = self.normalizer.unnormalize(inputs)
        if self.inputs_stacked:
            inputs = [input for input in inputs] # unstack
        data = dict(inputs=inputs, data_samples=self.data_samples)
        data = self.my_model.data_preprocessor(data, False)
        losses = self.my_model._run_forward(data, mode='loss')
        loss, _ = self.my_model.parse_losses(losses)
        loss, restore_type = ep.astensor_(loss)
        return loss
