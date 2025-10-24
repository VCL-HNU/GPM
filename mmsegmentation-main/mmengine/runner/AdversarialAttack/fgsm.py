
import torch

from .utils import Normalize, ModelWrapper
from . import foolbox

class FGSM(foolbox.attacks.FGSM):
    def __init__(self, my_model):
        super(FGSM, self).__init__()

        self.normalizer = Normalize()

        assert isinstance(my_model, torch.nn.Module)
        self.wrapped_model = ModelWrapper(my_model, (0, 1), self.normalizer)

    def get_loss_fn(self, *args, **kwargs):
        return self.wrapped_model.loss

    def __call__(self, data, epsilon):
        assert isinstance(data, dict)
        assert isinstance(epsilon, (int, float))
        inputs, data_samples = data.values()
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
            self.wrapped_model.inputs_stacked = True
        inputs = self.normalizer.normalize(inputs)

        labels = torch.zeros(len(inputs)).to(
            device=inputs.device,
            dtype=torch.long,
        ) # not used

        self.wrapped_model.before_run(data_samples)
        attacked_inputs = self.run(self.wrapped_model, inputs, labels, epsilon=epsilon)
        attacked_inputs = self.normalizer.unnormalize(attacked_inputs)
        if self.wrapped_model.inputs_stacked:
            attacked_inputs = [attacked_input for attacked_input in attacked_inputs] # unstack
        data = dict(inputs=attacked_inputs, data_samples=data_samples)

        self.wrapped_model.after_run()
        return data
