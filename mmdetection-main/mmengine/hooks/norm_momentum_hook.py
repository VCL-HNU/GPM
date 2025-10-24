
import warnings

import torch
from torch.nn import GroupNorm, LayerNorm

from mmengine.registry import HOOKS
from mmengine.model import is_model_wrapper
from mmengine.utils.dl_utils.misc import _BatchNorm, _InstanceNorm

from .hook import Hook


@HOOKS.register_module()
class NormMomentumHook(Hook):
    """This hook change momentums of norm layers. It is only
    conducted one time before train.

    Args:
        do (bool): Do this or not.
        momentum (float): The momentum to use.
        batch_size_ratio (float):
    """

    priority = 'NORMAL'

    def __init__(self,
                 do=True,
                 momentum: float = None,
                 batch_size_ratio: float = None):
        assert momentum is not None or batch_size_ratio is not None
        assert momentum is None or batch_size_ratio is None
        assert batch_size_ratio > 0, batch_size_ratio
        self.do = do
        self.momentum = momentum
        self.batch_size_ratio = batch_size_ratio

    def change_norm_momentum(self, model):
        if is_model_wrapper(model):
            model = model.module
        if self.do and not hasattr(model, '_momentum_changed'):
            from mmcv.ops import SyncBatchNorm as _SyncBatchNorm
            SyncBatchNorm = (_SyncBatchNorm, torch.nn.modules.batchnorm.SyncBatchNorm)
            norm_layers = dict()
            for module in model.modules():
                if isinstance(module, _BatchNorm):
                    norm_layers.setdefault('BatchNorm', []).append(module)
                elif isinstance(module, SyncBatchNorm):
                    norm_layers.setdefault('BatchNorm', []).append(module)
                elif isinstance(module, _InstanceNorm):
                    norm_layers.setdefault('InstanceNorm', []).append(module)
                elif isinstance(module, GroupNorm):
                    norm_layers.setdefault('GroupNorm', []).append(module)
                elif isinstance(module, LayerNorm):
                    norm_layers.setdefault('LayerNorm', []).append(module)
            if 'GroupNorm' in norm_layers:
                warnings.warn('GroupNorm does not have momentum value, changing momentums '
                              'are skipped.')
            if 'LayerNorm' in norm_layers:
                warnings.warn('LayerNorm does not have momentum value, changing momentums '
                              'are skipped.')
            layers = []
            if 'BatchNorm' in norm_layers:
                print(
                    f'Changing momentum of {len(norm_layers["BatchNorm"])} BatchNorms.'
                    )
                layers += norm_layers['BatchNorm']
            if 'InstanceNorm' in norm_layers:
                print(
                    f'Changing momentum of {len(norm_layers["InstanceNorm"])} InstanceNorms.'
                    )
                layers += norm_layers['InstanceNorm']
            for norm_layer in layers:
                if self.momentum:
                    norm_layer.momentum = self.momentum
                if self.batch_size_ratio:
                    norm_layer.momentum = 1 - (1 - norm_layer.momentum) ** (1 / self.batch_size_ratio)
            model._momentum_changed = torch.nn.Parameter(torch.Tensor([1.]))

    def before_train(self, runner) -> None:
        """Args:
            runner (Runner): The runner of the training process.
        """
        self.change_norm_momentum(runner.model)
