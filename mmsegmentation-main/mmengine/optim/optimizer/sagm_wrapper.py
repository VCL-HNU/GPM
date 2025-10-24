
from typing import Dict, Optional

import torch

from mmengine.registry import OPTIM_WRAPPERS
from .amp_optimizer_wrapper import OptimWrapper
from mmengine import mm_transfer

from .sagm import SAGM
from .sagm.scheduler import CosineScheduler, LinearScheduler, PolyScheduler, ProportionScheduler


@OPTIM_WRAPPERS.register_module()
class SAGMOptimWrapper(OptimWrapper):
    def __init__(self, optimizer, SAGM_cfg, **optimizer_cfg):
        self.model = mm_transfer.runner.model

        # 构建rho调度器
        rho_scheduler = LinearScheduler(T_max=1e10, max_value=0.05, min_value=0.05)

        # 初始化SAGM优化器
        SAGM_optimizer = SAGM(
            params=self.model.parameters(),
            base_optimizer=optimizer,
            model=self.model,
            alpha=SAGM_cfg.get('alpha', 0.001),
            rho_scheduler=rho_scheduler,
            adaptive=SAGM_cfg.get('adaptive', False),
            perturb_eps=SAGM_cfg.get('perturb_eps', 1e-12),
            grad_reduce=SAGM_cfg.get('grad_reduce', 'mean')
        )
        SAGM_optimizer.defaults.update(optimizer.defaults)

        super().__init__(
            optimizer=SAGM_optimizer,
            **optimizer_cfg
        )

        @torch.no_grad()
        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                losses = self.model._run_forward(mm_transfer.data, mode='loss')
                loss, _ = self.model.parse_losses(losses)
            self.backward(loss=loss)
            return None, loss.data.clone().detach()
        self.closure = get_grad

    def update_params(
            self,
            loss: torch.Tensor,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None) -> None:
        super(SAGMOptimWrapper, self).update_params(
            loss=loss,
            step_kwargs=dict(closure=self.closure),
            zero_kwargs=zero_kwargs
        )
