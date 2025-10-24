# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
import shutil

import sys
base_path = os.path.abspath(__file__ + '/../../')
sys.path.insert(0, base_path)

import os.path as osp
from copy import deepcopy

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.dist import is_main_process

from mmengine import mm_transfer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        default=True,
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='whether to disable the persistent_workers option in dataloaders.'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def get_ngpus():
    gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    gpus = [n.strip(' ') for n in gpus]
    gpus = [n for n in gpus if n != '']
    return len(gpus)


def copy_logs(root_0='Logs', root_1='Logs_2'):
    # copy logs from root_0 to root_1
    if is_main_process():
        if not os.path.exists(root_1):
            os.makedirs(root_1)
        files_0 = list(os.listdir(root_0))
        files_1 = list(os.listdir(root_1))
        for file_0 in files_0:
            if file_0 == 'run_ckpt' or file_0.endswith('.pth'):
                continue
            if os.path.isdir(os.path.join(root_0, file_0)):
                copy_logs(os.path.join(root_0, file_0), os.path.join(root_1, file_0))
            elif file_0 in files_1:
                ori_file = os.path.join(root_0, file_0)
                new_file = os.path.join(root_1, file_0)
                if os.path.getsize(ori_file) == os.path.getsize(new_file):
                    continue
                else:
                    shutil.copy(ori_file, new_file)
            else:
                shutil.copy(os.path.join(root_0, file_0), os.path.join(root_1, file_0))


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set dataloader args
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type='default_collate'),
    )
    if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
        default_dataloader_cfg.persistent_workers = False

    def set_default_dataloader_cfg(cfg, field):
        results = cfg.get(field, None)
        if results is None:
            return
        results = results if isinstance(results, list) else [results]
        for result in results:
            dataloader_cfg = deepcopy(default_dataloader_cfg)
            dataloader_cfg.update(result)
            result.update(dataloader_cfg)
            if args.no_pin_memory:
                result['pin_memory'] = False
            if args.no_persistent_workers:
                result['persistent_workers'] = False

    set_default_dataloader_cfg(cfg, 'train_dataloader')
    set_default_dataloader_cfg(cfg, 'val_dataloader')
    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def before_runner(args, cfg):
    def _initialize():
        mm_transfer.args = args
        mm_transfer.cfg = cfg
        cfg.setdefault('for_debug', False)
        assert "CUDA_VISIBLE_DEVICES" in os.environ, 'GPU usage should be set in os.environ.'
        assert 'batch_size' in cfg, 'batch_size should be set in cfg.'
        cfg.ngpus = get_ngpus()
        # check if second derivative loss is used.
        def _check_sd_loss(cfg):
            sdloss_cfg = cfg.get('sdloss_cfg', dict())
            losses = sdloss_cfg.get('losses', [])
            return len(losses) > 0
        cfg.has_sd_loss = _check_sd_loss(cfg)
        # torch implementation of SyncBatchNorm does not support higher derivative
        if cfg.has_sd_loss and cfg.ngpus>1:
            cfg.sync_bn = 'mmcv'
        # if batch_size is different with base_batch_size, change max_iter
        # for IterBasedTrainLoop.
        do_change_iter = cfg.get('do_change_iter', True)
        def _change_iter(cfg, settings=None, ratio=1):
            if isinstance(cfg, tuple):
                return tuple(int(c * ratio) for c in cfg)
            for setting in settings:
                if setting in cfg:
                    cfg[setting] = int(cfg[setting] * ratio)
        iter_based = False
        train_cfg = cfg.get('train_cfg', None)
        if train_cfg:
            if 'type' in train_cfg:
                if train_cfg['type'] == 'IterBasedTrainLoop':
                    iter_based = True
            elif not train_cfg['by_epoch']:
                iter_based = True
        base_batch_size = cfg.batch_size
        auto_scale_lr = cfg.get('auto_scale_lr', dict())
        base_batch_size = auto_scale_lr.get('base_batch_size')
        if iter_based and cfg.batch_size != base_batch_size and do_change_iter:
            ratio = base_batch_size / cfg.batch_size
            max_iters = train_cfg['max_iters']
            new_max_iters = int(max_iters * ratio)
            print(f'A smaller batch_size is detected. For IterBasedTrainLoop, ' \
                  f'scale max_iters from {max_iters} to {new_max_iters}, based on '
                  f'base_batch_size is {base_batch_size} and batch_size is '
                  f'{cfg.batch_size}.')
            settings = ['warmup', 'max_iters', 'max_epochs']
            _change_iter(cfg, settings, ratio)
            settings = ['max_iters', 'val_begin', 'val_interval']
            _change_iter(train_cfg, settings, ratio)
            if hasattr(train_cfg, 'dynamic_intervals'):
                train_cfg['dynamic_intervals'] = _change_iter(train_cfg['dynamic_intervals'], ratio=ratio)
            if hasattr(cfg, 'param_scheduler'):
                param_scheduler = cfg.param_scheduler
                if not isinstance(param_scheduler, list):
                    param_scheduler = [param_scheduler]
                settings = ['begin', 'end']
                for ps in param_scheduler:
                    _change_iter(ps, settings, ratio)
                cfg.param_scheduler = param_scheduler
        # automatically set batch_size based on the number of used gpus.
        # the original batch_size is regarded as all batch_size on all gpus.
        batch_size = cfg.batch_size
        assert batch_size % cfg.ngpus == 0, f'batch_size = {batch_size}, cfg.ngpus = {cfg.ngpus}'
        if cfg.ngpus > 1:
            cfg.train_dataloader.batch_size = max(1, batch_size // cfg.ngpus)
            # cfg.val_dataloader.batch_size = max(1, batch_size // cfg.ngpus // 2)
            # cfg.test_dataloader.batch_size = max(1, batch_size // cfg.ngpus // 5)
            print(f'set batch_size on each GPU to {batch_size // cfg.ngpus} based on nGPUs = {cfg.ngpus}.'
                  f'real batch_size is {batch_size}.')
        mm_transfer.setdefault('accumulative_counts', 1)
        if hasattr(cfg, 'optim_wrapper'):
            mm_transfer.accumulative_counts = cfg.optim_wrapper.get('accumulative_counts', 1)
        # set test_mode for datasets
        def _set_test_mode(dataloader, test_mode):
            def _get_dataset(dataset):
                dataset_wrappers = ['RepeatDataset', 'ClassBalancedDataset']
                if hasattr(dataset, 'type') and dataset['type'] in dataset_wrappers:
                    datasets = _get_dataset(dataset['dataset'])
                elif hasattr(dataset, 'type') and dataset['type'] == 'ConcatDataset':
                    datasets = []
                    for d in dataset['datasets']:
                        datasets += _get_dataset(d)
                else:
                    datasets = [dataset]
                if isinstance(datasets, list):
                    return datasets
                else:
                    return [datasets]
            if dataloader is None:
                return None
            dataloaders = dataloader if isinstance(dataloader, list) else [dataloader]
            for dataloader in dataloaders:
                datasets = _get_dataset(dataloader['dataset'])
                for dataset in datasets:
                    dataset['test_mode'] = test_mode
        train_dataloader = cfg.get('train_dataloader', None)
        _set_test_mode(train_dataloader, False)
        val_dataloader = cfg.get('val_dataloader', None)
        _set_test_mode(val_dataloader, True)
        test_dataloader = cfg.get('test_dataloader', None)
        _set_test_mode(test_dataloader, True)
        # set log_processor
        cfg.setdefault('log_processor', dict())
        cfg.log_processor.setdefault('custom_cfg', [])
        cfg.log_processor.custom_cfg.append(dict(data_src='sd_grad_nan_ratio', method_name='mean', window_size=10))
        # check for mixed-precision
        cfg.mixed_precision = False
        if hasattr(cfg, 'optim_wrapper'):
            optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
            if optim_wrapper in ['AmpOptimWrapper', 'ApexOptimWrapper']:
                cfg.mixed_precision = True
        # gradient accumulation is to mimic large bs. If using larger bs, lr should be
        # larger. Reduce base_batch_size would enlarge lr.
        if hasattr(cfg, 'auto_scale_lr'):
            cfg.auto_scale_lr['base_batch_size'] /= mm_transfer.accumulative_counts
        # for debug
        if cfg.for_debug:
            cfg.default_hooks['logger']['interval'] = 1

    def _assert():
        if cfg.has_sd_loss:
            # If second derivative loss is used, only support 'OptimWrapper' or 'AmpOptimWrapper'
            if hasattr(cfg, 'optim_wrapper') and hasattr(cfg.optim_wrapper, 'type'):
                error_msg = f"For second derivative losses, optim_wrapper.type need to be in " \
                            f"['OptimWrapper', 'AmpOptimWrapper'], got type {cfg.optim_wrapper.type}."
                assert cfg.optim_wrapper.type in ['OptimWrapper', 'AmpOptimWrapper'], error_msg
        # scale iter-based settings based on accumulative_counts
        if hasattr(cfg, 'sdloss_cfg'):
            if hasattr(cfg.sdloss_cfg, 'after_iter'):
                if isinstance(cfg.sdloss_cfg.after_iter, int):
                    cfg.sdloss_cfg.after_iter *= mm_transfer.accumulative_counts
                else:
                    assert isinstance(cfg.sdloss_cfg.after_iter, float) and 0 <= cfg.sdloss_cfg.after_iter <= 1
            if hasattr(cfg.sdloss_cfg, 'warmup_iter'):
                if isinstance(cfg.sdloss_cfg.warmup_iter, int):
                    cfg.sdloss_cfg.warmup_iter *= mm_transfer.accumulative_counts
                else:
                    assert isinstance(cfg.sdloss_cfg.warmup_iter, float) and 0 <= cfg.sdloss_cfg.warmup_iter <= 1

    _initialize()
    _assert()


def after_runner(args, cfg, runner):
    def _initialize():
        mm_transfer.runner = runner
        if hasattr(cfg, 'sdloss_cfg'):
            if hasattr(cfg.sdloss_cfg, 'after_iter') and isinstance(cfg.sdloss_cfg.after_iter, float):
                cfg.sdloss_cfg.after_iter = int(cfg.sdloss_cfg.after_iter * runner.train_loop.max_iters)
            if hasattr(cfg.sdloss_cfg, 'warmup_iter') and isinstance(cfg.sdloss_cfg.warmup_iter, float):
                cfg.sdloss_cfg.warmup_iter = int(cfg.sdloss_cfg.warmup_iter * runner.train_loop.max_iters)

    def _assert():
        pass

    _initialize()
    _assert()


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    before_runner(args, cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    after_runner(args, cfg, runner)

    # start training
    runner.train()
    runner.test()

    copy_logs()


if __name__ == '__main__':
    main()
