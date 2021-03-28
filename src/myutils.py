import os
import datetime
import shutil

import torch

_CHECKPOINT_PREFIX = "checkpoint_"
_CHECKPOINT_SUFFIX = "_.pth"


def timestamped_path(suffix_dirname: str, add_hms=True) -> str:
    """
    Create Timestamped Pathname.
    The path of `suffix_dirname`/<y><date>/<hour><minutes><second>/ will be automatically generated
    (e.g.) Directory "./log/20210101/082458" if suffix_dirname == ./log
    If `add_hms` is False, <hour><minutes><second>/ will be abbreviated.

    :param suffix_dirname:
    :param add_hms:
    :return: <suffix_dirname>/
    """
    dt_now = datetime.datetime.now()
    dirname = str(dt_now.year) + str(dt_now.month).zfill(2) + str(dt_now.day).zfill(2)
    to_save_dir = os.path.join(suffix_dirname, dirname)
    if add_hms:
        hms = str(dt_now.hour).zfill(2) + str(dt_now.minute).zfill(2) + str(dt_now.second).zfill(2)
        to_save_dir = os.path.join(to_save_dir, hms)
    if not os.path.exists(to_save_dir):
        print("=> Created Directory :" + to_save_dir)
        os.makedirs(to_save_dir)
        pass
    return to_save_dir


def optimizer_to_device(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to('cuda')
                pass
            pass
        pass
    print('=> Optimizer Set to cuda')
    return optimizer


def load_checkpoint(model, optimizer, filename, scheduler=None, show_log=False):
    """

    :param model:
    :param optimizer:
    :param filename:
    :param scheduler:
    :param show_log:
    :return:
    """
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = None
        if torch.cuda.is_available():
            checkpoint = torch.load(filename)  # TODO cpu handling
        else:
            print("GPU NOT FOUND")
            checkpoint = torch.load(filename, map_location='cpu')
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']  # Because Training stop epoch are storaged on `epoch`
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if show_log:
            print("state_dict: ", model)
            print("optimizer: ", optimizer)
            print("epoch: ", start_epoch)
        if scheduler is None:
            return model, optimizer, start_epoch
        else:
            scheduler.load_state_dict(checkpoint['scheduler'])
            return model, optimizer, start_epoch, scheduler
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        print("LOADING FAILED")

    # return model, optimizer, start_epoch, losslogger
    return model, optimizer, start_epoch


def save_checkpoint(state: dict, is_best: bool, save_path: str, filename: str, timestamp=''):
    """
    Saving torch's checkpoint
    URL: https://discuss.pytorch.org/t/how-to-save-and-load-lr-scheduler-stats-in-pytorch/20208

    :param state:
    :param is_best:
    :param save_path:
    :param filename:
    :param timestamp:
    :return:
    """
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best_{0}.pth.tar'.format(timestamp))
        shutil.copyfile(filename, bestname)
        pass
    pass


def save_checkpoint_autoname():
    pass


def get_timestamped_weight_path(yd: str, epoch: int, hms="", zero_fill=4):
    if hms != "":
        ret_ = os.path.join(yd, hms)
    else:
        ret_ = yd
        pass
    w_name = _CHECKPOINT_PREFIX + str(epoch).zfill(zero_fill) + _CHECKPOINT_SUFFIX
    ret_ = os.path.join(ret_, w_name)
    return ret_


if __name__ == '__main__':
    u = get_timestamped_weight_path(yd="20210321", hms="155417", epoch=3)
    print(u)
    pass
