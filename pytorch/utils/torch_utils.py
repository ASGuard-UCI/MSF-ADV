import torch
import os
import shutil

def save_checkpoint(model, optimizer, curr_epoch, curr_step, args, curr_loss, curr_acc, filename):
    """
        Saves a checkpoint and updates the best loss and best weighted accuracy
    """
    is_best_loss = curr_loss < args.best_loss
    is_best_acc = curr_acc > args.best_acc

    args.best_acc = max(args.best_acc, curr_acc)
    args.best_loss = min(args.best_loss, curr_loss)

    state = {   'epoch':curr_epoch,
                'step': curr_step,
                'args': args,
                'state_dict': model.state_dict(),
                'val_loss': args.best_loss,
                'val_acc': args.best_acc,
                'optimizer' : optimizer.state_dict(),
             }
    path = os.path.join(args.experiment_path, filename)
    torch.save(state, path)
    if is_best_loss:
        shutil.copyfile(path, os.path.join(args.experiment_path, 'model_best_loss.pkl'))
    if is_best_acc:
        shutil.copyfile(path, os.path.join(args.experiment_path, 'model_best_acc.pkl'))

    return args

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.autograd.Variable(x, volatile=volatile)

def accuracy(output, target, topk=(1,)):
    """ From The PyTorch ImageNet example """
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
