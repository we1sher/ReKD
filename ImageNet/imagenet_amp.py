import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from models.rekd import build_re_kd
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
scaler = GradScaler(enabled=True)

import numpy as np

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='Initial learning rate.')
parser.add_argument('--lr_adjust_step', default=[30,60,90], type=int, nargs='+',
                    help='initial learning rate')
parser.add_argument('--lr_adjust_type', default='step', type=str,
                    help='lr adjust type')
parser.add_argument('--gamma', default='0.99', type=float,
                    help='exp lr gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save_dir', type=str, default='')

parser.add_argument('--prof', default=-1, type=int,
                    help='Only run 10 iterations for profiling.')
parser.add_argument("--seed", default=None, type=int)
parser.add_argument('--opt-level', type=str, default='O0')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)

parser.add_argument('--teacher', type=str, default='',
                    help='teacher model')
parser.add_argument('--teacher-weight', type=str, default='torchvision',
                    help='teacher model weight path')
parser.add_argument('--re-pr-loss-weight', type=float, default=1.0,
                    help='feature konwledge distillation loss weight')
parser.add_argument('--re-kd-loss-weight', type=float, default=1.0,
                    help='feature konwledge distillation loss weight')
parser.add_argument('--re-kd-plan', type=str, default='PlanA',
                    help='plan of rekd')

cudnn.benchmark = True

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array.copy())
        
    return tensor, targets


best_prec1 = 0
args = parser.parse_args()
lr_base = []

if args.seed is not None:
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

print(args)

print("opt_level = {}".format(args.opt_level))
print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))

print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))
def main():
    global best_prec1, args

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.teacher != '':
            model = build_re_kd(args.arch)
        else:
            model = models.__dict__[args.arch]()
                                                                        
    model = model.cuda()
    print(model)
    args.re_kd_loss_weight *= (model.t_c / 2**10)

    # Scale learning rate based on global batch size
    print('learning rate: ', args.lr)
    param = [
                {"params": model.student.parameters()},
                {"params": model.adapL.parameters()},
                {"params": model.vae.parameters(), "weight_decay": 0.0},
        ]
    optimizer = torch.optim.SGD(param, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lr_adjust_type == 'step':
        scheduler = MultiStepLR(optimizer, milestones=args.lr_adjust_step, gamma=0.1)
    elif args.lr_adjust_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
    elif args.lr_adjust_type == 'exp':
        scheduler = ExponentialLR(optimizer, args.gamma)

    criterion = nn.CrossEntropyLoss().cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda())
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    if(args.arch == "inception_v3"):
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320
    else:
        crop_size = 224
        val_size = 256
    
    trans = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip()
        ])
    train_dataset = datasets.ImageFolder(
        traindir, trans
    )
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop_size),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=fast_collate)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=fast_collate)

    if args.teacher != '':
        if args.teacher_weight == 'torchvision':
            print('Using torchvision pretrained model for teacher {}'.format(args.teacher))
            teacher = models.__dict__[args.teacher](pretrained=True)
        else:
            teacher = models.__dict__[args.teacher]()
            for param in teacher.parameters():
                param.requires_grad = False
            checkpoint = torch.load(args.teacher_weight, map_location = lambda storage, loc: storage.cuda(args.gpu))
            teacher.load_state_dict(checkpoint)
        teacher = teacher.cuda()
    else:
        teacher = None
 
    if args.evaluate:
        validate(val_loader, model, teacher, criterion)
        return
    st_time = time.time()
    prec1 = 0.
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, teacher, criterion, optimizer, epoch, scheduler)

        # evaluate on validation set
        prec1 = validate(val_loader, model, teacher, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print(epoch)
        print('Learning rate:', optimizer.param_groups[0]['lr'])
        print('Total Time: '+ format_time(time.time()-st_time))
        print('Remaining Time: '+ format_time((time.time()-st_time)/(epoch-args.start_epoch+1)*(args.epochs-epoch-1)))
        print('Best Acc: '+ str(best_prec1))
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def train(train_loader, model, teacher, criterion, optimizer, epoch, scheduler):
    batch_time = AverageMeter()
    losses = AverageMeter()
    rekd_losses = AverageMeter()
    cls_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    teacher.eval()
    end = time.time()

    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1

        adjust_learning_rate(scheduler, optimizer, epoch, i, len(train_loader))
        
        with autocast(enabled=True, device_type='cuda'):
            if teacher is not None:
                with torch.no_grad():
                    fteacher, _ = teacher(input, is_feat=True, preact=False)
                [s_features, rt_features, statistic], output = model(input, fteacher[-2].detach())

                if args.re_kd_plan == 'PlanA':
                    fp = (s_features.detach() - statistic[0])/statistic[1]
                    fk = (s_features - statistic[0].detach())/statistic[1].detach()
                    mu_p = fp.mean([2,3])
                    var_p = fp.var([2,3], unbiased=True)
                    mu_k = fk.mean([2,3])
                    var_k = fk.var([2,3], unbiased=True)

                    feature_kd_loss = torch.mean(1 + var_k.log() - mu_k.pow(2) - var_k).mul(-0.5)
                    feature_pr_loss = torch.mean(1 + var_p.log() - mu_p.pow(2) - var_p).mul(-0.5)
                    
                elif args.re_kd_plan == 'PlanB':
                    fp = (s_features.detach() - statistic[0])/statistic[1]
                    fk = (s_features - statistic[0].detach())/statistic[1].detach()
                    ff = fp * args.pr_loss_weight + fk * (1-args.pr_loss_weight)
                    mu_ = ff.mean([2,3])
                    var_ = ff.var([2,3], unbiased=True)

                    feature_kd_loss = torch.mean(1 + var_.log() - mu_.pow(2) - var_).mul(-0.5)

                else:
                    assert False

                loss_rekd = feature_kd_loss * args.re_kd_loss_weight +\
                            feature_pr_loss * args.re_pr_loss_weight +\
                            F.mse_loss(rt_features, fteacher[-2].detach())
            else:
                output = model(input)
            if args.prof >= 0: torch.cuda.nvtx.range_pop()
            loss_cls = criterion(output, target)
            loss = loss_cls + loss_rekd if teacher is not None else loss_cls
            
        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
   
            # Average loss and accuracy across processes for logging 
            reduced_loss = loss.data
            reduced_rekd_loss = loss_rekd.data
            reduced_cls_loss = loss_cls.data
   
            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            rekd_losses.update(to_python_float(reduced_rekd_loss), input.size(0))
            cls_losses.update(to_python_float(reduced_cls_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))
    
            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()
            
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Speed {3:.3f} ({4:.3f})\t'
                    'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                    'ReKDLoss {rekdloss.val:.10f} ({rekdloss.avg:.4f})\t'
                    'ClsLoss {clsloss.val:.10f} ({clsloss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader),
                    args.batch_size/batch_time.val,
                    args.batch_size/batch_time.avg,
                    batch_time=batch_time,
                    loss=losses, rekdloss = rekd_losses, clsloss = cls_losses,
                    top1=top1, top5=top5))

        input, target = prefetcher.next()


def validate(val_loader, model, teacher, criterion):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            output = model(input, None)
        if teacher is not None:
            _, output = output

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader),
                   args.batch_size / batch_time.val,
                   args.batch_size / batch_time.avg,
                   batch_time=batch_time,
                   top1=top1, top5=top5))

        input, target = prefetcher.next()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    filename = os.path.join(args.save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save_dir, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(scheduler, optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""

    """Warmup"""
    lr = args.lr
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return
    
    """Adjust learning rate"""
    for i in args.lr_adjust_step:
        if i <= epoch:
            lr *= 0.1
    for param_group in optimizer.param_groups:
         param_group['lr'] = lr



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

import torch
import numpy as np

def to_python_float(x):
    if isinstance(x, float):
        return x
    elif isinstance(x, int):
        return float(x)
    elif isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(f"Tensor with more than one element cannot convert to float: shape={x.shape}")
        return x.item()
    elif isinstance(x, np.generic):
        return float(x)
    else:
        raise TypeError(f"Cannot convert type {type(x)} to float")


if __name__ == '__main__':
    main()
