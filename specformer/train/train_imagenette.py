import tqdm
from functools import partial
import random
from timm.loss import  SoftTargetCrossEntropy
from timm.data import Mixup
from parser.parser_imagenette import get_args
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from auto_LiRPA.utils import MultiAverageMeter
from train.utils import *
from checkpoint_saver import CheckpointSaver
from torch.autograd import Variable
from robust_evaluate.pgd import evaluate_pgd,evaluate_CW
from robust_evaluate.fgsm import validate_fgsm
from robust_evaluate.aa import evaluate_aa
from auto_LiRPA.utils import logger
import torch.backends.cudnn as cudnn
import time
import os
import csv
args = get_args()
args.out_origin=args.out_dir
args.out_dir = args.out_dir+"_"+args.dataset+"_"+args.model+"_"+args.method
args.out_dir = args.out_dir +"/seed"+str(args.seed)

param=''
if args.pen_for_qkv:
    for key in args.pen_for_qkv:
        param = param +'_' +str(key)


args.out_dir = args.out_dir +"/pen_for_qkv"+param+"_tradeoff"+str(args.trade_off)


print(args.out_dir)
os.makedirs(args.out_dir,exist_ok=True)
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
logfile = os.path.join(args.out_dir, 'log_{:.4f}-{}.log'.format(args.weight_decay,now))

file_handler = logging.FileHandler(logfile)
file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
logger.addHandler(file_handler)
logger.info(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
resize_size = args.resize
crop_size = args.crop
train_loader, test_loader = get_loaders(args)


if args.model == "vit_small_patch16_224_sn":
    from model_for_imagenet_sn.vit import vit_small_patch16_224_sn
    model = vit_small_patch16_224_sn(pretrained= (not args.scratch),num_classes =10,pen_for_qkv=args.pen_for_qkv).cuda()
    model.init_sn(seed=args.seed); model = nn.DataParallel(model)
    logger.info('Model{}'.format(model))
elif args.model == "vit_base_patch16_224_sn":
    from model_for_imagenet_sn.vit import vit_base_patch16_224_sn
    model = vit_base_patch16_224_sn(pretrained= (not args.scratch),num_classes =10,pen_for_qkv=args.pen_for_qkv).cuda()
    model.init_sn(seed=args.seed); model = nn.DataParallel(model)
    logger.info('Model{}'.format(model))
elif args.model == "vit_base_patch16_224_sn_in21k":
    from model_for_imagenet_sn.vit import vit_base_patch16_224_sn_in21k
    model = vit_base_patch16_224_sn_in21k(pretrained=(not args.scratch), num_classes =10,pen_for_qkv=args.pen_for_qkv).cuda()
    model.init_sn(seed=args.seed); model = nn.DataParallel(model)
    logger.info('Model{}'.format(model))
elif args.model == "deit_small_patch16_224_sn":
    from model_for_imagenet_sn.deit import deit_small_patch16_224_sn
    model = deit_small_patch16_224_sn(pretrained = (not args.scratch),num_classes =10,pen_for_qkv=args.pen_for_qkv ).cuda()
    model.init_sn(seed=args.seed); model = nn.DataParallel(model)
    logger.info('Model{}'.format(model))
elif args.model == "deit_tiny_patch16_224_sn":
    from model_for_imagenet_sn.deit import deit_tiny_patch16_224_sn
    model = deit_tiny_patch16_224_sn(pretrained = (not args.scratch),num_classes =10,pen_for_qkv=args.pen_for_qkv).cuda()
    model.init_sn(seed=args.seed); model = nn.DataParallel(model)
    logger.info('Model{}'.format(model))
elif args.model == "convit_base_sn":
    from model_for_imagenet_sn.convit import convit_base_sn
    model = convit_base_sn(pretrained=(not args.scratch), num_classes =10,pen_for_qkv=args.pen_for_qkv).cuda()
    model.init_sn(seed=args.seed); model = nn.DataParallel(model)
    logger.info('Model{}'.format(model))
elif args.model == "convit_small_sn":
    from model_for_imagenet_sn.convit import convit_small_sn
    model = convit_small_sn(pretrained=(not args.scratch), num_classes =10,pen_for_qkv=args.pen_for_qkv).cuda()
    model.init_sn(seed=args.seed); model = nn.DataParallel(model)
    logger.info('Model{}'.format(model))
elif args.model == "convit_tiny_sn":
    from model_for_imagenet_sn.convit import convit_tiny_sn
    model = convit_tiny_sn(pretrained = (not args.scratch),num_classes =10,pen_for_qkv=args.pen_for_qkv).cuda()
    model.init_sn(seed=args.seed); model = nn.DataParallel(model)
    logger.info('Model{}'.format(model))
elif args.model  == "swin_tiny_patch4_window7_224_sn":
    args.momentum = 0.5
    from model_for_imagenet_sn.swin import swin_tiny_patch4_window7_224_sn
    model = swin_tiny_patch4_window7_224_sn(pretrained = (not args.scratch),num_classes =10,pen_for_qkv=args.pen_for_qkv).cuda()
    model.init_sn(seed=args.seed); model = nn.DataParallel(model)
    logger.info('Model{}'.format(model))
elif args.model  == "swin_small_patch4_window7_224_sn":
    args.momentum = 0.5
    from model_for_imagenet_sn.swin import swin_small_patch4_window7_224_sn
    model = swin_small_patch4_window7_224_sn(pretrained = (not args.scratch), num_classes =10,pen_for_qkv=args.pen_for_qkv).cuda()
    model.init_sn(seed=args.seed); model = nn.DataParallel(model)
    logger.info('Model{}'.format(model))
elif args.model  == "swin_base_patch4_window7_224_sn":
    args.momentum = 0.5
    from model_for_imagenet_sn.swin import swin_base_patch4_window7_224_sn
    model = swin_base_patch4_window7_224_sn(pretrained = (not args.scratch),num_classes =10,pen_for_qkv=args.pen_for_qkv).cuda()
    model.init_sn(seed=args.seed); model = nn.DataParallel(model)
    logger.info('Model{}'.format(model))
else:
    raise ValueError("Model doesn't exist！")
model.train()



if args.load:
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['state_dict'])
def evaluate_natural(args, model, test_loader, verbose=False):
    model.eval()
    with torch.no_grad():
        meter = MultiAverageMeter()
        def test_step(step, X_batch, y_batch):
            X, y = X_batch.cuda(), y_batch.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            meter.update('test_loss', loss.item(), y.size(0))
            meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))
        for step, (X_batch, y_batch) in enumerate(test_loader):
            test_step(step, X_batch, y_batch)
        logger.info('Evaluation {}'.format(meter))
        return round(meter.avg('test_acc'),4)


def train_adv(args, model, ds_train, ds_test, logger):
    mu = torch.tensor(imagenet_mean).view(3, 1, 1).cuda()
    std = torch.tensor(imagenet_std).view(3, 1, 1).cuda()

    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()

    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    train_loader, test_loader = ds_train, ds_test

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active :
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.labelsmoothvalue, num_classes =10,)
    if mixup_active:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    
    
    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    lr_steps = args.epochs * steps_per_epoch
    def lr_schedule(t):
        import math
        return 1e-4 + (args.lr_max - 1e-4) * (1 + (math.cos(math.pi * t / 40)))/2
    epoch_s = 0
    evaluate_natural(args, model, test_loader, verbose=False)
    for epoch in tqdm.tqdm(range(epoch_s + 1, args.epochs + 1)):
        train_loss = 0
        Spe_loss = 0
        train_acc = 0
        train_n = 0

        def train_step(X, y,t,mixup_fn):
            model.train()
            if args.method == "CLEAN":
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                output,spe_loss = model(X)
                spe_loss=spe_loss.mean()
                cls_loss = criterion(output, y)
                loss=cls_loss + spe_loss *args.trade_off
            elif args.method == 'AT':
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                def pgd_attack():
                    model.eval()
                    epsilon = epsilon_base.cuda()
                    delta = torch.zeros_like(X).cuda()
                    if args.delta_init == 'random':
                        for i in range(len(epsilon)):
                            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta.requires_grad = True
                    for _ in range(args.attack_iters):
                        output = model(X + delta)
                        loss = criterion(output, y)
                        grad = torch.autograd.grad(loss, delta)[0].detach()
                        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta = delta.detach()
                    model.train()
                    return delta
                delta = pgd_attack()
                X_adv = X + delta
                output,spe_loss = model(X_adv)
                spe_loss=spe_loss.mean()
                cls_loss = criterion(output, y)
                loss=cls_loss + spe_loss *args.trade_off
            else:
                raise ValueError(args.method)
            opt.zero_grad()
            (loss / args.accum_steps).backward()
            if args.method == 'AT' or args.method == 'CLEAN':
                acc = (output.max(1)[1] == y.max(1)[1]).float().mean()
            else:
                acc = (output.max(1)[1] == y).float().mean()
            return cls_loss,spe_loss, acc,y

        for step, (X, y) in enumerate(train_loader):
            batch_size = args.batch_size // args.accum_steps
            epoch_now = epoch - 1 + (step + 1) / len(train_loader)
            for t in range(args.accum_steps):
                X_ = X[t * batch_size:(t + 1) * batch_size].cuda()  
                y_ = y[t * batch_size:(t + 1) * batch_size].cuda()  
                if len(X_) == 0:
                    break
                cls_loss, spe_loss, acc,y = train_step(X,y,epoch_now,mixup_fn)
                train_loss += cls_loss.item() * y_.size(0)
                if spe_loss==0:
                    pass
                else:
                    Spe_loss += spe_loss.item() * y_.size(0)
                train_acc += acc.item() * y_.size(0)
                train_n += y_.size(0)
            if args.clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            else:
                pass
            opt.step()
            opt.zero_grad()
            if (step + 1) % args.log_interval == 0 or step + 1 == steps_per_epoch:
                logger.info('Training epoch {} step {}/{}, lr {:.4f} cls_loss {:.4f} spe_loss {:.6f} acc {:.4f}'.format(
                    epoch, step + 1, len(train_loader),
                    opt.param_groups[0]['lr'],
                           train_loss / train_n, Spe_loss/train_n ,train_acc / train_n
                ))
            lr = lr_schedule(epoch_now)
            opt.param_groups[0].update(lr=lr)
        path = os.path.join(args.out_dir, 'checkpoint_{}'.format(epoch))
        if args.test:
            with open(os.path.join(args.out_dir, 'test_PGD20.txt'),'a') as new:
                if args.method=="CLEAN":
                    args.eval_iters = 2 
                    args.epsilon=2
                else:
                    args.eval_iters = 20 
                args.eval_restarts = 1
                pgd_loss, pgd_acc = evaluate_pgd(args, model, test_loader)
                logger.info('test_PGD{:d} : loss {:.4f} acc {:.4f}'.format(args.eval_iters, pgd_loss, pgd_acc))
                new.write('{:.4f}   {:.4f}\n'.format(pgd_loss, pgd_acc))
            with open(os.path.join(args.out_dir, 'test_acc.txt'), 'a') as new:
                meter_test = evaluate_natural(args, model, test_loader, verbose=False)
                new.write('{}\n'.format(meter_test))
            
        else:
            with open(os.path.join(args.out_dir, 'test_acc.txt'), 'a') as new:
                meter_test = evaluate_natural(args, model, test_loader, verbose=False)
                new.write('{}\n'.format(meter_test))

        if epoch == args.epochs:
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'opt': opt.state_dict()}, path)
            logger.info('Checkpoint saved to {}'.format(path))
            



train_adv(args, model, train_loader, test_loader, logger)


def evaluate(args,model,test_loader,logger,epoch,flag="last"):
    if args.method!="CLEAN":
        args.eval_iters = 20
        logger.info(args.out_dir)
        print(args.out_dir)
        nat=evaluate_natural(args, model, test_loader, verbose=False)

        cw_loss, cw_acc = evaluate_CW(args, model, test_loader)
        logger.info('cw20 : loss {:.4f} acc {:.4f}'.format(cw_loss, cw_acc))


        pgd_loss, pgd_acc20 = evaluate_pgd(args, model, test_loader)
        logger.info('PGD20 : loss {:.4f} acc {:.4f}'.format(pgd_loss, pgd_acc20))


        args.eval_iters = 100
        pgd_loss, pgd_acc100 = evaluate_pgd(args, model, test_loader)
        logger.info('PGD100 : loss {:.4f} acc {:.4f}'.format(pgd_loss, pgd_acc100))
        
        at_path = os.path.join(args.out_dir, 'result_'+'_autoattack.txt')
        aa_acc=evaluate_aa(args, model,at_path, args.AA_batch)
        logger.info('AutoAttack : acc {:.4f}'.format( aa_acc))


    else:
        logger.info(args.out_dir)
        print(args.out_dir)
        nat=evaluate_natural(args, model, test_loader, verbose=False)
        
        args.eval_iters = 2
        args.epsilon = 2
        pgd_loss, pgd_acc2 = evaluate_pgd(args, model, test_loader)
        logger.info('PGD2 : loss {:.4f} acc {:.4f}'.format(pgd_loss, pgd_acc2))


        fgsm=validate_fgsm(args, model,logger,test_loader,'cuda',2)
        fgsm=round(fgsm.item(),4)

evaluate(args,model,test_loader,logger,epoch=args.epochs)








