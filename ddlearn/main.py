# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from DG_aug import DDLearn
from data_util.get_dataloader import load
import torch
import tqdm
import argparse
from utils import str2bool, param_init, print_environ
import utils


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='dsads')
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--n_act_class', type=int, default=19,
                        help='the number of the category of activities')
    parser.add_argument('--n_aug_class', type=int,
                        default=8, help='including the ori one')
    parser.add_argument('--auglossweight', type=float, default=1)
    parser.add_argument('--conweight', type=float, default=1.0)
    parser.add_argument('--dp', type=str, default='dis',
                        help='this is for oirginal and aug feature discrimination')
    parser.add_argument('--dpweight', type=float, default=10.0,
                        help='this is the weight of dp')
    parser.add_argument('--n_feature', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--epoch_based_training', type=str2bool, default=False,
                        help="Epoch-based training / Iteration-based training")
    parser.add_argument('--n_iter_per_epoch', type=int,
                        default=200, help="Used in Iteration-based training")
    parser.add_argument('--remain_data_rate', type=float, default=1.0,
                        help='The percentage of data used for training after reducing training data.')
    parser.add_argument('--scaler_method', type=str, default='minmax')
    parser.add_argument('--root_path', type=str,
                        default="/home/ddlearn/data/")
    parser.add_argument('--data_save_path', type=str,
                        default='/home/ddlearn/data/')
    parser.add_argument('--save_path', type=str,
                        default="/home/ddlearn/results/")

    args = parser.parse_args()
    args.step_per_epoch = 100000000000
    args = param_init(args)
    print_environ()
    return args


def train(model, optimizer, loaders, savename):
    train_ori_loader, train_aug_loader, val_ori_loader, _, test_ori_loader, _ = loaders
    mx = 0.0
    stop = 0
    iter_train_ori, iter_train_aug = iter(
        train_ori_loader), iter(train_aug_loader)
    for epoch in range(args.n_epoch):
        stop += 1
        model.train()
        correct_act, total_act = 0, 0
        total_loss = []
        for iter_num in range(args.step_per_epoch):
            x_ori, actlabel_ori, auglabel_ori = next(iter_train_ori)
            x_aug, actlabel_aug, auglabel_aug = next(iter_train_aug)
            x_ori, actlabel_ori, auglabel_ori = x_ori.unsqueeze(2).permute(
                0, 3, 2, 1).cuda().float(), actlabel_ori.cuda().long(), auglabel_ori.cuda().long()
            x_aug, actlabel_aug, auglabel_aug = x_aug.unsqueeze(2).permute(
                0, 3, 2, 1).cuda().float(), actlabel_aug.cuda().long(), auglabel_aug.cuda().long()
            actlabel_p, loss_c, loss_selfsup, loss_dp, con_loss = model(
                x_ori, x_aug, (actlabel_ori, actlabel_aug, auglabel_ori, auglabel_aug))
            loss = loss_c + args.auglossweight * loss_selfsup + \
                args.dpweight * loss_dp + args.conweight * con_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = utils.record_losses(
                total_loss, loss)
            correct_act, total_act = utils.record_trainingacc_labels(
                (actlabel_ori, actlabel_aug), actlabel_p, correct_act, total_act)
        acc_train_cls = utils.record_acc(
            correct_act, total_act)

        # Testing
        acc_val_act = utils.test_model(
            model, test_loader=(val_ori_loader, val_aug_loader), batch_size=args.batch_size)
        acc_test_act = utils.test_model(
            model, test_loader=(test_ori_loader, test_aug_loader), batch_size=args.batch_size)
        if acc_val_act > mx:
            mx = acc_val_act
            torch.save(model.state_dict(), savename+'.pkl')
            stop = 0
        tqdm.tqdm.write(
            f'Epoch: [{epoch+1}/{args.n_epoch}], total_loss: {sum(total_loss)/len(total_loss):.4f}, train_acc_c: {acc_train_cls:.2f}%, val_acc_c: {mx:.2f}%, test_acc_c: {acc_test_act:.2f}')
        if stop == args.early_stop:
            print('---early stop---')
            break


if __name__ == '__main__':
    args = args_parse()
    print(args)
    utils.set_random_seed(args.seed)
    savename = f'{args.save_path}{args.dataset}_{args.dp}_t{args.target}_remain{args.remain_data_rate}_{args.batch_size}_{args.lr}_{args.auglossweight}_{args.dpweight}_{args.conweight}_seed{args.seed}_{args.scaler_method}'
    train_ori_loader, train_aug_loader, val_ori_loader, val_aug_loader, test_ori_loader, test_aug_loader = load(
        args)
    model = DDLearn(n_feature=args.n_feature, n_act_class=args.n_act_class,
                    n_aug_class=args.n_aug_class, dataset=args.dataset, dp=args.dp).cuda()
    optimizer = torch.optim.Adam(model.params, lr=args.lr)
    train(model, optimizer, (train_ori_loader, train_aug_loader, val_ori_loader,
                             val_aug_loader, test_ori_loader, test_aug_loader), savename)

    # test
    acc_val_act = utils.test_model(
        model, test_loader=(val_ori_loader, val_aug_loader), batch_size=args.batch_size)
    acc_test_act = utils.test_model(model, test_loader=(test_ori_loader, test_aug_loader),
                                    model_file=savename+'.pkl', batch_size=args.batch_size)
    res = f'remain: {args.remain_data_rate}\t {args.auglossweight}\t {args.dpweight}\t{args.conweight}\t{args.lr:.6f}\t {args.batch_size:3d}\t {acc_val_act:.2f} {acc_test_act:.2f}'
    utils.write_file(
        f'{args.save_path}{args.dataset}_t{args.target}_seed{args.seed}_{args.scaler_method}'+'_testresult.txt', res)
    print(f'Test acc: {acc_test_act}')
