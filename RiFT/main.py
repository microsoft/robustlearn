# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy
from tqdm import tqdm

from utils import *
from dataloader import *
from model import create_model
from optimizer import *

from robustbench.utils import load_model


def generate_adv_dataset(args, model):
    adv_train_dataset = adv_dataset()

    model = model.eval()
    atk_model = torchattacks.PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
    transform_test = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    if args.dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == "CIFAR100":
        train_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        train_dataset = TinyImageNet("train", transform_test)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)

    for images, labels in trainloader:

        images = images.to(args.device)
        labels = labels.to(args.device)
        
        adv_images = atk_model(images, labels)  
        adv_train_dataset.append_data(adv_images, labels)

    return adv_train_dataset


def layer_sharpness(args, model, epsilon=0.1):
    
    if "CIFAR" in args.dataset:
        norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    else:
        norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model = nn.Sequential(norm_layer, model).to(args.device)
    
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(generate_adv_dataset(args, deepcopy(model)), batch_size=512, shuffle=True, num_workers=0)
    origin_total = 0
    origin_loss = 0.0
    origin_acc = 0
    with torch.no_grad():
        model.eval()
        

        for inputs, targets in trainloader:
            outputs = model(inputs)
            origin_total += targets.shape[0]
            origin_loss += criterion(outputs, targets).item() * targets.shape[0]
            _, predicted = outputs.max(1)
            origin_acc += predicted.eq(targets).sum().item()        
        
        origin_acc /= origin_total
        origin_loss /= origin_total

    args.logger.info("{:35}, Robust Loss: {:10.2f}, Robust Acc: {:10.2f}".format("Origin", origin_loss, origin_acc*100))

    model.eval()
    layer_sharpness_dict = {} 
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # print(name)
            # For WideResNet
            if "sub" in name:
                continue
            layer_sharpness_dict[name] = 1e10

    for layer_name, _ in model.named_parameters():
        if "weight" in layer_name and layer_name[:-len(".weight")] in layer_sharpness_dict.keys():
            # print(layer_name)
            cloned_model = deepcopy(model)
            # set requires_grad sign for each layer
            for name, param in cloned_model.named_parameters():
                # print(name)
                if name == layer_name:
                    # print(name)
                    param.requires_grad = True
                    init_param = param.detach().clone()
                else:
                    param.requires_grad = False
        
            optimizer = torch.optim.SGD(cloned_model.parameters(), lr=1)

            max_loss = 0.0
            min_acc = 0
    
            for epoch in range(10):
                # Gradient ascent
                for inputs, targets in trainloader:
                    optimizer.zero_grad()
                    outputs = cloned_model(inputs)
                    loss = -1 * criterion(outputs, targets) 
                    loss.backward()
                    optimizer.step()
                sd = cloned_model.state_dict()
                diff = sd[layer_name] - init_param
                times = torch.linalg.norm(diff)/torch.linalg.norm(init_param)
                # print(times)
                if times > epsilon:
                    diff = diff / times * epsilon
                    sd[layer_name] = deepcopy(init_param + diff)
                    cloned_model.load_state_dict(sd)

                with torch.no_grad():
                    total = 0
                    total_loss = 0.0
                    correct = 0
                    for inputs, targets in trainloader:
                        outputs = cloned_model(inputs)
                        total += targets.shape[0]
                        total_loss += criterion(outputs, targets).item() * targets.shape[0]
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(targets).sum().item()  
                    
                    total_loss /= total
                    correct /= total

                if total_loss > max_loss:
                    max_loss = total_loss
                    min_acc = correct
            
            layer_sharpness_dict[layer_name[:-len(".weight")]] = max_loss - origin_loss
            args.logger.info("{:35}, MRC: {:10.2f}, Dropped Robust Acc: {:10.2f}".format(layer_name[:-len(".weight")], max_loss-origin_loss, (origin_acc-min_acc)*100))

    sorted_layer_sharpness = sorted(layer_sharpness_dict.items(), key=lambda x:x[1])
    for (k, v) in sorted_layer_sharpness:
        args.logger.info("{:35}, Robust Loss: {:10.2f}".format(k, v))
    
    return sorted_layer_sharpness


def train(args, model, dataloader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item() * targets.size(0)

        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss / total, correct / total * 100


def main():

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--model', default="ResNet18", type=str, help='model used')
    parser.add_argument('--dataset', default="CIFAR10", type=str, help="dataset", choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
    parser.add_argument('--num_classes', default=10, type=int, help='num classes')
    parser.add_argument('--input_size', default=32, type=int, help='input_size')
    parser.add_argument('--layer', default=None, type=str, help='Trainable layer')
    parser.add_argument("--cal_mrc", action="store_true", help='If to calculate Module Robust Criticality (MRC) value of each module.')
    
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--patch', default=4, type=int, help='num patch (used by vit)')
    parser.add_argument('--optim', default="SGDM", type=str, help="optimizer")

    parser.add_argument('--device', default="cuda", type=str, help='device')
    
    parser.add_argument('--lr_scheduler', default="step", choices=["step", 'cosine'])

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGDM')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=float, help='lr_decay_gamma')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--epochs', default=10, type=int, help='num of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    proj_name = "rift"
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    suffix = '{}_{}_lr={}_wd={}_epochs={}_{}'.format(proj_name, args.optim, args.lr, args.wd, args.epochs, args.layer)
    model_save_dir = './results/{}_{}/checkpoint/'.format(args.model, args.dataset) + suffix + "/"

    for path in [model_save_dir]:
        if not os.path.isdir(path):
            os.makedirs(path)

    logger = create_logger(model_save_dir+'output.log')
    logger.info(args)

    args.logger = logger

    # create dataloader
    logger.info('==> Preparing data and create dataloaders...')
    if "CIFAR" in args.dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    transform_dict = {"train": transform_train, "test": transform_test}

    trainloader, _, testloader = create_dataloader(args.dataset, args.batch_size, use_val=False, transform_dict=transform_dict)

    logger.info('==> Building dataloaders...')
    logger.info(args.dataset)

    # create model
    logger.info('==> Building model...')
    model = create_model(args.model, args.input_size, args.num_classes, args.device, args.patch, args.resume)
    logger.info(args.model)

    logger.info('==> Building optimizer and learning rate scheduler...')
    optimizer = create_optimizer(args.optim, model, args.lr, args.momentum, weight_decay=args.wd)
    logger.info(optimizer)
    lr_decays = [int(args.epochs // 2)]
    scheduler = create_scheduler(args, optimizer, lr_decays=lr_decays)
    logger.info(scheduler)

    criterion = nn.CrossEntropyLoss()

    init_sd = deepcopy(model.state_dict())
    torch.save(init_sd, model_save_dir + "init_params.pth")

    if "CIFAR" in args.dataset:
        evalulate_robustness = evaluate_cifar_robustness
    else:
        evalulate_robustness = evaluate_tiny_robustness


    if args.cal_mrc:    
        layer_sharpness(args, deepcopy(model), epsilon=0.1)
        exit()

    assert args.layer is not None

    for name, param in model.named_parameters():
        param.requires_grad = False
        if args.layer in name:
            param.requires_grad = True

    _, train_acc = evaluate(args, model, trainloader, criterion)
    _, test_acc = evaluate(args, model, testloader, criterion)
    test_robust_acc = evalulate_robustness(args, model)
    logger.info("==> Init train acc: {:.2f}%, test acc: {:.2f}%, robust acc: {:.2f}%".format(train_acc, test_acc, test_robust_acc))


    for epoch in range(start_epoch, start_epoch + args.epochs):

        logger.info("==> Epoch {}".format(epoch))
        logger.info("==> Training...")
        train_loss, train_acc = train(args, model, trainloader, optimizer, criterion)

        logger.info("==> Train loss: {:.2f}, train acc: {:.2f}%".format(train_loss, train_acc))

        logger.info("==> Testing...")
        test_loss, test_acc = evaluate(args, model, testloader, criterion)

        logger.info("==> Test loss: {:.2f}, test acc: {:.2f}%".format(test_loss, test_acc))

        state = {
            'model': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if test_acc > best_acc:
            best_acc = test_acc
            params = "best_params.pth"
            logger.info('==> Saving best params...')
            torch.save(state, model_save_dir + params)
        else:
            if epoch % 2 == 0:
                params = "epoch{}_params.pth".format(epoch)
                logger.info('==> Saving checkpoints...')
                torch.save(state, model_save_dir + params)

        scheduler.step()

    checkpoint = torch.load(model_save_dir + "best_params.pth")
    model.load_state_dict(checkpoint["model"])

    test_loss, test_acc = evaluate(args, model, testloader, criterion)
    
    test_robust_acc = evalulate_robustness(args, model)

    logger.info("==> Finetune test acc: {:.2f}%, robust acc: {:.2f}".format(test_acc, test_robust_acc))

    logger.info(interpolation(args, logger, init_sd, deepcopy(model.state_dict()), model, testloader, criterion, model_save_dir, evalulate_robustness))


if __name__ == "__main__":
    main()






