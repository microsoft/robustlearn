from utils import *
from dataloader import *
from model import create_model


def main():

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--dataset', default="CIFAR100", type=str, help="dataset", choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
    parser.add_argument('--num_classes', default=100, type=int, help='num classes')
    parser.add_argument('--input_size', default=32, type=int, help='input_size')
    parser.add_argument('--patch', default=4, type=int, help='num patch (used by vit)')

    parser.add_argument('--device', default="cuda", type=str, help='device')

    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')

    parser.add_argument('--model', default="ResNet18", type=str, help='model used')
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')

    args = parser.parse_args()

    # create model
    model = create_model(args.model, args.input_size, args.num_classes, args.device, args.patch, args.resume)
    if args.dataset == "CIFAR10":
        corruption_acc_dict = evaluate_cifar_corruption(args, model, data_dir="./data/CIFAR-10-C")
        print(corruption_acc_dict)
    elif args.dataset == "CIFAR100":
        corruption_acc_dict = evaluate_cifar_corruption(args, model, data_dir="./data/CIFAR-100-C")
        print(corruption_acc_dict)
    elif args.dataset == "TinyImageNet":
        corruption_acc_dict = evaluate_tiny_corruption(args, model)
        print(corruption_acc_dict)


if __name__ == "__main__":
    main()
