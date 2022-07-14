from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and args.rank == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data) * args.world_size,
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        if args.verbose:
            print('Batch', batch_idx, "from rank", args.rank)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch ResNet CIFAR10 Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        metavar='N',
                        help='number of epochs to train (default: 20)')

    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='For Saving the current Model')
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='For displaying smdistributed.dataparallel-specific logs')
    parser.add_argument('--data-path',
                        type=str,
                        default='/tmp/data',
                        help='Path for downloading '
                        'the CIFAR10 dataset')

    args = parser.parse_args()
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = rank = dist.get_rank()
    local_rank = args.local_rank
    args.batch_size //= args.world_size // 8
    args.batch_size = max(args.batch_size, 1)
    print("batch size is {}".format(args.batch_size))

    if args.verbose:
        print('Hello from rank', rank, 'of local_rank', local_rank,
              'in world size of', args.world_size)

    if not torch.cuda.is_available():
        raise Exception(
            "Must run smdistributed.dataparallel MNIST example on CUDA-capable devices."
        )
    
    torch.manual_seed(args.seed)

    device = torch.device(f'cuda:{local_rank}')

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # select a single rank per node to download data
    is_first_local_rank = (local_rank == 0)
    if is_first_local_rank:
        train_dataset = datasets.CIFAR10(root="data",
                                         train=True,
                                         download=True,
                                         transform=transform)
    dist.barrier()  # prevent other ranks from accessing the data early
    if not is_first_local_rank:
        train_dataset = datasets.CIFAR10(root="data",
                                         train=True,
                                         download=False,
                                         transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)
    if rank == 0:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root="data", train=False, transform=transform),
            batch_size=args.test_batch_size,
            shuffle=True)
    model = DDP(torchvision.models.resnet50(pretrained=False).to(device),
                broadcast_buffers=False)
    torch.cuda.set_device(local_rank)
    #model.cuda(local_rank)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr * dist.get_world_size())
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        if rank == 0:
            test(model, device, test_loader)
        optimizer.step()

    if args.save_model:
        torch.save(model.state_dict(), "resnet_50.pt")


if __name__ == '__main__':
    main()