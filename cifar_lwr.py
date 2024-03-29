from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from LearningWithRetrospection import LWR
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


global_step = 0


def train(
    args, model, device, train_loader, optimizer, epoch, lwr, writer, k, snapshot=None,
):
    model.train()
    for i, (batch_idx, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if args.usekl:
            output = model(data)
            loss = lwr(batch_idx, output, target, eval=False)

        else:
            output = model(data)
            if epoch >= k:
                assert snapshot != None
                previous_output = snapshot(data)
            else:
                previous_output = None

            loss = lwr(batch_idx, output, target, previous_output, eval=False)

        loss.backward()
        optimizer.step()

        global global_step
        writer.add_scalar("train/loss", loss.item(), global_step)
        global_step += 1
        if i % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    i * len(data),
                    len(train_loader.dataset),
                    100.0 * i / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    global global_step
    writer.add_scalar("test/loss", test_loss, global_step)
    writer.add_scalar("test/acc", correct / len(test_loader.dataset), global_step)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return idx, self.ds[idx][0], self.ds[idx][1]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    parser.add_argument(
        "--usekl",
        action="store_true",
        default=True,
        help="Use KL Divergence loss | Uses L1 loss from https://arxiv.org/abs/2006.13593 if False",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset1 = datasets.CIFAR10("./", train=True, download=False, transform=transform)
    dataset1 = DatasetWrapper(dataset1)

    dataset2 = datasets.CIFAR10("./", train=False, transform=transform)
    dataset2 = DatasetWrapper(dataset2)

    lwr = LWR(
        k=5,
        update_rate=0.9,
        num_batches_per_epoch=len(dataset1) // train_kwargs["batch_size"],
        dataset_length=len(dataset1),
        output_shape=(10,),
        tau=5,
        max_epochs=20,
        softmax_dim=1,
        use_kl=args.usekl,
    )
    writer = SummaryWriter("./logs/lwr/")

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    snapshot = None

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        if (not args.usekl) and (epoch % k == 0):
            snapshot = deepcopy(model)

        train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            lwr,
            writer,
            k,
            snapshot=snapshot,
        )
        test(model, device, test_loader, writer)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_cnn.pt")


if __name__ == "__main__":
    main()
