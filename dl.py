import os
import time
import argparse

import torch
import torchvision
import torch.nn as nn

from wrn import WideResNet


__add__ = ['dl']


def load_model(path):
    # model is saved in pytorch
    model = WideResNet(16, 10, 2)
    model.load_state_dict(torch.load(path))
    return model


def train(train_loader, network, criterion, optimizer, epoch, logger=None):
    network.train()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        preds = network(images)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

    train_acc = num_corrects.float() / total_images
    train_loss = total_loss / total_images

    return train_acc, train_loss


def test(test_loader, network, criterion):
    network.eval()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()

        preds = network(images)

        loss = criterion(preds, labels)

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

    test_acc = num_corrects.float() / total_images
    test_loss = total_loss / total_images

    return test_acc, test_loss


def dl(opt):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            'data',
            train=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            'data',
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])),
        batch_size=opt.test_batch_size,
        num_workers=opt.num_workers)

    network = load_model(opt.network)
    network.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210])

    best_acc = 0

    for epoch in range(opt.num_epochs):
        start = time.time()
        train_acc, train_loss = train(train_loader, network, criterion, optimizer, epoch)
        end = time.time()
        print('total time: {:.2f}s - epoch: {} - accuracy: {} - loss: {}'.format(end-start, epoch, train_acc, train_loss))

        test_acc, test_loss = test(test_loader, network, criterion)

        # save the best model
        if opt.save and test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': network.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()}
            save_file = os.path.join(opt.save_folder, '{}_ckpt.pth'.format('wrn-16-2', epoch))
            torch.save(state, save_file)
            print('in test, epoch: {} - best accuracy: {} - loss: {}'.format(epoch, best_acc, test_loss))

        lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--network', default='model/wrn-16-2.pth')
    parser.add_argument('--num_epochs', default=240, type=int)
    parser.add_argument('--temperature', default=2.5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--save_folder', default='pretrained')

    dl(parser.parse_args())
