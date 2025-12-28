import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.mobilenet import mobilenet_v2, mobilenet_v3_small
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
import pandas as pd
# from models import *


def train(model, device, train_loader, optimizer, epoch):
    log_interval = 10
    loss_func = CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data = data.repeat(1, 3, 1, 1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def tst(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            # data = data.repeat(1, 3, 1, 1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    batch_size = 1000
    learning_rate = 0.01
    reduce_lr_gamma = 0.7
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {} Epochs: {} Batch size: {}'.format(
        device, epochs, batch_size))

    kwargs = {'batch_size': batch_size}
    if torch.cuda.is_available():
        kwargs.update({'num_workers': 1, 'pin_memory': True})

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.EMNIST('./', train=True, split='byclass',
                              download=True, transform=transform)
    dataset2 = datasets.EMNIST(
        './', train=False, split='byclass', transform=transform)
    print('Length train: {} Length test: {}'.format(
        len(dataset1), len(dataset2)))

    train_loader = torch.utils.data.DataLoader(
        dataset1, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset2, shuffle=False, **kwargs)
    print('Number of train batches: {} Number of test batches: {}'.format(
        len(train_loader), len(test_loader)))
    model=LeNet5(num_classes=62)
    # model = mobilenet_v3_small(norm_layer=torch.nn.Identity, num_classes=10)
    # model.features[0][0] = torch.nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1,bias=False)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=reduce_lr_gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        tst(model, device, test_loader)
        # scheduler.step()

    torch.save(model.state_dict(), "mnist_cnn.pt")

    # Final prediction
    ids = list(range(len(dataset2)))
    submission = pd.DataFrame(ids, columns=['id'])
    predictions = []
    real = []
    for data, target in test_loader:
        # data = data.repeat(1, 3, 1, 1)
        data = data.to(device)
        output = model(data)
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        predictions += list(pred.cpu().numpy()[:, 0])
        real += list(target.numpy())
    submission['pred'] = predictions
    submission['real'] = real
    submission.to_csv('submission.csv', index=False)
    print('Submission saved in: {}'.format('submission.csv'))


if __name__ == '__main__':
    main()
