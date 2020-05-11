from data_tranform import get_loader, plot_sample_data
from model import MyNet, nn
from torch import argmax as t_max, optim


def train(epoch, batch):
    dataloader = get_loader(batch)

    net = MyNet(3)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(epoch):
        net.train()
        running_loss = 0.0
        correct = 0.0
        for _, data in enumerate(dataloader):
            image, label = data
            output = net(image)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * image.data.size(0)
            _, predicted = t_max(output.data, 1)
            correct += (predicted == label).sum().item()
        accuracy = 100 * correct / len(dataloader.dataset)

        print('Epoch: {}, Avg. Loss: {}, Accuracy: {}'.format(
            epoch + 1, running_loss/len(dataloader.dataset), accuracy))
    return net