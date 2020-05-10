from model import torch, MyNet, nn
from dataset import create_dataset
from data_tranform import dataloader, plot_sample_data


create_dataset()
plot_sample_data()

net = MyNet(3)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


for epoch in range(10):
    net.train()
    running_loss = 0.0
    correct = 0.0
    for batch, data in enumerate(dataloader):
        image, label = data
        output = net(image)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * image.data.size(0)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label).sum().item()
    accuracy = 100 * correct / len(dataloader.dataset)

    print('Epoch: {}, Avg. Loss: {}, Accuracy: {}'.format(
        epoch + 1, running_loss/len(dataloader.dataset), accuracy))
