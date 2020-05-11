from data_tranform import get_loader, plot_sample_data, create_dataset
from model import MyNet, nn
from torch import optim
import torch
from PIL import Image
from data_tranform import transformation, dataset


MODEL = None

def predict(image_name_with_path):
    global MODEL
    model = MODEL.cpu()
    img = Image.open(image_name_with_path)
    print(dataset.classes[torch.argmax(model(torch.unsqueeze(transformation(img),1))).item()])


def train(epoch, batch):
    dataloader = get_loader(batch)

    net = MyNet(len(dataset.classes))

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
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()
        accuracy = 100 * correct / len(dataloader.dataset)

        print('Epoch: {}, Avg. Loss: {}, Accuracy: {}'.format(
            epoch + 1, running_loss/len(dataloader.dataset), accuracy))

    global MODEL
    MODEL = net

    return net