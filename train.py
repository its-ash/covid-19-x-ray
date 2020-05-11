from data_tranform import get_loader, plot_sample_data, create_dataset
from model import MyNet, nn
from torch import optim
import torch
from PIL import Image
from data_tranform import transformation, dataset


MODEL = None

def predict():
    from google.colab import files
    source = files.upload()
    global MODEL
    model = MODEL.cpu()
    img = Image.open(list(source.keys())[0])
    response = model(torch.unsqueeze(transformation(img),1))
    position = torch.argmax(response).item()
    print(dataset.classes[position], response[position])


def train(epoch, batch):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    dataloader = get_loader(batch)

    net = MyNet(len(dataset.classes))
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(epoch):
        net.train()
        running_loss = 0.0
        correct = 0.0
        for _, data in enumerate(dataloader):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            output = net(image)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.cpu()
            running_loss += loss.item() * image.data.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()
        accuracy = 100 * correct / len(dataloader.dataset)

        print('Epoch: {}, Avg. Loss: {}, Accuracy: {}'.format(
            epoch + 1, running_loss/len(dataloader.dataset), accuracy))

    global MODEL
    MODEL = net

    return net