import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models.convnet import ConvNet
from utils.data_loader import load_cifar10, create_dataloaders
from utils.train import train

device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter('runs/exercise-2_1')

train_data, val_data, test_data = load_cifar10()
train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_data, val_data, test_data, batch_size=32)

n_runs = 10

for i in range(n_runs):
    n_epochs = 10

    convnet = ConvNet()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(convnet.parameters(), lr=0.001, momentum=0.9)

    train(epochs=n_epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader, model=convnet, loss_fn=loss_fn, optimizer=optimizer, device=device, model_name='ConvNet34', writer=writer, save_gradients=True, run_id=i)

    resnet34 = ConvNet(is_res_net=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet34.parameters(), lr=0.001, momentum=0.9)

    train(epochs=n_epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader, model=resnet34, loss_fn=loss_fn, optimizer=optimizer, device=device, model_name='ResNet34', writer=writer, save_gradients=True, run_id=i)

