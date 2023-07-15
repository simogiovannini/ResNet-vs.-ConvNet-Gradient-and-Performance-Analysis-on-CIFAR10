import torch
from torch import nn

from models.convnet import ConvNet
from models.mlp import MultiLayerPerceptron
from models.resnet import ResNet50
from utils.data_loader import load_cifar10, create_dataloaders
from utils.train import train
from torch.utils.tensorboard import SummaryWriter


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter('runs/exercise-1_1')

train_data, val_data, test_data = load_cifar10()
train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_data, val_data, test_data, batch_size=32)

n_epochs = 10


mlp = MultiLayerPerceptron(input_shape=3*128*128, hidden_units=128, output_shape=10)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.1, momentum=0.9)

train(epochs=n_epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader, model=mlp, loss_fn=loss_fn, optimizer=optimizer, device=device, model_name='MLP', writer=writer, run_id=0)
torch.save(mlp.state_dict(), 'trained_models/mlp')


convnet = ConvNet()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(convnet.parameters(), lr=0.001, momentum=0.9)

train(epochs=n_epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader, model=convnet, loss_fn=loss_fn, optimizer=optimizer, device=device, model_name='ConvNet34', writer=writer, run_id=0)
torch.save(convnet.state_dict(), 'trained_models/convnet34')

resnet34 = ConvNet(is_res_net=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet34.parameters(), lr=0.001, momentum=0.9)

train(epochs=n_epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader, model=resnet34, loss_fn=loss_fn, optimizer=optimizer, device=device, model_name='ResNet34', writer=writer, run_id=0)
torch.save(resnet34.state_dict(), 'trained_models/resnet')

resnet50 = ResNet50(num_classes=10)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)

train(epochs=n_epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader, model=resnet50, loss_fn=loss_fn, optimizer=optimizer, device=device, model_name='ResNet50', writer=writer, run_id=0)
torch.save(resnet50.state_dict(), 'trained_models/resnet50')


writer.close()
