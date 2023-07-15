import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
import numpy as np
import cv2
from models.convnet import ConvNet
from utils.data_loader import create_dataloaders, load_cifar10


def save_image_grid(images, labels, class_names, file_name):
    rows, columns = 4, 4
    fig = plt.figure(figsize=(10, 10))

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        image = images[i - 1]
        image = torch.permute(image, (1, 2, 0)) / 2 + 0.5
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.title("{}".format(class_names[labels[i - 1]]))
    plt.savefig(f'CAMs/{file_name}.png')


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (128, 128)
    bz, nc, h, w = feature_conv.shape

    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cv2.resize(cam_img, size_upsample)


train_data, val_data, test_data = load_cifar10()
train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_data, val_data, test_data, batch_size=1024)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

image_batch = next(iter(test_dataloader))

images = image_batch[0]
labels = image_batch[1]

imageId = np.random.randint(0, len(images), 16)
images = images[imageId]
labels = [labels[i] for i in imageId]

save_image_grid(images, labels, class_names, file_name='pre_image_grid')

net = ConvNet()
net.load_state_dict(torch.load('trained_models/convnet34'))
net.eval()

finalconv_name = 'layer4'

features_blobs = []
net._modules.get(finalconv_name).register_forward_hook(hook_feature)

params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

for k in range(len(images)):
    img = images[k][None, :, :, :]
    logit = net(img)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    img = torch.permute(img[0], (1, 2, 0)) / 2 + 0.5
    img = img.numpy()
    img = np.uint8(255 * img)

    CAM = returnCAM(features_blobs[0], weight_softmax, idx[0])
    heatmap = cv2.applyColorMap(cv2.resize(CAM, (128, 128)), cv2.COLORMAP_JET)

    result = heatmap * 0.3 + img * 0.5

    cv2.imwrite(f'CAMs/{k + 1}-{class_names[idx[0]]}.jpg', result)

