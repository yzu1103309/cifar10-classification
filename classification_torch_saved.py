import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# hyper-params
batch_size = 100
lr = 0.001
epoch = 15
seed = 12345

trans_method = transforms.Compose([
    transforms.ToTensor(),     # This will map the pixels to 0~1
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))    # map the pixels to -1~1
])

# load data
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=trans_method)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=trans_method)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class_names = train_data.classes

# Show the image
# plt.figure()
# plt.imshow(train_data[0][0].squeeze())
# plt.colorbar()
# plt.grid(False)
# plt.show()

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(seed)


# define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            # 32*32, 3
            nn.Conv2d(3, 100, kernel_size=3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 15*15, 100
            nn.Conv2d(100, 50, kernel_size=4),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 6*6, 50
            nn.Conv2d(50, 30, kernel_size=3),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 2*2, 30
            nn.Flatten(),
            nn.Linear(120, 10)
        )

    def forward(self, x):
        x = self.net(x)
        return x


# checking device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# use the best model to predict
saved_model = Net()
saved_model.load_state_dict(torch.load("model.pth"))

saved_model.eval()
with torch.no_grad():
    # id = 55
    # pred = saved_model(test_loader.dataset[id][0].reshape(1, 3, 32, 32))
    # true = test_loader.dataset[id][1]
    # pred_label = class_names[pred.argmax(1)]
    # print(f"{pred_label} ({100 * torch.max(F.softmax(pred, dim=1)):0.2f}%) / correct: {class_names[true]} ({100 * F.softmax(pred, dim=1)[0][true]:0.2f}%)")
    for X, y in test_loader:
        pred = saved_model(X)
        pred_label = pred.argmax(dim=1)
        for i, boolean in enumerate(pred_label == y):
            if not boolean:
                plt.figure()
                plt.imshow(X[i].squeeze().T * .5 + .5, cmap=plt.cm.binary)
                plt.grid(False)
                plt.xlabel(f"pred: {class_names[pred_label[i]]} / ans: {class_names[y[i]]}")
                plt.show()
                os.system("pause")
