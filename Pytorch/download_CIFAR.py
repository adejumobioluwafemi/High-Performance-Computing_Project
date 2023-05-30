# import torch
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.models as models
# import torchvision
# import torchvision.transforms as transforms

# torch.manual_seed(42)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# torch.manual_seed(42)

# batch_size = 64

# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# train_set = torchvision.datasets.CIFAR10(root='./py_data', train=True, download=True, transform=train_transform)

# train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)


# test_set = torchvision.datasets.CIFAR10(root='./py_data', train=False, download=True, transform=test_transform)

# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)



# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# print('Train size', len(train_set))
# print('Test size', len(test_set))
###################################
import torch
import torchvision
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Define the path to save the dataset
path = "./py_data"

# Download and save the CIFAR10 training dataset
train_set = torchvision.datasets.CIFAR10(root=path, train=True, download=True,transform=train_transform)

# Download and save the CIFAR10 testing dataset
test_set = torchvision.datasets.CIFAR10(root=path, train=False, download=True,transform=test_transform)