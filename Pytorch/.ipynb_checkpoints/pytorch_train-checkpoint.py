import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
# from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# train_set = torchvision.datasets.CIFAR10(root='./cifar_train', train=True, download=False, transform=train_transform)

# train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)


# test_set = torchvision.datasets.CIFAR10(root='./cifar_test', train=False, download=False, transform=test_transform)

# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

# import torch
# import torchvision

# Define the path where the dataset is saved



path = "./py_data"

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the CIFAR10 training dataset
train_set = torchvision.datasets.CIFAR10(root=path, train=True, download=False,transform=train_transform)

# Load the CIFAR10 testing dataset
test_set = torchvision.datasets.CIFAR10(root=path, train=False, download=False, transform=test_transform)

# Create data loaders to iterate over the datasets
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)




def epoch_train(loader, clf, criterion, opt):
 
    clf.train(True)  # set the model to training mode
    total_loss, total_acc = [], []
    for X, y_true in loader:
        X, y_true = X.to(device), y_true.to(device)

        # forward pass and compute the loss
        y_pred = clf.forward(X)
        loss = criterion(y_pred, y_true)
        __, best_pred  = y_pred.max(axis=1)

        # zero the gradients
        opt.zero_grad()
        # backward pass and optimization step
        loss.backward()
        opt.step()

        # update the loss and accuracy
        total_loss.append(loss.item())
        total_acc.append(sum((best_pred == y_true)/ len(y_true)).item())
        

    avg_loss = sum(total_loss)/ len(loader)
    avg_acc = sum(total_acc) / len(loader)

    return avg_loss, avg_acc

def epoch_test(loader, clf, criterion):

    clf.eval()  # set the model to evaluation mode
    total_loss, total_acc = [], []
    
    for X, y_true in loader:
        X, y_true = X.to(device), y_true.to(device)

        # forward pass and compute the loss
        y_pred = clf.forward(X)
        loss = criterion(y_pred, y_true)
        __, best_pred  = y_pred.max(axis=1)
       
        # update the loss and accuracy
        total_loss.append(loss.item())
        total_acc.append(sum((best_pred == y_true)/ len(y_true)).item())
       
    avg_loss = sum(total_loss)/ len(loader)
    avg_acc = sum(total_acc) / len(loader)

    return avg_loss, avg_acc


# The function which you are going to use for model training
def train(train_loader, test_loader, clf, criterion, opt, n_epochs=50):
    # for epoch in tqdm(range(n_epochs)):
    for epoch in range(n_epochs):    
        train_loss, train_acc = epoch_train(train_loader, clf, criterion, opt)
        test_loss, test_acc = epoch_test(test_loader, clf, criterion)

        print(f'[Epoch {epoch + 1}] train loss: {train_loss:.3f}; train acc: {train_acc:.2f}; ' + 
              f'test loss: {test_loss:.3f}; test acc: {test_acc:.2f}')
        
        
        
        
        
        
        
class MLP(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            nn.Flatten(), 
            nn.Linear(3*32*32, 2048),
            nn.ReLU(),

            nn.Linear(2048,1024), 
            nn.ReLU(),

            nn.Linear(1024,512), 
            nn.ReLU(),  

            nn.Linear(512, num_classes)  
        )

    def forward(self, x):
        return self.layers(x)
    
clf_mlp = MLP(num_classes=10).cuda()
# print('Number of weights:', np.sum([np.prod(p.shape) for p in clf_mlp.parameters()]))

# Check that the output size of the network is BATCH_SIZE x NUM_CLASSES
X = next(iter(train_loader))[0].cuda()
with torch.no_grad():
    clf_X = clf_mlp(X)
    assert len(clf_X) == len(X)
    assert clf_X.shape[1] == 10

    
    
clf_mlp = clf_mlp.to(device)
opt = torch.optim.SGD(clf_mlp.parameters(), lr=0.0001) 

criterion = nn.CrossEntropyLoss()

train(train_loader, test_loader, clf_mlp, criterion, opt, n_epochs=5000)