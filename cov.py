import torch
import torchvision
import torchvision.transforms as transforms
from perceiver_pytorch import Perceiver
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys
from time import time
import os
import subprocess as sp
torch.multiprocessing.freeze_support()
if not os.path.isfile("cov_per.txt"):
    with open("cov_per.txt","w") as f:
        f.write("")

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    return f'{prefix}|{bar}|{percent}%{suffix}'

def exp_lr_scheduler(optimizer, epoch, lr_decay=0.05, lr_decay_epoch=10):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


model = Perceiver(
    input_channels = 3,          # number of channels for each token of the input
    input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
    num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
    max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
    freq_base = 2,
    depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
                                 #   depth * (cross attention -> self_per_cross_attn * self attention)
    num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,         # number of dimensions per cross attention head
    latent_dim_head = 64,        # number of dimensions per latent self attention head
    num_classes = 2,             # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
    fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
    self_per_cross_attn = 2      # number of self attention blocks per cross attention
)

img = torch.randn(1, 128, 128, 3) # 1 imagenet image, pixelized
epochs_true = 0

if os.path.isfile("epochs_true.txt"):
    with open("epochs_true.txt","r") as f:
        epochs_true_raw = f.read()
        if epochs_true_raw:
            epochs_true = int(epochs_true_raw)
print(f"True Epoch: {epochs_true}")

X_train = torch.from_numpy(np.load("X_train.npy"))
y_train = torch.from_numpy(np.load("y_train.npy"))#.reshape(-1,1))#.to(torch.float32)
X_test = torch.from_numpy(np.load("X_test.npy"))
y_test = torch.from_numpy(np.load("y_test.npy"))#.reshape(-1,1))#.to(torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("DEVICE",device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64
trainset = torch.utils.data.TensorDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)                
testset = torch.utils.data.TensorDataset(X_test, y_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)

classes = ("COVID","NORMAL")
model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if os.path.isfile(f"per_model_{epochs_true}.h5"):
    print("Loaded Saved Model")
    checkpoint = torch.load(f"per_model_{epochs_true}.h5")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # model = torch.load(f"per_model_{epochs_true}.h5")
    torch.save(checkpoint,f"per_model_backup_{epochs_true}.h5")
else:
    print("Saved MODEL not found!")

limit_files = 999999
epochs = 500

try:
    for epoch in range(epochs_true, epochs):  # loop over the dataset multiple times
        count = 0
        accuracy_count = 0
        running_loss = 0.0
        accuracy = 0
        exec_time = 0
        epoch_time = time()
        for i, data in enumerate(trainloader, 0):
            start = time()
            inputs, labels = data
            labels = labels.long().to(device)
            inputs = inputs.reshape(-1,128,128,3).to(device)
            optimizer.zero_grad()
            outputs = model(inputs).to(device)

            for x,y in zip(outputs.reshape(-1),labels.reshape(-1)):
                if int(x)==int(y):
                    accuracy_count+=1

            loss = criterion(outputs, labels).to(device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            inputs.detach()
            labels.detach()
            outputs.detach()
            loss.detach()
            if i*batch_size>=limit_files:
                break
            count+=1
            end = time()
            dur =end-start
            exec_time = (dur+0.9*exec_time)/2
            rem_time = round((X_train.shape[0]//batch_size - count)*exec_time,2)
            sys.stdout.write("\r\033[F")
            print('{:d}/{:d} {} {}s rem, epoch: {:d}, loss: {:.4f}, acc: {:.4f}'.format(i+1,X_train.shape[0]//batch_size,printProgressBar(i+1,X_train.shape[0]//batch_size,length=50),rem_time,epoch + 1, running_loss/count,accuracy_count/(count*batch_size)),end="\r")

        print('{:d}/{:d} {} {:.2f}s taken, epoch: {:d}, loss: {:.4f}, acc: {:.4f}'.format(i+1,X_train.shape[0]//batch_size,printProgressBar(i+1,X_train.shape[0]//batch_size,length=50),time()-epoch_time,epoch + 1, running_loss/count,accuracy_count/(count*batch_size)))
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict() }
        torch.save(state, f"per_model_{epochs_true}.h5")
        # torch.save(model, f"per_model_{epochs_true}.h5")
        epochs_true+=1
        # optimizer = exp_lr_scheduler(optimizer, epochs_true, lr_decay=0.1, lr_decay_epoch=10)
        with open("cov_per.txt","a") as f:
            f.write('{:.2f}s taken, epoch: {:d}, loss: {:.4f}, acc: {:.4f}\n'.format(time()-epoch_time,epoch + 1, running_loss/count,accuracy_count/(count*batch_size)))
        with open("epochs_true.txt","w") as f:
            # print("Writing epochs_true",epochs_true)
            f.write(str(epochs_true))
    print('Finished Training')
    sp.run("git add .".split(" "))
    sp.run("git commit -m \"uploaded successfully\"".split(" "))
    sp.run("git push origin master".split(" "))
    print("Uploaded files to GIT")

except KeyboardInterrupt:
    torch.save(model, f"per_model_interrupted_{epochs_true}.h5")
    sp.run("git add .".split(" "))
    sp.run("git commit -m \"uploaded successfully\"".split(" "))
    sp.run("git push origin master".split(" "))
    print("Uploaded files to GIT")

