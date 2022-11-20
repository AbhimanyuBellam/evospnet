
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse

import numpy as np
import matplotlib.pyplot as plt

from base_net import BasicNet

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# parser = argparse.ArgumentParser()

# parser.add_argument(-)

batch_size = 100
epochs = 50


train_dataset = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

validation_dataset = datasets.MNIST('./data',
                                    train=False,
                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

model = BasicNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

print(model)


def train(losst, acct, epoch, model_save_path, log_interval=200):
    # Set model to training mode
    model.train()

    epoch_loss = 0
    train_acc = 0
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.view(-1, 28*28)
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model.forward(data)

        # Calculate loss
        loss = criterion(output, target)

        epoch_loss += loss.data.item()
        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

        predicted = torch.max(output.data, 1)[1]
        train_acc += (predicted == target).sum()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

    train_acc = (train_acc)/(batch_idx*batch_size)
    print("Train acc:", train_acc)
    # gen_train_loss_total = sum(gen_train_loss)
    epoch_loss /= len(train_loader)
    print("Train loss:", epoch_loss)
    losst.append(epoch_loss)
    acct.append(train_acc)

    if epoch == epochs:
        torch.save(model.state_dict(), model_save_path)


def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.view(-1, 28*28)
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            val_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

        val_loss /= len(validation_loader)
        loss_vector.append(val_loss)

        accuracy = 100. * correct.to(torch.float32) / \
            len(validation_loader.dataset)
        accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))


lossv, accv = [], []
losst, acct = [], []


for i in range(56):
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model_save_path = f"results/basic_net_res/weights/sgd/basicnet/basicnet_{i}.pth"
        train(losst, acct, epoch, model_save_path)
        validate(lossv, accv)

    print("Time:", time.time()-start_time)

x = [i+1 for i in range(epochs)]

# train
file_path = "results/sgd/plots/single_train_basicnet.jpg"
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, losst, 'b-')
ax2.plot(x, acct, 'g-')

ax1.set_xlabel("epochs")
ax1.set_ylabel("Train Cross Entropy Loss", color='b')
ax2.set_ylabel("Train Accuracy", color='g')
fig.savefig(file_path)

# test
file_path = "results/sgd/plots/single_test_basicnet.jpg"
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, lossv, 'b-')
ax2.plot(x, accv, 'g-')

ax1.set_xlabel("epochs")
ax1.set_ylabel("Test Cross Entropy Loss", color='b')
ax2.set_ylabel("Test Accuracy", color='g')
fig.savefig(file_path)


# plt.figure(figsize=(5, 3))
# plt.plot(np.arange(1, epochs+1), lossv)
# plt.title('validation loss')

# plt.figure(figsize=(5, 3))
# plt.plot(np.arange(1, epochs+1), accv)
# plt.title('validation accuracy')
