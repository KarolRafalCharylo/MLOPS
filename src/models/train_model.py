import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim

# Loading the model and the data
model = MyAwesomeModel()
train_data = torch.load("data/processed/train.pth")
test_data = torch.load("data/processed/test.pth")
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

"""Hyper parameters"""
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), 1e-3)
num_epochs = 3
model.train()
# List for plotting
train_losses = []
epochs = []

# Traning loop
for epoch in range(num_epochs):
    running_loss = 0

    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_losses.append(running_loss / len(trainloader))
        epochs.append(epoch)
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

# Saving the model to models folder
print("Finished Training Trainset")
torch.save(model.state_dict(), "models/corruptmnist/trained_model.pth")

# Learning curve plot (need to add interpolation/smoothing)
plt.plot(np.array(epochs), np.array(train_losses), "r")
plt.show()
plt.savefig("reports/figures/loss.png")
