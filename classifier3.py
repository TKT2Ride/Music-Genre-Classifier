from network3 import Genre_Model
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import os

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

path = 'cleaned/spectrograms5sec'

# to store files in a list
files = []

for (root, dirs, file) in os.walk(path):
    for f in file:
        if '.npy' in f:
            files.append(root+'/'+f)

# arr = np.load(files[0])

bad_data = []

for a in files:
    k = np.load(a)
    if (k.shape != (1, 128, 552)):
        bad_data.append(a)


class Spectrogram_Dataset(Dataset):
    def __init__(self, datapath, transform=None, bad_data=None):

        files = []
        for (root, dirs, file) in os.walk(datapath):
            for f in file:
                if '.npy' in f:
                    files.append(root+'/'+f)

        self.data = list(set(files) - set(bad_data))
        self.transform = transform

    def __getitem__(self, index):
        filename = self.data[index]
        x = np.load(filename)
        y = self.get_y(filename)

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

    def get_y(self, filename):
        if "blues" in filename:
            return 0
        elif "classical" in filename:
            return 1
        elif "country" in filename:
            return 2
        elif "disco" in filename:
            return 3
        elif "hiphop" in filename:
            return 4
        elif "jazz" in filename:
            return 5
        elif "metal" in filename:
            return 6
        elif "pop" in filename:
            return 7
        elif "reggae" in filename:
            return 8
        elif "rock" in filename:
            return 9
        else:
            return -1


trials = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300]
# trials = [300]
train_accuracies = []
test_accuracies = []

for trial in trials:
    print(trial)

    data_path = 'cleaned/spectrograms5sec'

    transform = transforms.Compose([torch.from_numpy])

    dataset = Spectrogram_Dataset(data_path, transform, bad_data)
    split_percent = 0.2
    length = len(dataset)
    test_threshold = int(length * split_percent)

    test, train = torch.utils.data.random_split(
        dataset, [test_threshold, length-test_threshold])

    net = Genre_Model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    batch_size = 32
    epochs = trial

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)

    losses = []
    train_correct = 0
    train_total = 0

    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # Uses the GPU if available
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                #             print(f'[{epoch+1},{i+1:5d}] loss: {running_loss/100}')
                losses.append(running_loss/100)
                running_loss = 0.0

    train_accuracies.append(100 * train_correct / train_total)

    #-------------------------------------------------------------------------------------------#

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)

    test_correct = 0
    test_total = 0

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
    #         print(labels, predicted)
            test_correct += (predicted == labels).sum().item()
            y_pred.extend(predicted)
            y_true.extend(labels)

    test_accuracies.append(100 * test_correct / test_total)

    #-------------------------------------------------------------------------------------------#
    plt.figure(figsize=(12, 7))
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Across Training")
    plt.savefig("loss_"+str(trial)+".png")
    plt.close()

    #-------------------------------------------------------------------------------------------#

    plt.figure(figsize=(12, 7))
    y_pred2 = []
    for i in y_pred:
        y_pred2.append(i.item())

    y_true2 = []
    for i in y_true:
        y_true2.append(i.item())

    classes = ('Blues', 'Classical', 'Country', 'Disco', 'Hiphop',
               'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock')

    cf_matrix = confusion_matrix(y_true2, y_pred2)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    hm = sn.heatmap(df_cm, annot=True)
    plt.title("")
    plt.savefig("cf_"+str(trial)+".png")
    plt.close()

#-------------------------------------------------------------------------------------------#

plt.figure(figsize=(12, 7))
plt.plot(trials, train_accuracies)
plt.plot(trials, test_accuracies)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Across Training")
plt.legend(['Training Accuracy', 'Testing Accuracy'])
plt.savefig("accuracies.png")
plt.close()
print(train_accuracies, test_accuracies, sep=":|:")

#-------------------------------------------------------------------------------------------#

# torch.save(net, "model.pt")
