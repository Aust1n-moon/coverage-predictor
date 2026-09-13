from model import NeuralNetwork
import torch, os
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


#hyperparameters
epochs = 200
batch_size = 64
learning_rate = 1e-4

#loading in the data from processed nfl data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_2023.npz')
data = np.load(data_path)
scaler = StandardScaler()
X_raw = data['X']
mask = X_raw != 0  
X_scaled = scaler.fit_transform(X_raw)
X_scaled[~mask] = 0  
X = torch.tensor(X_scaled, dtype=torch.float32)
y = torch.tensor(data['y'], dtype = torch.long)

#device config to use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

train_loader = DataLoader(
    TensorDataset(X_train, y_train), 
    batch_size = batch_size,
    shuffle = True
    )

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size = batch_size
    )

#model, loss function, optimizer
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#training loop, testing loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    #prediction and loss
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss: >7f} [{current: >5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, AVG loss: {test_loss:>8f} \n")

for epoch in range(epochs):
    print(f"Epoch{epoch+1}\n --------------")
    train_loop(train_loader,model,loss_fn,optimizer)
    test_loop(test_loader,model,loss_fn)
    print("Done")