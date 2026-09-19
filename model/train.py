from model import NeuralNetwork
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# No fixed seed — model performance varies with initialization

#hyperparameters
epochs = 200
batch_size = 64
learning_rate = 1e-4
NOISE_STD = 0.1
PLAYER_DROP_PROB = 0.05


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def augment_batch(X, noise_std=NOISE_STD, drop_prob=PLAYER_DROP_PROB):
    """Add Gaussian noise and randomly zero out whole player slots to simulate pipeline noise."""
    mask = X != 0
    noise = torch.randn_like(X) * noise_std
    X = X + noise * mask.float()
    # Randomly drop entire player slots (25-feat defender or 22-feat offense blocks)
    for i in range(X.shape[0]):
        # Defenders: 8 slots × 25 features (indices 0-199)
        for slot in range(8):
            if torch.rand(1).item() < drop_prob:
                X[i, slot*25:(slot+1)*25] = 0
        # Offense: 6 slots × 22 features (indices 200-331)
        for slot in range(6):
            if torch.rand(1).item() < drop_prob:
                X[i, 200+slot*22:200+(slot+1)*22] = 0
    return X


#loading in the data from processed nfl data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_2023.npz')
data = np.load(data_path)
scaler = StandardScaler()
X_raw = data['X']
raw_mask = X_raw != 0
X_scaled = scaler.fit_transform(X_raw)
X_scaled[~raw_mask] = 0
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

save_path = os.path.join(os.path.dirname(__file__), 'coverage_model.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
}, save_path)
print(f"Saved model to {save_path}")