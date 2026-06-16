import torch
import matplotlib.pyplot as plt

import numpy as np
import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(f'Device: {device}')

IN_N = 10
OUT_N = 1

class NeuralNet(torch.nn.Module):    
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(IN_N, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),

            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            
            torch.nn.Linear(64, OUT_N)
        )

    def forward(self, X):
        return self.model(X)


EPOCHS = 10000

def fit(model, X, y, loss_fn, optimizer):
    
    model.train()

    for epoch in range(EPOCHS):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Loss at [{epoch + 1}/{EPOCHS}] {loss.item()}')


X = torch.rand((100, IN_N))                             # 100 rows
y = (torch.sum(X, dim=1) > 5).float().unsqueeze(1)      # simple sum > 5 comparator output

X = X.to(device)
y = y.to(device)

model = NeuralNet().to(device)
st = time.time()
fit(model, X, y, torch.nn.BCEWithLogitsLoss(), torch.optim.Adam(model.parameters()))
ed = time.time()

print(f'Trained for {ed - st}')

# model.eval()
# y_hat = torch.sigmoid(model(X)).detach().cpu().numpy().flatten()
# y = y.detach().cpu().numpy().flatten()

# sorted = np.argsort(y)

# plt.plot(y_hat[sorted], label='Pred')
# plt.plot(y[sorted], label='True')
# plt.legend()
# plt.show()  