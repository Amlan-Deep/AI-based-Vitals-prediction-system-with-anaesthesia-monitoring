import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_excel('yashoda_data.xlsx')
X = df.iloc[:-1, 1:].values
y = df.iloc[1:, 1:].values

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

class TimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_dataset = TimeDataset(X_train, y_train)
test_dataset = TimeDataset(X_test, y_test)

batch_size=2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = torch.relu(out)
        return out
    
input_size = X.shape[1]
hidden_size = 64
output_size = y.shape[1]
lr, num_epochs = 1e-3, 100

model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)
device = torch.device('cuda' if torch.cuda().is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(X_batch.unsqueeze(1))
        
        loss = criterion(y_pred, y_batch)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    avg_loss = train_loss / len(train_loader)
    print(f'Epoch: {epoch + 1} | Loss: {avg_loss:.4f}')
    
model.eval()
test_loss = 0.0

with torch.inference_mode():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        y_pred = model(X_batch.unsqueeze(1))
        
        loss = criterion(y_pred, y_batch)
        
        test_loss += loss.item()
        
avg_test_loss = test_loss / len(test_loader)
print(f"Avg Test Loss: {avg_test_loss:.4f}")