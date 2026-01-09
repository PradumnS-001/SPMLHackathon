# ===================================================
# FedEx DCA - P2P Model Training Notebook
# ===================================================
# Use this notebook on Kaggle to train your P2P model
# After training, download the .pth file and place it in:
# backend/models/p2p_model.pth
# ===================================================

# %% [markdown]
# # P2P (Probability to Pay) Model Training
# 
# This notebook trains a neural network to predict the probability 
# that a debtor will pay their outstanding debt.

# %% Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# %% Define the model (MUST match backend/app/ml/models.py)
class P2PNet(nn.Module):
    """
    Neural network for Probability-to-Pay prediction.
    
    Input features (8 features):
    - debt_amount (normalized)
    - days_overdue (normalized)
    - has_dispute (0/1)
    - previous_payments (count)
    - segment_retail (0/1)
    - segment_commercial (0/1)
    - segment_international (0/1)
    - payment_history_ratio
    """
    
    def __init__(self, input_size=8, hidden_sizes=[64, 32, 16]):
        super(P2PNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# %% Load and prepare your dataset
# ===================================================
# REPLACE THIS WITH YOUR ACTUAL DATASET
# ===================================================

# Example: Generate synthetic data for demonstration
np.random.seed(42)
n_samples = 10000

# Generate synthetic features
data = {
    'debt_amount': np.random.exponential(5000, n_samples),
    'days_overdue': np.random.exponential(30, n_samples),
    'has_dispute': np.random.binomial(1, 0.15, n_samples),
    'previous_payments': np.random.poisson(1, n_samples),
    'segment': np.random.choice(['retail', 'commercial', 'international'], n_samples, p=[0.5, 0.35, 0.15]),
    'payment_history_ratio': np.random.beta(2, 5, n_samples)
}

df = pd.DataFrame(data)

# Create target variable (1 = paid, 0 = not paid)
# This is a synthetic formula - replace with your actual labels
probability = (
    0.5 
    - 0.01 * np.clip(df['days_overdue'], 0, 30) / 30
    - 0.3 * df['has_dispute']
    + 0.2 * df['previous_payments'].clip(0, 3) / 3
    + 0.1 * df['payment_history_ratio']
    - 0.1 * np.log1p(df['debt_amount']) / 10
)
df['paid'] = (np.random.random(n_samples) < probability).astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Payment rate: {df['paid'].mean():.2%}")
df.head()

# %% Feature engineering
# One-hot encode segment
df['segment_retail'] = (df['segment'] == 'retail').astype(int)
df['segment_commercial'] = (df['segment'] == 'commercial').astype(int)
df['segment_international'] = (df['segment'] == 'international').astype(int)

# Select features (must match the model input)
feature_columns = [
    'debt_amount',
    'days_overdue', 
    'has_dispute',
    'previous_payments',
    'segment_retail',
    'segment_commercial',
    'segment_international',
    'payment_history_ratio'
]

X = df[feature_columns].values
y = df['paid'].values

# Normalize numerical features
scaler = StandardScaler()
X[:, :2] = scaler.fit_transform(X[:, :2])  # Normalize debt_amount and days_overdue

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# %% Create PyTorch datasets
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# %% Initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = P2PNet(input_size=8).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print(model)

# %% Training loop
epochs = 50
train_losses = []
test_losses = []
best_loss = float('inf')

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    scheduler.step(test_loss)
    
    # Save best model
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'p2p_model.pth')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

print(f"\nBest test loss: {best_loss:.4f}")

# %% Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('P2P Model Training')
plt.legend()
plt.grid(True)
plt.savefig('training_curve.png')
plt.show()

# %% Evaluate final model
model.load_state_dict(torch.load('p2p_model.pth'))
model.eval()

with torch.no_grad():
    predictions = model(X_test_tensor.to(device)).cpu().numpy()

# Calculate accuracy
pred_classes = (predictions > 0.5).astype(int)
accuracy = (pred_classes == y_test.reshape(-1, 1)).mean()
print(f"Test Accuracy: {accuracy:.2%}")

# %% Save the model
# ===================================================
# DOWNLOAD THIS FILE AND PLACE IT IN:
# backend/models/p2p_model.pth
# ===================================================
print("\n✅ Model saved as 'p2p_model.pth'")
print("📥 Download this file and place it in: backend/models/p2p_model.pth")
print("🔄 Restart the backend server to load the new model!")
