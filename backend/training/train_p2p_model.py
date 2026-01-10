# ===================================================
# FedEx DCA - P2P Model Training (Kaggle Notebook)
# ===================================================
# Uses realistic FedEx ERP-style data
# Upload fedex_training_data.csv to Kaggle and run this
# ===================================================

# %% [markdown]
# # P2P (Probability to Pay) Model Training
# 
# This notebook trains a neural network to predict the probability 
# that a customer will pay their outstanding FedEx invoice.

# %% Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# %% Define the model (MUST match backend/app/ml/models.py)
class P2PNet(nn.Module):
    """
    Neural network for Probability-to-Pay prediction.
    
    Input features (8 features):
    - outstanding_balance (normalized)
    - days_past_due (normalized)
    - dispute_flag (0/1)
    - payment_count (normalized)
    - segment_retail (0/1)
    - segment_commercial (0/1)
    - segment_international (0/1)
    - risk_score (0-1)
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


# %% Load FedEx training data
# ===================================================
# Upload fedex_training_data.csv to Kaggle first!
# ===================================================

df = pd.read_csv('fedex_training_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nPayment rate: {df['paid'].mean():.2%}")
print(f"\nColumn info:")
print(df.info())
df.head()


# %% Feature Engineering
print("Preparing features...")

# Create segment one-hot encoding
df['segment_retail'] = (df['customer_segment'] == 'retail').astype(int)
df['segment_commercial'] = (df['customer_segment'] == 'commercial').astype(int)
df['segment_international'] = (df['customer_segment'] == 'international').astype(int)

# Normalize numerical features
scaler = StandardScaler()

# Select and prepare features (8 features to match P2PNet)
feature_columns = [
    'outstanding_balance',
    'days_past_due', 
    'dispute_flag',
    'payment_count',
    'segment_retail',
    'segment_commercial',
    'segment_international',
    'risk_score'
]

# Create feature matrix
X = df[feature_columns].copy()

# Normalize continuous features
X['outstanding_balance'] = scaler.fit_transform(X[['outstanding_balance']])
X['days_past_due'] = scaler.fit_transform(X[['days_past_due']])
X['payment_count'] = X['payment_count'] / X['payment_count'].max()

X = X.values.astype(np.float32)
y = df['paid'].values.astype(np.float32)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution: {np.bincount(y.astype(int))}")


# %% Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Training payment rate: {y_train.mean():.2%}")
print(f"Test payment rate: {y_test.mean():.2%}")


# %% Create PyTorch DataLoaders
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# %% Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = P2PNet(input_size=8).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

print("\nModel architecture:")
print(model)


# %% Training Loop
epochs = 100
train_losses = []
test_losses = []
test_accs = []
best_loss = float('inf')
patience = 20
patience_counter = 0

print("\nStarting training...")
for epoch in range(epochs):
    # Training phase
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
    
    # Evaluation phase
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = correct / total
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    scheduler.step(test_loss)
    
    # Early stopping & best model saving
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'p2p_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Acc: {test_acc:.2%}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\nBest test loss: {best_loss:.4f}")


# %% Plot Training Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
axes[0].plot(train_losses, label='Train Loss', color='#4D148C')
axes[0].plot(test_losses, label='Test Loss', color='#FF6600')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (BCE)')
axes[0].set_title('Training & Test Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy curve
axes[1].plot(test_accs, label='Test Accuracy', color='#10B981')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Test Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()


# %% Final Evaluation
model.load_state_dict(torch.load('p2p_model.pth'))
model.eval()

with torch.no_grad():
    predictions = model(X_test_tensor.to(device)).cpu().numpy()

pred_classes = (predictions > 0.5).astype(int).flatten()
true_classes = y_test.astype(int)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_classes, pred_classes)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(true_classes, pred_classes, target_names=['Will NOT Pay', 'Will Pay']))


# %% Test Sample Predictions
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

sample_indices = [0, 5, 10, 15, 20]
for idx in sample_indices:
    pred = predictions[idx][0]
    actual = y_test[idx]
    print(f"Sample {idx}: Predicted P2P={pred:.2%}, Actual={'Paid' if actual else 'Not Paid'}")


# %% Export Model
print("\n" + "="*60)
print("MODEL EXPORT")
print("="*60)
print("✅ Model saved as 'p2p_model.pth'")
print("📥 Download this file from Kaggle")
print("📁 Place it in: backend/models/p2p_model.pth")
print("🔄 Restart backend server to load new model")
print("="*60)
