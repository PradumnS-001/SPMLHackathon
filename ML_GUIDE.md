# ML Training Guide for DCA Management System

## 👋 Hey! Here's what you need to do for the ML part.

The app already works with **rule-based fallback** logic. Your job is to train better ML models that will replace the fallback when deployed.

---

## 🎯 Models to Train

| Model | Priority | Purpose | Input Features |
|-------|----------|---------|----------------|
| **P2P Scorer** | HIGH | Predict probability customer will pay | 8 features |
| **Compliance** | MEDIUM | Detect aggressive language in transcripts | Text embeddings |
| **Agency Fit** | LOW | Score how well agency matches a case | 12 features |

---

## 📋 Step-by-Step: Train P2P Model

### 1. Open Kaggle
Go to [kaggle.com](https://kaggle.com) and create a new notebook.

### 2. Copy the Model Class
Copy this **exactly** from `backend/app/ml/models.py`:

```python
import torch
import torch.nn as nn

class P2PNet(nn.Module):
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
```

> ⚠️ **IMPORTANT**: The class must be IDENTICAL or `load_state_dict()` fails!

### 3. Prepare Your Features

The model expects **8 input features** (normalized 0-1):

```python
features = [
    debt_amount / 50000,          # Normalized amount
    days_overdue / 180,           # Normalized days
    1 if has_dispute else 0,      # Binary
    previous_payments / 5,         # Normalized count
    1 if segment == "retail" else 0,
    1 if segment == "commercial" else 0,
    1 if segment == "international" else 0,
    payment_history_ratio          # 0-1 ratio
]
```

### 4. Train the Model

Use the template at `backend/training/train_p2p_model.py` as reference.

Basic training loop:
```python
model = P2PNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(50):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

### 5. Save the Model

```python
torch.save(model.state_dict(), "p2p_model.pth")
```

### 6. Deploy to Project

1. Download `p2p_model.pth` from Kaggle
2. Place it in: `backend/models/p2p_model.pth`
3. Restart the backend server
4. Done! Model auto-loads ✅

---

## 📊 Dataset You'll Need

For P2P model, you need historical data with:

| Column | Type | Description |
|--------|------|-------------|
| debt_amount | float | Amount owed |
| days_overdue | int | Days past due |
| has_dispute | bool | Customer disputed? |
| previous_payments | int | Past payment count |
| segment | str | retail/commercial/international |
| payment_history_ratio | float | Past payment success rate |
| **paid** | bool | **TARGET**: Did they pay? |

---

## 🔄 Testing Your Model Locally

Before deploying, test in Kaggle:

```python
# Load and test
model = P2PNet()
model.load_state_dict(torch.load("p2p_model.pth"))
model.eval()

# Test prediction
test_input = torch.FloatTensor([[0.1, 0.2, 0, 0.2, 1, 0, 0, 0.5]])
with torch.no_grad():
    prediction = model(test_input)
    print(f"P2P Score: {prediction.item():.2%}")
```

---

## ❓ Questions?

- Model class location: `backend/app/ml/models.py`
- Training template: `backend/training/train_p2p_model.py`
- Model destination: `backend/models/p2p_model.pth`

The app works without models (uses rule-based fallback), so take your time to train a good model! 🚀
