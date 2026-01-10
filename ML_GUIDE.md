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

### 3. Upload Training Data
Upload `fedex_training_data.csv` to Kaggle.

### 4. Train & Export
```python
torch.save(model.state_dict(), "p2p_model.pth")
```

### 5. Deploy
Place `p2p_model.pth` in `backend/models/` and restart server.
