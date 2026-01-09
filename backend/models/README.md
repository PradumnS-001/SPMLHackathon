# Models Directory

Place your trained PyTorch model files here:

## Expected Files

| Model | Filename | Purpose |
|-------|----------|---------|
| P2P Scorer | `p2p_model.pth` | Predicts probability to pay |
| Compliance | `compliance_model.pth` | Detects aggressive language |
| Agency Fit | `agency_fit_model.pth` | Scores agency-case match |

## Training on Kaggle

1. Copy `backend/app/ml/models.py` to your Kaggle notebook
2. Train using your dataset
3. Save with: `torch.save(model.state_dict(), "p2p_model.pth")`
4. Download and place the `.pth` file in this folder
5. Restart the backend - models will auto-load!

## Model Not Present?

If a model file is missing, the system automatically uses **rule-based fallback** logic.
This ensures the app works even without trained models.
