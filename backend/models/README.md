# Models Directory

Place your trained PyTorch model files here:

| Model | Filename | Purpose |
|-------|----------|---------|
| P2P Scorer | `p2p_model.pth` | Predicts probability to pay |
| Compliance | `compliance_model.pth` | Detects aggressive language |
| Agency Fit | `agency_fit_model.pth` | Scores agency-case match |

## No Model? No Problem!
If a model file is missing, the system uses **rule-based fallback** logic.
