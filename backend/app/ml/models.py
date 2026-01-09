"""
PyTorch Model Definitions for P2P Scoring
Train these models on Kaggle and export the .pth file
"""
import torch
import torch.nn as nn


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
    
    Output:
    - p2p_score (0-1)
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
        
        # Output layer (sigmoid for 0-1 probability)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ComplianceNet(nn.Module):
    """
    Neural network for compliance violation detection.
    Uses LSTM for sequence processing of text embeddings.
    
    Input: Text embeddings (batch_size, seq_len, embedding_dim)
    Output: Violation probability (0-1)
    """
    
    def __init__(self, embedding_dim=768, hidden_size=128, num_layers=2):
        super(ComplianceNet, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        # Concatenate forward and backward hidden states
        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(hidden_concat)


class AgencyFitNet(nn.Module):
    """
    Neural network for agency-case fit scoring.
    
    Input features (12 features):
    - agency_performance_score
    - agency_compliance_score
    - agency_current_load_ratio
    - agency_category_match (0/1)
    - case_debt_amount (normalized)
    - case_p2p_score
    - case_days_overdue (normalized)
    - case_segment_retail (0/1)
    - case_segment_commercial (0/1)
    - case_segment_international (0/1)
    - case_has_dispute (0/1)
    - case_priority_score
    
    Output:
    - fit_score (0-1)
    """
    
    def __init__(self, input_size=12, hidden_sizes=[32, 16]):
        super(AgencyFitNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
