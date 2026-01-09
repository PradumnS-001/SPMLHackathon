"""
P2P (Probability to Pay) Scorer Service.
Includes PyTorch ML model loading with rule-based fallback.
"""
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


class P2PScorer:
    """
    Probability to Pay scorer with PyTorch ML model and rule-based fallback.
    
    To use ML model:
    1. Train model on Kaggle using backend/training/train_p2p_model.py
    2. Download the .pth file
    3. Place it at backend/models/p2p_model.pth
    4. Restart the server
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_loaded = False
        self.model_path = model_path or os.getenv(
            "P2P_MODEL_PATH", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "p2p_model.pth")
        )
        self._try_load_model()
    
    def _try_load_model(self):
        """Attempt to load the PyTorch ML model, fail gracefully to fallback."""
        try:
            import torch
            from ..ml.models import P2PNet
            
            if os.path.exists(self.model_path):
                self.model = P2PNet()
                self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
                self.model.eval()
                self.model_loaded = True
                logger.info(f"✅ P2P PyTorch model loaded from {self.model_path}")
            else:
                logger.warning(f"⚠️ P2P model not found at {self.model_path}, using rule-based fallback")
        except ImportError as e:
            logger.warning(f"PyTorch not available: {e}, using rule-based fallback")
            self.model_loaded = False
        except Exception as e:
            logger.warning(f"Failed to load P2P model: {e}, using rule-based fallback")
            self.model_loaded = False

    
    def calculate_score(
        self,
        debt_amount: float,
        days_overdue: int,
        has_dispute: bool = False,
        previous_payments: int = 0,
        segment: str = "retail"
    ) -> dict:
        """
        Calculate P2P score using ML model or fallback.
        
        Returns:
            dict with 'score' (0-1), 'method' ('ml' or 'rule_based'), and 'factors'
        """
        if self.model_loaded:
            return self._ml_predict(debt_amount, days_overdue, has_dispute, previous_payments, segment)
        else:
            return self._rule_based_score(debt_amount, days_overdue, has_dispute, previous_payments, segment)
    
    def _ml_predict(
        self,
        debt_amount: float,
        days_overdue: int,
        has_dispute: bool,
        previous_payments: int,
        segment: str
    ) -> dict:
        """Use PyTorch ML model for prediction."""
        try:
            import torch
            
            # Normalize features (same normalization as training)
            normalized_debt = min(debt_amount / 50000, 1.0)
            normalized_days = min(days_overdue / 180, 1.0)
            
            # Prepare features tensor (8 features as expected by P2PNet)
            features = torch.FloatTensor([
                normalized_debt,
                normalized_days,
                1.0 if has_dispute else 0.0,
                min(previous_payments / 5.0, 1.0),  # Normalized
                1.0 if segment == "retail" else 0.0,
                1.0 if segment == "commercial" else 0.0,
                1.0 if segment == "international" else 0.0,
                0.5  # Default payment history ratio
            ]).unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                score = self.model(features).item()
            
            return {
                "score": max(0.0, min(1.0, score)),
                "method": "ml",
                "factors": {
                    "model": "pytorch_p2pnet",
                    "confidence": "high"
                }
            }
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}, falling back to rules")
            return self._rule_based_score(debt_amount, days_overdue, has_dispute, previous_payments, segment)
    
    def _rule_based_score(
        self,
        debt_amount: float,
        days_overdue: int,
        has_dispute: bool,
        previous_payments: int,
        segment: str
    ) -> dict:
        """
        Rule-based P2P scoring fallback.
        Business logic based on industry patterns.
        """
        score = 0.5  # Base probability
        factors = {}
        
        # Days overdue factor (-0.3 to +0.3)
        if days_overdue <= 15:
            adjustment = 0.3
            factors["days_overdue"] = "very_recent"
        elif days_overdue <= 30:
            adjustment = 0.15
            factors["days_overdue"] = "recent"
        elif days_overdue <= 60:
            adjustment = -0.1
            factors["days_overdue"] = "moderate"
        elif days_overdue <= 90:
            adjustment = -0.2
            factors["days_overdue"] = "high"
        else:
            adjustment = -0.3
            factors["days_overdue"] = "severe"
        score += adjustment
        
        # Debt amount factor (-0.15 to +0.15)
        if debt_amount < 500:
            score += 0.15
            factors["debt_amount"] = "low"
        elif debt_amount < 2000:
            score += 0.05
            factors["debt_amount"] = "moderate"
        elif debt_amount < 10000:
            score += 0
            factors["debt_amount"] = "standard"
        else:
            score -= 0.15
            factors["debt_amount"] = "high"
        
        # Customer history factor
        if previous_payments > 0:
            score += min(0.15, previous_payments * 0.05)
            factors["previous_payments"] = "positive_history"
        else:
            factors["previous_payments"] = "no_history"
        
        # Dispute flag
        if has_dispute:
            score -= 0.2
            factors["dispute"] = "active"
        else:
            factors["dispute"] = "none"
        
        # Segment adjustment
        segment_adjustments = {
            "retail": 0.05,      # Retail customers slightly more likely to pay
            "commercial": 0.0,   # Neutral
            "international": -0.05  # International slightly harder to collect
        }
        score += segment_adjustments.get(segment, 0)
        factors["segment"] = segment
        
        # Clamp to valid range
        final_score = max(0.0, min(1.0, score))
        
        return {
            "score": round(final_score, 3),
            "method": "rule_based",
            "factors": factors
        }
    
    def get_priority_score(self, p2p_score: float, debt_amount: float, days_overdue: int) -> float:
        """
        Calculate priority score for case assignment.
        Higher priority = should be assigned sooner.
        
        Formula: Balances recovery probability with potential value.
        """
        # Normalize debt amount (0-1 scale, capped at 50000)
        normalized_debt = min(debt_amount / 50000, 1.0)
        
        # Urgency factor based on days overdue
        if days_overdue <= 30:
            urgency = 0.3
        elif days_overdue <= 60:
            urgency = 0.5
        elif days_overdue <= 90:
            urgency = 0.8
        else:
            urgency = 1.0
        
        # Priority = weighted combination
        # High P2P + High Value + High Urgency = High Priority
        priority = (p2p_score * 0.4) + (normalized_debt * 0.3) + (urgency * 0.3)
        
        return round(max(0.0, min(1.0, priority)), 3)


# Singleton instance
_scorer_instance: Optional[P2PScorer] = None


def get_p2p_scorer() -> P2PScorer:
    """Get or create P2P scorer singleton."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = P2PScorer()
    return _scorer_instance
