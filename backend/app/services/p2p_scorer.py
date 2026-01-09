"""
P2P (Probability to Pay) Scorer Service.
Includes ML model loading with rule-based fallback.
"""
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


class P2PScorer:
    """
    Probability to Pay scorer with ML model and rule-based fallback.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_loaded = False
        self.model_path = model_path or os.getenv("P2P_MODEL_PATH", "models/p2p_model.pkl")
        self._try_load_model()
    
    def _try_load_model(self):
        """Attempt to load the ML model, fail gracefully to fallback."""
        try:
            import joblib
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.model_loaded = True
                logger.info(f"P2P model loaded from {self.model_path}")
            else:
                logger.warning(f"P2P model not found at {self.model_path}, using rule-based fallback")
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
        """Use ML model for prediction."""
        try:
            # Prepare features (adjust based on actual model features)
            features = [
                debt_amount,
                days_overdue,
                1 if has_dispute else 0,
                previous_payments,
                {"retail": 0, "commercial": 1, "international": 2}.get(segment, 0)
            ]
            
            score = float(self.model.predict_proba([features])[0][1])
            
            return {
                "score": max(0.0, min(1.0, score)),
                "method": "ml",
                "factors": {
                    "model": "xgboost",
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
