"""
Compliance Checker Service.
NLP-based compliance monitoring with keyword fallback.
"""
from typing import List, Dict, Optional
import re
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# COMPLIANCE RULES CONFIGURATION
# =============================================================================

AGGRESSIVE_KEYWORDS = [
    # Threats
    "threaten", "threat", "sue", "lawsuit", "court", "arrest", "jail", "prison",
    "garnish", "garnishment", "seize", "seizure", "lien",
    
    # Insults
    "stupid", "idiot", "moron", "liar", "fraud", "criminal", "deadbeat",
    "loser", "worthless", "pathetic",
    
    # Harassment indicators
    "harassment", "harass", "stalk", "constant", "non-stop", "every day",
    
    # False claims
    "guarantee", "promised", "definitely", "100%", "absolutely will"
]

REQUIRED_DISCLOSURES = [
    {
        "pattern": r"(this is.*(an attempt|attempting) to collect|collect(ing)? a debt)",
        "name": "Debt Collection Notice",
        "required": True
    },
    {
        "pattern": r"(mini.?miranda|debt collector)",
        "name": "Mini-Miranda Warning",
        "required": True
    },
    {
        "pattern": r"(information.*(will be|may be) used|purposes of collecting)",
        "name": "Information Usage Disclosure",
        "required": False  # Recommended but not always required
    }
]

# FDCPA contact time rules (8 AM - 9 PM local time)
ALLOWED_CONTACT_START = time(8, 0)
ALLOWED_CONTACT_END = time(21, 0)


class ComplianceChecker:
    """
    Compliance monitoring with NLP model and rule-based fallback.
    """
    
    def __init__(self, use_ml: bool = False):
        self.use_ml = use_ml
        self.ml_model = None
        
        if use_ml:
            self._try_load_ml_model()
    
    def _try_load_ml_model(self):
        """Attempt to load NLP sentiment model."""
        try:
            # Placeholder for actual ML model loading
            # from transformers import pipeline
            # self.ml_model = pipeline("sentiment-analysis")
            logger.warning("ML compliance model not implemented, using rule-based")
            self.use_ml = False
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}, using rule-based")
            self.use_ml = False
    
    def check_transcript(self, transcript: str, contact_time: Optional[datetime] = None) -> Dict:
        """
        Check a call/email transcript for compliance violations.
        
        Args:
            transcript: The text content to analyze
            contact_time: When the contact was made (for time-of-day checks)
            
        Returns:
            Dict with violations list, severity, and recommendations
        """
        violations = []
        
        # Check for aggressive language
        aggressive_violations = self._check_aggressive_language(transcript)
        violations.extend(aggressive_violations)
        
        # Check for required disclosures
        disclosure_violations = self._check_disclosures(transcript)
        violations.extend(disclosure_violations)
        
        # Check contact time if provided
        if contact_time:
            time_violations = self._check_contact_time(contact_time)
            violations.extend(time_violations)
        
        # Calculate overall severity
        severity = self._calculate_severity(violations)
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "violation_count": len(violations),
            "severity": severity,
            "method": "ml" if self.use_ml else "rule_based",
            "recommendations": self._get_recommendations(violations)
        }
    
    def _check_aggressive_language(self, transcript: str) -> List[Dict]:
        """Check for aggressive language patterns."""
        violations = []
        text_lower = transcript.lower()
        
        for keyword in AGGRESSIVE_KEYWORDS:
            if keyword in text_lower:
                # Find context around the keyword
                idx = text_lower.find(keyword)
                start = max(0, idx - 30)
                end = min(len(transcript), idx + len(keyword) + 30)
                excerpt = transcript[start:end]
                
                violations.append({
                    "type": "aggressive_language",
                    "keyword": keyword,
                    "excerpt": f"...{excerpt}...",
                    "severity": self._get_keyword_severity(keyword)
                })
        
        return violations
    
    def _check_disclosures(self, transcript: str) -> List[Dict]:
        """Check for required disclosures."""
        violations = []
        text_lower = transcript.lower()
        
        for disclosure in REQUIRED_DISCLOSURES:
            if disclosure["required"]:
                pattern = disclosure["pattern"]
                if not re.search(pattern, text_lower, re.IGNORECASE):
                    violations.append({
                        "type": "missing_disclosure",
                        "disclosure": disclosure["name"],
                        "severity": "high" if disclosure["required"] else "low"
                    })
        
        return violations
    
    def _check_contact_time(self, contact_time: datetime) -> List[Dict]:
        """Check if contact was made during allowed hours."""
        violations = []
        
        contact_hour = contact_time.time()
        
        if contact_hour < ALLOWED_CONTACT_START or contact_hour > ALLOWED_CONTACT_END:
            violations.append({
                "type": "contact_time_violation",
                "contact_time": contact_time.strftime("%H:%M"),
                "allowed_hours": f"{ALLOWED_CONTACT_START.strftime('%H:%M')} - {ALLOWED_CONTACT_END.strftime('%H:%M')}",
                "severity": "high"
            })
        
        return violations
    
    def _get_keyword_severity(self, keyword: str) -> str:
        """Determine severity based on keyword type."""
        high_severity = ["threaten", "arrest", "jail", "prison", "sue", "lawsuit", "garnish"]
        critical_severity = ["fraud", "criminal"]
        
        if keyword in critical_severity:
            return "critical"
        elif keyword in high_severity:
            return "high"
        else:
            return "medium"
    
    def _calculate_severity(self, violations: List[Dict]) -> str:
        """Calculate overall severity from violations list."""
        if not violations:
            return "none"
        
        severities = [v.get("severity", "medium") for v in violations]
        
        if "critical" in severities:
            return "critical"
        elif "high" in severities:
            return "high"
        elif "medium" in severities:
            return "medium"
        else:
            return "low"
    
    def _get_recommendations(self, violations: List[Dict]) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []
        
        violation_types = set(v.get("type") for v in violations)
        
        if "aggressive_language" in violation_types:
            recommendations.append("Review agent training on professional communication")
            recommendations.append("Consider call monitoring for this agent")
        
        if "missing_disclosure" in violation_types:
            recommendations.append("Ensure all calls include required disclosures within first 30 seconds")
            recommendations.append("Update call scripts to include mandatory disclosures")
        
        if "contact_time_violation" in violation_types:
            recommendations.append("Review contact scheduling to ensure FDCPA compliance")
            recommendations.append("Implement time-zone aware calling restrictions")
        
        return recommendations
    
    def batch_check(self, transcripts: List[Dict]) -> List[Dict]:
        """
        Check multiple transcripts at once.
        
        Args:
            transcripts: List of dicts with 'text' and optional 'contact_time'
            
        Returns:
            List of compliance check results
        """
        results = []
        
        for item in transcripts:
            text = item.get("text", "")
            contact_time = item.get("contact_time")
            
            if isinstance(contact_time, str):
                try:
                    contact_time = datetime.fromisoformat(contact_time)
                except ValueError:
                    contact_time = None
            
            result = self.check_transcript(text, contact_time)
            result["id"] = item.get("id")
            results.append(result)
        
        return results


# Singleton instance
_checker_instance: Optional[ComplianceChecker] = None


def get_compliance_checker() -> ComplianceChecker:
    """Get or create compliance checker singleton."""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = ComplianceChecker(use_ml=False)
    return _checker_instance
