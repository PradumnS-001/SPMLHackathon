"""
Case Router Service.
AI-powered case assignment with smart round-robin fallback.
"""
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_
import logging

from ..models import Case, Agency
from .p2p_scorer import get_p2p_scorer

logger = logging.getLogger(__name__)


class CaseRouter:
    """
    Intelligent case routing with AI scoring and fallback logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.p2p_scorer = get_p2p_scorer()
    
    def calculate_fit_score(self, agency: Agency, case: Case) -> float:
        """
        Calculate agency fit score for a case.
        Formula: (Performance × 0.7) + (Compliance × 0.3)
        """
        # Base fit score
        fit_score = (agency.performance_score * 0.7) + (agency.compliance_score * 0.3)
        
        # Category match bonus
        if agency.category == case.segment:
            fit_score *= 1.1  # 10% bonus for category match
        
        # Capacity penalty (prefer agencies with more room)
        capacity_ratio = agency.current_load / max(agency.max_capacity, 1)
        if capacity_ratio > 0.9:
            fit_score *= 0.7  # Heavy penalty when nearly full
        elif capacity_ratio > 0.75:
            fit_score *= 0.85  # Moderate penalty
        
        return min(1.0, round(fit_score, 3))
    
    def get_eligible_agencies(self, case: Case) -> List[Agency]:
        """Get agencies eligible for a case based on category and capacity."""
        # First try category-matched agencies with capacity
        eligible = self.db.query(Agency).filter(
            and_(
                Agency.category == case.segment,
                Agency.current_load < Agency.max_capacity
            )
        ).all()
        
        # Fallback to any agency with capacity
        if not eligible:
            logger.info(f"No category-matched agencies for {case.segment}, using general pool")
            eligible = self.db.query(Agency).filter(
                Agency.current_load < Agency.max_capacity
            ).all()
        
        return eligible
    
    def assign_case(self, case: Case) -> Tuple[Optional[Agency], str, float]:
        """
        Assign a single case to the best-fit agency.
        
        Returns:
            Tuple of (assigned_agency, method, fit_score)
            method is 'ai' or 'fallback'
        """
        if case.status != "unassigned":
            logger.warning(f"Case {case.id} already assigned")
            return None, "skipped", 0.0
        
        # Update P2P score if not set
        if case.p2p_score == 0.5:  # Default value
            score_result = self.p2p_scorer.calculate_score(
                debt_amount=case.debt_amount,
                days_overdue=case.days_overdue,
                has_dispute=case.has_dispute,
                previous_payments=case.previous_payments,
                segment=case.segment
            )
            case.p2p_score = score_result["score"]
            case.priority_score = self.p2p_scorer.get_priority_score(
                p2p_score=score_result["score"],
                debt_amount=case.debt_amount,
                days_overdue=case.days_overdue
            )
        
        # Get eligible agencies
        eligible_agencies = self.get_eligible_agencies(case)
        
        if not eligible_agencies:
            logger.warning(f"No agencies available for case {case.id}")
            return None, "no_capacity", 0.0
        
        # Try AI-based scoring
        try:
            return self._ai_assignment(case, eligible_agencies)
        except Exception as e:
            logger.warning(f"AI assignment failed: {e}, using fallback")
            return self._fallback_assignment(case, eligible_agencies)
    
    def _ai_assignment(self, case: Case, agencies: List[Agency]) -> Tuple[Agency, str, float]:
        """AI-based assignment using fit scores."""
        # Calculate fit scores for all agencies
        scored_agencies = []
        for agency in agencies:
            fit_score = self.calculate_fit_score(agency, case)
            scored_agencies.append((agency, fit_score))
        
        # Sort by fit score (descending)
        scored_agencies.sort(key=lambda x: x[1], reverse=True)
        
        # Select best agency
        best_agency, best_score = scored_agencies[0]
        
        # Update agency load
        best_agency.current_load += 1
        
        # Update case
        case.agency_id = best_agency.id
        case.status = "assigned"
        from datetime import datetime
        case.assigned_at = datetime.utcnow()
        
        self.db.commit()
        
        logger.info(f"Case {case.id} assigned to {best_agency.name} (fit: {best_score})")
        
        return best_agency, "ai", best_score
    
    def _fallback_assignment(self, case: Case, agencies: List[Agency]) -> Tuple[Agency, str, float]:
        """
        Smart round-robin fallback assignment.
        Assigns to agency with lowest current load (category-matched first).
        """
        # Prefer category match
        category_matched = [a for a in agencies if a.category == case.segment]
        pool = category_matched if category_matched else agencies
        
        # Sort by current load (ascending)
        pool.sort(key=lambda a: a.current_load)
        
        # Select agency with lowest load
        selected_agency = pool[0]
        
        # Calculate approximate fit score for consistency
        fit_score = (selected_agency.performance_score * 0.7) + (selected_agency.compliance_score * 0.3)
        
        # Update agency load
        selected_agency.current_load += 1
        
        # Update case
        case.agency_id = selected_agency.id
        case.status = "assigned"
        from datetime import datetime
        case.assigned_at = datetime.utcnow()
        
        self.db.commit()
        
        logger.info(f"Case {case.id} assigned to {selected_agency.name} via fallback (load-balanced)")
        
        return selected_agency, "fallback", round(fit_score, 3)
    
    def bulk_assign(self, case_ids: Optional[List[int]] = None) -> List[dict]:
        """
        Assign multiple cases at once.
        
        Args:
            case_ids: Specific case IDs to assign, or None for all unassigned
            
        Returns:
            List of assignment results
        """
        results = []
        
        # Get cases to assign
        if case_ids:
            cases = self.db.query(Case).filter(
                and_(Case.id.in_(case_ids), Case.status == "unassigned")
            ).all()
        else:
            cases = self.db.query(Case).filter(Case.status == "unassigned").all()
        
        if not cases:
            logger.info("No unassigned cases to process")
            return results
        
        # Sort by priority score (high priority first)
        cases.sort(key=lambda c: c.priority_score, reverse=True)
        
        for case in cases:
            agency, method, fit_score = self.assign_case(case)
            
            if agency:
                results.append({
                    "case_id": case.id,
                    "invoice_id": case.invoice_id,
                    "agency_id": agency.id,
                    "agency_name": agency.name,
                    "fit_score": fit_score,
                    "method": method
                })
            else:
                results.append({
                    "case_id": case.id,
                    "invoice_id": case.invoice_id,
                    "error": method  # 'no_capacity' or 'skipped'
                })
        
        return results
