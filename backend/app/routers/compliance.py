"""
Compliance API Router.
Handles violation tracking and transcript checking.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from ..database import get_db
from .. import models, schemas
from ..services import get_compliance_checker

router = APIRouter(prefix="/compliance", tags=["Compliance"])


class TranscriptCheckRequest(BaseModel):
    transcript: str
    case_id: Optional[int] = None
    agency_id: Optional[int] = None
    contact_time: Optional[datetime] = None


class TranscriptCheckResponse(BaseModel):
    compliant: bool
    violations: List[dict]
    violation_count: int
    severity: str
    method: str
    recommendations: List[str]
    violation_id: Optional[int] = None


@router.get("/violations", response_model=List[schemas.ViolationResponse])
async def list_violations(
    agency_id: Optional[int] = None,
    case_id: Optional[int] = None,
    severity: Optional[str] = None,
    is_resolved: Optional[bool] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List compliance violations with filters."""
    query = db.query(models.Violation)
    
    if agency_id:
        query = query.filter(models.Violation.agency_id == agency_id)
    if case_id:
        query = query.filter(models.Violation.case_id == case_id)
    if severity:
        query = query.filter(models.Violation.severity == severity)
    if is_resolved is not None:
        query = query.filter(models.Violation.is_resolved == is_resolved)
    
    violations = query.order_by(models.Violation.detected_at.desc()).offset(skip).limit(limit).all()
    
    return violations


@router.get("/violations/{violation_id}", response_model=schemas.ViolationResponse)
async def get_violation(violation_id: int, db: Session = Depends(get_db)):
    """Get a single violation."""
    violation = db.query(models.Violation).filter(models.Violation.id == violation_id).first()
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    return violation


@router.post("/check-transcript", response_model=TranscriptCheckResponse)
async def check_transcript(
    request: TranscriptCheckRequest,
    db: Session = Depends(get_db)
):
    """Check a transcript for compliance violations."""
    checker = get_compliance_checker()
    
    result = checker.check_transcript(
        transcript=request.transcript,
        contact_time=request.contact_time
    )
    
    violation_id = None
    
    # If violations found and case/agency provided, log to database
    if result["violations"] and request.case_id and request.agency_id:
        # Verify case and agency exist
        case = db.query(models.Case).filter(models.Case.id == request.case_id).first()
        agency = db.query(models.Agency).filter(models.Agency.id == request.agency_id).first()
        
        if case and agency:
            # Create violation record
            violation = models.Violation(
                case_id=request.case_id,
                agency_id=request.agency_id,
                violation_type=result["violations"][0]["type"],  # Primary violation
                severity=result["severity"],
                description="; ".join([str(v) for v in result["violations"]]),
                transcript_excerpt=request.transcript[:500] if len(request.transcript) > 500 else request.transcript,
                detection_method=result["method"]
            )
            db.add(violation)
            
            # Update agency compliance score
            agency.compliance_score = max(0.0, agency.compliance_score - 0.02)  # Small penalty
            
            db.commit()
            db.refresh(violation)
            
            violation_id = violation.id
    
    return TranscriptCheckResponse(
        compliant=result["compliant"],
        violations=result["violations"],
        violation_count=result["violation_count"],
        severity=result["severity"],
        method=result["method"],
        recommendations=result["recommendations"],
        violation_id=violation_id
    )


@router.post("/violations/{violation_id}/resolve")
async def resolve_violation(
    violation_id: int,
    notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Mark a violation as resolved."""
    violation = db.query(models.Violation).filter(models.Violation.id == violation_id).first()
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    
    violation.is_resolved = True
    violation.resolved_at = datetime.utcnow()
    violation.resolution_notes = notes
    
    # Restore some compliance score
    agency = db.query(models.Agency).filter(models.Agency.id == violation.agency_id).first()
    if agency:
        agency.compliance_score = min(1.0, agency.compliance_score + 0.01)
    
    db.commit()
    
    return {"message": "Violation resolved", "violation_id": violation_id}


@router.get("/stats")
async def get_compliance_stats(db: Session = Depends(get_db)):
    """Get compliance statistics."""
    total_violations = db.query(func.count(models.Violation.id)).scalar() or 0
    unresolved = db.query(func.count(models.Violation.id)).filter(
        models.Violation.is_resolved == False
    ).scalar() or 0
    
    # Violations by type
    by_type = db.query(
        models.Violation.violation_type,
        func.count(models.Violation.id).label("count")
    ).group_by(models.Violation.violation_type).all()
    
    # Violations by severity
    by_severity = db.query(
        models.Violation.severity,
        func.count(models.Violation.id).label("count")
    ).group_by(models.Violation.severity).all()
    
    # Top offending agencies
    by_agency = db.query(
        models.Agency.name,
        func.count(models.Violation.id).label("count")
    ).join(models.Violation).group_by(models.Agency.id).order_by(
        func.count(models.Violation.id).desc()
    ).limit(5).all()
    
    return {
        "total_violations": total_violations,
        "unresolved_violations": unresolved,
        "by_type": [{"type": t, "count": c} for t, c in by_type],
        "by_severity": [{"severity": s, "count": c} for s, c in by_severity],
        "top_offenders": [{"agency": name, "violations": count} for name, count in by_agency]
    }
