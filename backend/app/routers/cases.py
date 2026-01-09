"""
Cases API Router.
Handles case CRUD, ingestion, and assignment operations.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from typing import List, Optional
from datetime import datetime

from ..database import get_db
from .. import models, schemas
from ..services import get_p2p_scorer, CaseRouter
from ..auth import get_current_user

router = APIRouter(prefix="/cases", tags=["Cases"])


@router.get("/", response_model=List[schemas.CaseResponse])
async def list_cases(
    status: Optional[str] = None,
    agency_id: Optional[int] = None,
    segment: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List cases with optional filters."""
    query = db.query(models.Case)
    
    if status:
        query = query.filter(models.Case.status == status)
    if agency_id:
        query = query.filter(models.Case.agency_id == agency_id)
    if segment:
        query = query.filter(models.Case.segment == segment)
    if min_amount:
        query = query.filter(models.Case.debt_amount >= min_amount)
    if max_amount:
        query = query.filter(models.Case.debt_amount <= max_amount)
    
    cases = query.order_by(models.Case.priority_score.desc()).offset(skip).limit(limit).all()
    return cases


@router.get("/{case_id}", response_model=schemas.CaseWithAgency)
async def get_case(case_id: int, db: Session = Depends(get_db)):
    """Get a single case with agency details."""
    case = db.query(models.Case).filter(models.Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.post("/", response_model=schemas.CaseResponse)
async def create_case(case: schemas.CaseCreate, db: Session = Depends(get_db)):
    """Create a new case and calculate P2P score."""
    # Check for duplicate invoice_id
    existing = db.query(models.Case).filter(models.Case.invoice_id == case.invoice_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Invoice already exists")
    
    # Create case
    db_case = models.Case(**case.model_dump())
    
    # Calculate P2P score
    scorer = get_p2p_scorer()
    score_result = scorer.calculate_score(
        debt_amount=case.debt_amount,
        days_overdue=case.days_overdue,
        has_dispute=case.has_dispute,
        previous_payments=0,
        segment=case.segment
    )
    db_case.p2p_score = score_result["score"]
    db_case.priority_score = scorer.get_priority_score(
        p2p_score=score_result["score"],
        debt_amount=case.debt_amount,
        days_overdue=case.days_overdue
    )
    
    db.add(db_case)
    db.commit()
    db.refresh(db_case)
    
    return db_case


@router.post("/bulk", response_model=dict)
async def bulk_create_cases(bulk: schemas.CaseBulkCreate, db: Session = Depends(get_db)):
    """Bulk create cases from invoice data."""
    created = 0
    skipped = 0
    errors = []
    scorer = get_p2p_scorer()
    
    for case_data in bulk.cases:
        try:
            # Check for duplicate
            existing = db.query(models.Case).filter(
                models.Case.invoice_id == case_data.invoice_id
            ).first()
            
            if existing:
                skipped += 1
                continue
            
            db_case = models.Case(**case_data.model_dump())
            
            # Calculate scores
            score_result = scorer.calculate_score(
                debt_amount=case_data.debt_amount,
                days_overdue=case_data.days_overdue,
                has_dispute=case_data.has_dispute,
                previous_payments=0,
                segment=case_data.segment
            )
            db_case.p2p_score = score_result["score"]
            db_case.priority_score = scorer.get_priority_score(
                p2p_score=score_result["score"],
                debt_amount=case_data.debt_amount,
                days_overdue=case_data.days_overdue
            )
            
            db.add(db_case)
            created += 1
            
        except Exception as e:
            errors.append({"invoice_id": case_data.invoice_id, "error": str(e)})
    
    db.commit()
    
    return {
        "created": created,
        "skipped": skipped,
        "errors": errors
    }


@router.patch("/{case_id}", response_model=schemas.CaseResponse)
async def update_case(case_id: int, update: schemas.CaseUpdate, db: Session = Depends(get_db)):
    """Update a case."""
    case = db.query(models.Case).filter(models.Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    update_data = update.model_dump(exclude_unset=True)
    
    # Handle status changes
    if "status" in update_data:
        if update_data["status"] == "resolved" and case.status != "resolved":
            case.resolved_at = datetime.utcnow()
            
            # Update agency stats
            if case.agency_id:
                agency = db.query(models.Agency).filter(models.Agency.id == case.agency_id).first()
                if agency:
                    agency.current_load = max(0, agency.current_load - 1)
    
    for field, value in update_data.items():
        setattr(case, field, value)
    
    db.commit()
    db.refresh(case)
    
    return case


@router.post("/assign", response_model=schemas.BulkAssignmentResponse)
async def assign_cases(
    request: schemas.AssignmentRequest = schemas.AssignmentRequest(),
    db: Session = Depends(get_db)
):
    """Trigger case assignment workflow."""
    router_service = CaseRouter(db)
    results = router_service.bulk_assign(request.case_ids)
    
    assignments = []
    errors = []
    
    for result in results:
        if "error" in result:
            errors.append(f"Case {result['case_id']}: {result['error']}")
        else:
            assignments.append(schemas.AssignmentResult(**result))
    
    return schemas.BulkAssignmentResponse(
        total_assigned=len(assignments),
        assignments=assignments,
        errors=errors
    )


@router.post("/{case_id}/resolve", response_model=schemas.CaseResponse)
async def resolve_case(
    case_id: int,
    amount_recovered: float,
    notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Mark a case as resolved."""
    case = db.query(models.Case).filter(models.Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    case.status = "resolved"
    case.resolved_at = datetime.utcnow()
    case.amount_recovered = amount_recovered
    case.resolution_notes = notes
    
    # Update agency load
    if case.agency_id:
        agency = db.query(models.Agency).filter(models.Agency.id == case.agency_id).first()
        if agency:
            agency.current_load = max(0, agency.current_load - 1)
            
            # Update performance score based on recovery
            recovery_rate = amount_recovered / case.debt_amount if case.debt_amount > 0 else 0
            # Weighted average with existing score
            agency.performance_score = (agency.performance_score * 0.9) + (recovery_rate * 0.1)
    
    db.commit()
    db.refresh(case)
    
    return case
