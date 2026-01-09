"""
Agencies API Router.
Handles agency CRUD and performance tracking.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional

from ..database import get_db
from .. import models, schemas

router = APIRouter(prefix="/agencies", tags=["Agencies"])


@router.get("/", response_model=List[schemas.AgencyWithStats])
async def list_agencies(
    category: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List all agencies with performance stats."""
    query = db.query(models.Agency)
    
    if category:
        query = query.filter(models.Agency.category == category)
    
    agencies = query.offset(skip).limit(limit).all()
    
    # Enrich with stats
    result = []
    for agency in agencies:
        # Calculate stats
        total_cases = db.query(func.count(models.Case.id)).filter(
            models.Case.agency_id == agency.id
        ).scalar() or 0
        
        resolved_cases = db.query(func.count(models.Case.id)).filter(
            models.Case.agency_id == agency.id,
            models.Case.status == "resolved"
        ).scalar() or 0
        
        total_debt = db.query(func.sum(models.Case.debt_amount)).filter(
            models.Case.agency_id == agency.id
        ).scalar() or 0
        
        total_recovered = db.query(func.sum(models.Case.amount_recovered)).filter(
            models.Case.agency_id == agency.id,
            models.Case.status == "resolved"
        ).scalar() or 0
        
        recovery_rate = (total_recovered / total_debt * 100) if total_debt > 0 else 0
        
        # Calculate fit score
        fit_score = (agency.performance_score * 0.7) + (agency.compliance_score * 0.3)
        
        result.append(schemas.AgencyWithStats(
            id=agency.id,
            name=agency.name,
            category=agency.category,
            contact_email=agency.contact_email,
            contact_phone=agency.contact_phone,
            max_capacity=agency.max_capacity,
            performance_score=agency.performance_score,
            compliance_score=agency.compliance_score,
            current_load=agency.current_load,
            created_at=agency.created_at,
            fit_score=round(fit_score, 3),
            total_cases=total_cases,
            resolved_cases=resolved_cases,
            recovery_rate=round(recovery_rate, 2)
        ))
    
    # Sort by fit score
    result.sort(key=lambda x: x.fit_score or 0, reverse=True)
    
    return result


@router.get("/{agency_id}", response_model=schemas.AgencyWithStats)
async def get_agency(agency_id: int, db: Session = Depends(get_db)):
    """Get a single agency with stats."""
    agency = db.query(models.Agency).filter(models.Agency.id == agency_id).first()
    if not agency:
        raise HTTPException(status_code=404, detail="Agency not found")
    
    # Fetch stats
    total_cases = db.query(func.count(models.Case.id)).filter(
        models.Case.agency_id == agency.id
    ).scalar() or 0
    
    resolved_cases = db.query(func.count(models.Case.id)).filter(
        models.Case.agency_id == agency.id,
        models.Case.status == "resolved"
    ).scalar() or 0
    
    total_debt = db.query(func.sum(models.Case.debt_amount)).filter(
        models.Case.agency_id == agency.id
    ).scalar() or 0
    
    total_recovered = db.query(func.sum(models.Case.amount_recovered)).filter(
        models.Case.agency_id == agency.id,
        models.Case.status == "resolved"
    ).scalar() or 0
    
    recovery_rate = (total_recovered / total_debt * 100) if total_debt > 0 else 0
    fit_score = (agency.performance_score * 0.7) + (agency.compliance_score * 0.3)
    
    return schemas.AgencyWithStats(
        id=agency.id,
        name=agency.name,
        category=agency.category,
        contact_email=agency.contact_email,
        contact_phone=agency.contact_phone,
        max_capacity=agency.max_capacity,
        performance_score=agency.performance_score,
        compliance_score=agency.compliance_score,
        current_load=agency.current_load,
        created_at=agency.created_at,
        fit_score=round(fit_score, 3),
        total_cases=total_cases,
        resolved_cases=resolved_cases,
        recovery_rate=round(recovery_rate, 2)
    )


@router.post("/", response_model=schemas.AgencyResponse)
async def create_agency(agency: schemas.AgencyCreate, db: Session = Depends(get_db)):
    """Create a new agency."""
    existing = db.query(models.Agency).filter(models.Agency.name == agency.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Agency name already exists")
    
    db_agency = models.Agency(**agency.model_dump())
    db.add(db_agency)
    db.commit()
    db.refresh(db_agency)
    
    return db_agency


@router.patch("/{agency_id}", response_model=schemas.AgencyResponse)
async def update_agency(
    agency_id: int, 
    update: schemas.AgencyUpdate, 
    db: Session = Depends(get_db)
):
    """Update an agency."""
    agency = db.query(models.Agency).filter(models.Agency.id == agency_id).first()
    if not agency:
        raise HTTPException(status_code=404, detail="Agency not found")
    
    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(agency, field, value)
    
    db.commit()
    db.refresh(agency)
    
    return agency


@router.delete("/{agency_id}")
async def delete_agency(agency_id: int, db: Session = Depends(get_db)):
    """Delete an agency (only if no active cases)."""
    agency = db.query(models.Agency).filter(models.Agency.id == agency_id).first()
    if not agency:
        raise HTTPException(status_code=404, detail="Agency not found")
    
    # Check for active cases
    active_cases = db.query(models.Case).filter(
        models.Case.agency_id == agency_id,
        models.Case.status.not_in(["resolved", "unassigned"])
    ).count()
    
    if active_cases > 0:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot delete agency with {active_cases} active cases"
        )
    
    db.delete(agency)
    db.commit()
    
    return {"message": "Agency deleted successfully"}


@router.get("/{agency_id}/cases", response_model=List[schemas.CaseResponse])
async def get_agency_cases(
    agency_id: int,
    status: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get cases assigned to a specific agency."""
    agency = db.query(models.Agency).filter(models.Agency.id == agency_id).first()
    if not agency:
        raise HTTPException(status_code=404, detail="Agency not found")
    
    query = db.query(models.Case).filter(models.Case.agency_id == agency_id)
    
    if status:
        query = query.filter(models.Case.status == status)
    
    cases = query.order_by(models.Case.priority_score.desc()).offset(skip).limit(limit).all()
    
    return cases
