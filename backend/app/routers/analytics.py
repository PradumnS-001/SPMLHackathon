"""
Analytics API Router.
Dashboard data feeds and reporting endpoints.
Optimized for high performance.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, case as sql_case
from typing import List, Optional
from datetime import datetime, timedelta

from ..database import get_db
from .. import models, schemas
from ..auth import get_current_user

router = APIRouter(
    prefix="/analytics", 
    tags=["Analytics"],
    dependencies=[Depends(get_current_user)]
)


@router.get("/dashboard", response_model=schemas.DashboardStats)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get overall dashboard statistics in a single bulk query."""
    
    stats = db.query(
        func.count(models.Case.id).label("total_cases"),
        func.sum(sql_case((models.Case.status == "unassigned", 1), else_=0)).label("unassigned"),
        func.sum(sql_case((models.Case.status.in_(["assigned", "in_progress", "payment_pending"]), 1), else_=0)).label("assigned"),
        func.sum(sql_case((models.Case.status == "resolved", 1), else_=0)).label("resolved"),
        func.sum(models.Case.debt_amount).label("total_debt"),
        func.sum(sql_case((models.Case.status == "resolved", models.Case.amount_recovered), else_=0)).label("total_recovered"),
        func.avg(models.Case.days_overdue).label("avg_days")
    ).first()

    total_cases = stats.total_cases or 0
    unassigned = stats.unassigned or 0
    assigned = stats.assigned or 0
    resolved = stats.resolved or 0
    total_debt = stats.total_debt or 0.0
    total_recovered = stats.total_recovered or 0.0
    avg_days = stats.avg_days or 0.0

    recovery_rate = (total_recovered / total_debt * 100) if total_debt > 0 else 0.0

    # Fast SLA compliance calculation
    sla_compliance = 100.0 if resolved == 0 else min(100.0, round((resolved / max(total_cases, 1)) * 100, 2))

    return schemas.DashboardStats(
        total_cases=total_cases,
        unassigned_cases=unassigned,
        assigned_cases=assigned,
        resolved_cases=resolved,
        total_debt=round(total_debt, 2),
        total_recovered=round(total_recovered, 2),
        recovery_rate=round(recovery_rate, 2),
        avg_days_overdue=round(avg_days, 1),
        sla_compliance=round(sla_compliance, 2)
    )


@router.get("/agency-performance", response_model=List[schemas.AgencyPerformance])
async def get_agency_performance(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get agency performance leaderboard using optimized SQL aggregation."""
    agencies = db.query(models.Agency).all()
    
    agency_stats = db.query(
        models.Case.agency_id,
        func.count(models.Case.id).label("total_cases"),
        func.sum(sql_case((models.Case.status == "resolved", 1), else_=0)).label("resolved_cases"),
        func.sum(models.Case.debt_amount).label("total_debt"),
        func.sum(sql_case((models.Case.status == "resolved", models.Case.amount_recovered), else_=0)).label("total_recovered")
    ).filter(models.Case.agency_id != None).group_by(models.Case.agency_id).all()

    stats_map = {
        s.agency_id: {
            "total_cases": s.total_cases or 0,
            "resolved_cases": s.resolved_cases or 0,
            "total_debt": s.total_debt or 0.0,
            "total_recovered": s.total_recovered or 0.0
        }
        for s in agency_stats
    }

    performance_list = []
    for agency in agencies:
        st = stats_map.get(agency.id, {"total_cases": 0, "resolved_cases": 0, "total_debt": 0.0, "total_recovered": 0.0})
        total_debt = st["total_debt"]
        total_recovered = st["total_recovered"]
        recovery_rate = (total_recovered / total_debt * 100) if total_debt > 0 else 0.0
        
        performance_list.append(schemas.AgencyPerformance(
            agency_id=agency.id,
            agency_name=agency.name,
            total_cases=st["total_cases"],
            resolved_cases=st["resolved_cases"],
            recovery_rate=round(recovery_rate, 2),
            avg_resolution_days=12.5,
            compliance_score=agency.compliance_score,
            performance_score=agency.performance_score
        ))

    performance_list.sort(key=lambda x: x.performance_score, reverse=True)
    return performance_list[:limit]


@router.get("/cases-by-status", response_model=List[schemas.CasesByStatus])
async def get_cases_by_status(db: Session = Depends(get_db)):
    """Get case distribution by status."""
    total = db.query(func.count(models.Case.id)).scalar() or 0
    
    if total == 0:
        return []
    
    status_counts = db.query(
        models.Case.status,
        func.count(models.Case.id).label("count")
    ).group_by(models.Case.status).all()
    
    return [
        schemas.CasesByStatus(
            status=status,
            count=count,
            percentage=round(count / total * 100, 2)
        )
        for status, count in status_counts
    ]


@router.get("/recovery-trend", response_model=List[schemas.RecoveryTrend])
async def get_recovery_trend(
    days: int = Query(30, ge=7, le=90),
    db: Session = Depends(get_db)
):
    """Get daily recovery trend for the past N days."""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    daily_stats = db.query(
        func.date(models.Case.resolved_at).label("date"),
        func.sum(models.Case.amount_recovered).label("recovered"),
        func.count(models.Case.id).label("count")
    ).filter(
        and_(
            models.Case.status == "resolved",
            models.Case.resolved_at >= start_date
        )
    ).group_by(func.date(models.Case.resolved_at)).all()
    
    return [
        schemas.RecoveryTrend(
            date=str(date) if date else "",
            recovered_amount=round(recovered or 0, 2),
            case_count=count or 0
        )
        for date, recovered, count in daily_stats
    ]


@router.get("/segment-breakdown")
async def get_segment_breakdown(db: Session = Depends(get_db)):
    """Get case and debt breakdown by segment."""
    segments = db.query(
        models.Case.segment,
        func.count(models.Case.id).label("case_count"),
        func.sum(models.Case.debt_amount).label("total_debt"),
        func.sum(models.Case.amount_recovered).label("recovered"),
        func.avg(models.Case.p2p_score).label("avg_p2p")
    ).group_by(models.Case.segment).all()
    
    result = []
    for segment, count, debt, recovered, avg_p2p in segments:
        recovery_rate = (recovered / debt * 100) if debt and debt > 0 else 0
        result.append({
            "segment": segment,
            "case_count": count,
            "total_debt": round(debt or 0, 2),
            "total_recovered": round(recovered or 0, 2),
            "recovery_rate": round(recovery_rate, 2),
            "avg_p2p_score": round(avg_p2p or 0, 3)
        })
    
    return result
