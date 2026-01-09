"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ============== Enums ==============

class CaseStatus(str, Enum):
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PAYMENT_PENDING = "payment_pending"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class DebtSegment(str, Enum):
    RETAIL = "retail"
    COMMERCIAL = "commercial"
    INTERNATIONAL = "international"


class ViolationType(str, Enum):
    AGGRESSIVE_LANGUAGE = "aggressive_language"
    MISSING_DISCLOSURE = "missing_disclosure"
    CONTACT_TIME_VIOLATION = "contact_time_violation"
    UNAUTHORIZED_FEES = "unauthorized_fees"


# ============== Agency Schemas ==============

class AgencyBase(BaseModel):
    name: str
    category: str = "retail"
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    max_capacity: int = 100


class AgencyCreate(AgencyBase):
    pass


class AgencyUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    performance_score: Optional[float] = None
    compliance_score: Optional[float] = None
    max_capacity: Optional[int] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None


class AgencyResponse(AgencyBase):
    id: int
    performance_score: float
    compliance_score: float
    current_load: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class AgencyWithStats(AgencyResponse):
    fit_score: Optional[float] = None
    total_cases: int = 0
    resolved_cases: int = 0
    recovery_rate: float = 0.0


# ============== Case Schemas ==============

class CaseBase(BaseModel):
    invoice_id: str
    customer_id: str
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    debt_amount: float
    original_amount: Optional[float] = None
    days_overdue: int = 0
    segment: str = "retail"
    has_dispute: bool = False


class CaseCreate(CaseBase):
    pass


class CaseBulkCreate(BaseModel):
    cases: List[CaseCreate]


class CaseUpdate(BaseModel):
    status: Optional[str] = None
    days_overdue: Optional[int] = None
    debt_amount: Optional[float] = None
    has_dispute: Optional[bool] = None
    is_escalated: Optional[bool] = None
    resolution_notes: Optional[str] = None
    amount_recovered: Optional[float] = None


class CaseResponse(CaseBase):
    id: int
    status: str
    p2p_score: float
    priority_score: float
    agency_id: Optional[int] = None
    assigned_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    amount_recovered: float
    is_escalated: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class CaseWithAgency(CaseResponse):
    agency: Optional[AgencyResponse] = None


# ============== Violation Schemas ==============

class ViolationBase(BaseModel):
    case_id: int
    agency_id: int
    violation_type: str
    severity: str = "medium"
    description: Optional[str] = None
    transcript_excerpt: Optional[str] = None


class ViolationCreate(ViolationBase):
    pass


class ViolationResponse(ViolationBase):
    id: int
    detected_at: datetime
    detection_method: str
    is_resolved: bool
    resolved_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# ============== Interaction Schemas ==============

class InteractionBase(BaseModel):
    case_id: int
    interaction_type: str  # call, email, sms, letter
    direction: str = "outbound"
    notes: Optional[str] = None
    outcome: Optional[str] = None


class InteractionCreate(InteractionBase):
    pass


class InteractionResponse(InteractionBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============== Auth Schemas ==============

class UserBase(BaseModel):
    email: str
    full_name: Optional[str] = None
    role: str = "agency"


class UserCreate(UserBase):
    password: str
    agency_id: Optional[int] = None


class UserResponse(UserBase):
    id: int
    is_active: bool
    agency_id: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


# ============== Analytics Schemas ==============

class DashboardStats(BaseModel):
    total_cases: int
    unassigned_cases: int
    assigned_cases: int
    resolved_cases: int
    total_debt: float
    total_recovered: float
    recovery_rate: float
    avg_days_overdue: float
    sla_compliance: float


class AgencyPerformance(BaseModel):
    agency_id: int
    agency_name: str
    total_cases: int
    resolved_cases: int
    recovery_rate: float
    avg_resolution_days: float
    compliance_score: float
    performance_score: float


class CasesByStatus(BaseModel):
    status: str
    count: int
    percentage: float


class RecoveryTrend(BaseModel):
    date: str
    recovered_amount: float
    case_count: int


# ============== Assignment Schemas ==============

class AssignmentRequest(BaseModel):
    case_ids: Optional[List[int]] = None  # If None, assign all unassigned


class AssignmentResult(BaseModel):
    case_id: int
    invoice_id: str
    agency_id: int
    agency_name: str
    fit_score: float
    method: str  # "ai" or "fallback"


class BulkAssignmentResponse(BaseModel):
    total_assigned: int
    assignments: List[AssignmentResult]
    errors: List[str] = []
