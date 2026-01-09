"""
SQLAlchemy Models for DCA Management System.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
import enum


class CaseStatus(str, enum.Enum):
    """Case lifecycle status."""
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PAYMENT_PENDING = "payment_pending"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class DebtSegment(str, enum.Enum):
    """Debt category for routing."""
    RETAIL = "retail"
    COMMERCIAL = "commercial"
    INTERNATIONAL = "international"


class ViolationType(str, enum.Enum):
    """Compliance violation types."""
    AGGRESSIVE_LANGUAGE = "aggressive_language"
    MISSING_DISCLOSURE = "missing_disclosure"
    CONTACT_TIME_VIOLATION = "contact_time_violation"
    UNAUTHORIZED_FEES = "unauthorized_fees"


class Agency(Base):
    """Debt Collection Agency model."""
    __tablename__ = "agencies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    category = Column(String(50), default="retail")  # retail, commercial, international
    
    # Performance metrics (0.0 - 1.0)
    performance_score = Column(Float, default=0.7)
    compliance_score = Column(Float, default=0.9)
    
    # Capacity management
    current_load = Column(Integer, default=0)
    max_capacity = Column(Integer, default=100)
    
    # Contact info
    contact_email = Column(String(100))
    contact_phone = Column(String(20))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    cases = relationship("Case", back_populates="agency")
    violations = relationship("Violation", back_populates="agency")


class Case(Base):
    """Debt case/invoice model."""
    __tablename__ = "cases"
    
    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Customer info
    customer_id = Column(String(50), nullable=False, index=True)
    customer_name = Column(String(100))
    customer_email = Column(String(100))
    customer_phone = Column(String(20))
    
    # Debt details
    debt_amount = Column(Float, nullable=False)
    original_amount = Column(Float)
    days_overdue = Column(Integer, default=0)
    segment = Column(String(50), default="retail")
    
    # AI Scores
    p2p_score = Column(Float, default=0.5)  # Probability to Pay (0-1)
    priority_score = Column(Float, default=0.5)  # Priority for assignment (0-1)
    
    # Status tracking
    status = Column(String(20), default="unassigned")
    
    # Assignment
    agency_id = Column(Integer, ForeignKey("agencies.id"), nullable=True)
    assigned_at = Column(DateTime(timezone=True))
    
    # Resolution
    resolved_at = Column(DateTime(timezone=True))
    resolution_notes = Column(Text)
    amount_recovered = Column(Float, default=0.0)
    
    # Flags
    has_dispute = Column(Boolean, default=False)
    is_escalated = Column(Boolean, default=False)
    previous_payments = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    agency = relationship("Agency", back_populates="cases")
    violations = relationship("Violation", back_populates="case")
    interactions = relationship("Interaction", back_populates="case")


class Violation(Base):
    """Compliance violation log."""
    __tablename__ = "violations"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # References
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    agency_id = Column(Integer, ForeignKey("agencies.id"), nullable=False)
    
    # Violation details
    violation_type = Column(String(50), nullable=False)
    severity = Column(String(20), default="medium")  # low, medium, high, critical
    description = Column(Text)
    transcript_excerpt = Column(Text)
    
    # Detection info
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    detection_method = Column(String(50), default="rule_based")  # ml, rule_based, manual
    
    # Resolution
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True))
    resolution_notes = Column(Text)
    
    # Relationships
    case = relationship("Case", back_populates="violations")
    agency = relationship("Agency", back_populates="violations")


class Interaction(Base):
    """Case interaction/activity log."""
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    
    # Interaction details
    interaction_type = Column(String(20), nullable=False)  # call, email, sms, letter
    direction = Column(String(10), default="outbound")  # inbound, outbound
    notes = Column(Text)
    outcome = Column(String(50))  # promise_to_pay, no_answer, wrong_number, etc.
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="interactions")


class User(Base):
    """System user (admin or agency user)."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    
    # Role-based access
    role = Column(String(20), default="agency")  # admin, agency, viewer
    agency_id = Column(Integer, ForeignKey("agencies.id"), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
