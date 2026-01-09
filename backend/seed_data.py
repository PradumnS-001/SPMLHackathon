"""
Seed Data Script
Populates the database with realistic sample data for demo.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, Base, engine
from app import models
from app.auth import get_password_hash
from app.services import get_p2p_scorer
import random
from datetime import datetime, timedelta


def seed_agencies(db):
    """Create sample agencies."""
    agencies_data = [
        {
            "name": "RecoverMax Solutions",
            "category": "retail",
            "performance_score": 0.85,
            "compliance_score": 0.92,
            "max_capacity": 150,
            "contact_email": "ops@recovermax.com",
            "contact_phone": "+1-555-0101"
        },
        {
            "name": "Commercial Collections Inc",
            "category": "commercial",
            "performance_score": 0.78,
            "compliance_score": 0.95,
            "max_capacity": 100,
            "contact_email": "team@commercialcollect.com",
            "contact_phone": "+1-555-0102"
        },
        {
            "name": "Global Recovery Partners",
            "category": "international",
            "performance_score": 0.72,
            "compliance_score": 0.88,
            "max_capacity": 80,
            "contact_email": "global@grprecovery.com",
            "contact_phone": "+1-555-0103"
        },
        {
            "name": "Swift Debt Solutions",
            "category": "retail",
            "performance_score": 0.80,
            "compliance_score": 0.90,
            "max_capacity": 120,
            "contact_email": "contact@swiftdebt.com",
            "contact_phone": "+1-555-0104"
        },
        {
            "name": "Enterprise Collection Agency",
            "category": "commercial",
            "performance_score": 0.82,
            "compliance_score": 0.87,
            "max_capacity": 100,
            "contact_email": "info@enterpriseca.com",
            "contact_phone": "+1-555-0105"
        }
    ]
    
    agencies = []
    for data in agencies_data:
        agency = models.Agency(**data)
        db.add(agency)
        agencies.append(agency)
    
    db.commit()
    print(f"✅ Created {len(agencies)} agencies")
    return agencies


def seed_cases(db, agencies):
    """Create sample cases."""
    scorer = get_p2p_scorer()
    
    # Customer names for realistic data
    first_names = ["John", "Sarah", "Mike", "Emily", "David", "Lisa", "James", "Jennifer", "Robert", "Michelle"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    companies = ["ABC Corp", "XYZ Industries", "Global Tech", "Metro Solutions", "Prime Logistics", "Fast Freight", "Ocean Shipping", "Air Express", "Rail Transport", "Highway Haulers"]
    
    segments = ["retail", "commercial", "international"]
    statuses = ["unassigned", "assigned", "in_progress", "payment_pending", "resolved"]
    
    cases = []
    
    for i in range(100):
        segment = random.choice(segments)
        is_company = segment == "commercial" or random.random() > 0.7
        
        if is_company:
            customer_name = random.choice(companies)
        else:
            customer_name = f"{random.choice(first_names)} {random.choice(last_names)}"
        
        days_overdue = random.randint(5, 180)
        debt_amount = round(random.uniform(100, 25000), 2)
        has_dispute = random.random() < 0.15
        previous_payments = random.randint(0, 3)
        
        # Calculate P2P score
        score_result = scorer.calculate_score(
            debt_amount=debt_amount,
            days_overdue=days_overdue,
            has_dispute=has_dispute,
            previous_payments=previous_payments,
            segment=segment
        )
        
        priority_score = scorer.get_priority_score(
            p2p_score=score_result["score"],
            debt_amount=debt_amount,
            days_overdue=days_overdue
        )
        
        # Determine status (weighted towards unassigned for demo)
        status_weights = [0.4, 0.25, 0.15, 0.1, 0.1]
        status = random.choices(statuses, weights=status_weights)[0]
        
        case = models.Case(
            invoice_id=f"INV-{2024000 + i}",
            customer_id=f"CUST-{10000 + i}",
            customer_name=customer_name,
            customer_email=f"{customer_name.lower().replace(' ', '.')}@email.com",
            customer_phone=f"+1-555-{random.randint(1000, 9999)}",
            debt_amount=debt_amount,
            original_amount=debt_amount * random.uniform(1.0, 1.2),
            days_overdue=days_overdue,
            segment=segment,
            p2p_score=score_result["score"],
            priority_score=priority_score,
            status=status,
            has_dispute=has_dispute,
            previous_payments=previous_payments,
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 60))
        )
        
        # Assign some cases
        if status != "unassigned":
            matched_agencies = [a for a in agencies if a.category == segment]
            if not matched_agencies:
                matched_agencies = agencies
            
            agency = random.choice(matched_agencies)
            case.agency_id = agency.id
            case.assigned_at = case.created_at + timedelta(days=random.randint(1, 5))
            agency.current_load += 1
            
            if status == "resolved":
                case.resolved_at = case.assigned_at + timedelta(days=random.randint(5, 45))
                case.amount_recovered = case.debt_amount * random.uniform(0.5, 1.0)
        
        cases.append(case)
        db.add(case)
    
    db.commit()
    print(f"✅ Created {len(cases)} cases")
    return cases


def seed_violations(db, cases, agencies):
    """Create sample compliance violations."""
    violation_types = [
        ("aggressive_language", "high", "Agent used threatening language during call"),
        ("missing_disclosure", "medium", "Mini-Miranda warning not provided"),
        ("contact_time_violation", "high", "Call made outside permitted hours"),
        ("unauthorized_fees", "critical", "Mentioned non-existent fees")
    ]
    
    violations = []
    
    # Add violations to ~10% of assigned cases
    assigned_cases = [c for c in cases if c.agency_id is not None]
    
    for case in random.sample(assigned_cases, min(10, len(assigned_cases))):
        vtype, severity, description = random.choice(violation_types)
        
        violation = models.Violation(
            case_id=case.id,
            agency_id=case.agency_id,
            violation_type=vtype,
            severity=severity,
            description=description,
            transcript_excerpt="[Sample transcript excerpt for demo...]",
            detection_method="rule_based",
            is_resolved=random.random() < 0.3
        )
        
        if violation.is_resolved:
            violation.resolved_at = datetime.utcnow() - timedelta(days=random.randint(1, 10))
            violation.resolution_notes = "Reviewed and addressed with agent training"
        
        violations.append(violation)
        db.add(violation)
    
    db.commit()
    print(f"✅ Created {len(violations)} violations")
    return violations


def seed_users(db, agencies):
    """Create sample users."""
    users_data = [
        {
            "email": "admin@fedex.com",
            "full_name": "System Administrator",
            "role": "admin",
            "password": "admin123"
        },
        {
            "email": "viewer@fedex.com",
            "full_name": "Dashboard Viewer",
            "role": "viewer",
            "password": "viewer123"
        }
    ]
    
    # Add agency users
    for i, agency in enumerate(agencies):
        users_data.append({
            "email": f"agent{i+1}@{agency.name.lower().replace(' ', '')}.com",
            "full_name": f"Agency User {i+1}",
            "role": "agency",
            "password": "agent123",
            "agency_id": agency.id
        })
    
    users = []
    for data in users_data:
        password = data.pop("password")
        user = models.User(
            **data,
            hashed_password=get_password_hash(password)
        )
        db.add(user)
        users.append(user)
    
    db.commit()
    print(f"✅ Created {len(users)} users")
    return users


def main():
    """Main seed function."""
    print("\n🌱 Seeding DCA Management Database...\n")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    try:
        # Check if already seeded
        existing_agencies = db.query(models.Agency).count()
        if existing_agencies > 0:
            print("⚠️  Database already has data. Skipping seed.")
            print("   To reseed, delete the dca_management.db file and run again.\n")
            return
        
        agencies = seed_agencies(db)
        cases = seed_cases(db, agencies)
        violations = seed_violations(db, cases, agencies)
        users = seed_users(db, agencies)
        
        print("\n✨ Seed complete!")
        print("\n📊 Summary:")
        print(f"   - {len(agencies)} agencies")
        print(f"   - {len(cases)} cases")
        print(f"   - {len(violations)} violations")
        print(f"   - {len(users)} users")
        print("\n📝 Default Credentials:")
        print("   Admin: admin@fedex.com / admin123")
        print("   Viewer: viewer@fedex.com / viewer123")
        print("   Agency: agent1@recovermaxsolutions.com / agent123")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
