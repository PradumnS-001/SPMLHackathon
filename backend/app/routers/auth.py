"""
Authentication API Router.
Handles login, registration, and user management.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from ..database import get_db
from .. import models, schemas
from ..auth import (
    authenticate_user, create_access_token, create_user,
    get_current_user, get_current_admin
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=schemas.Token)
async def login(request: schemas.LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return JWT token."""
    user = authenticate_user(db, request.email, request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role}
    )
    
    return schemas.Token(access_token=access_token)


@router.post("/register", response_model=schemas.UserResponse)
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    # Check if email exists
    existing = db.query(models.User).filter(models.User.email == user.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Verify agency exists if provided
    if user.agency_id:
        agency = db.query(models.Agency).filter(models.Agency.id == user.agency_id).first()
        if not agency:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Agency not found"
            )
    
    db_user = create_user(db, user)
    return db_user


@router.get("/me", response_model=schemas.UserResponse)
async def get_me(current_user: models.User = Depends(get_current_user)):
    """Get current user profile."""
    return current_user


@router.get("/users", response_model=list[schemas.UserResponse])
async def list_users(
    current_user: models.User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """List all users (admin only)."""
    users = db.query(models.User).all()
    return users
