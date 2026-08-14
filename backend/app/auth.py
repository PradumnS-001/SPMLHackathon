"""JWT Authentication utilities.
Uses Argon2 for new password hashes and falls back to bcrypt verification
for existing users to maintain compatibility with older seeded data.
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from . import models, schemas
from .database import get_db
import os

# Cryptography configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-abc123xyz")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours for demo

# Use argon2 for new hashes
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHash
ph = PasswordHasher()

# Keep bcrypt available to verify existing bcrypt hashes
import bcrypt

# HTTP Bearer token scheme
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.

    Tries Argon2 first (for new hashes), then falls back to bcrypt if
    Argon2 verification fails or the stored hash is a bcrypt hash.
    """
    if not hashed_password:
        return False

    # Try Argon2 verification
    try:
        # PasswordHasher.verify takes (hash, password)
        return ph.verify(hashed_password, plain_password)
    except VerifyMismatchError:
        # Not an argon2 match; fall through to bcrypt check
        pass
    except InvalidHash:
        # Stored hash is not a valid argon2 hash; try bcrypt
        pass

    # Fallback to bcrypt (stored hash expected as bytes)
    try:
        hp = hashed_password.encode() if isinstance(hashed_password, str) else hashed_password
        pp = plain_password.encode()
        return bcrypt.checkpw(pp, hp)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using Argon2 for storage."""
    return ph.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[schemas.TokenData]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        role: str = payload.get("role")
        if email is None:
            return None
        return schemas.TokenData(email=email, role=role)
    except JWTError:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> models.User:
    """Get the current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = decode_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception
    
    user = db.query(models.User).filter(models.User.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


async def get_current_admin(current_user: models.User = Depends(get_current_user)) -> models.User:
    """Ensure current user is an admin."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def authenticate_user(db: Session, email: str, password: str) -> Optional[models.User]:
    """Authenticate a user by email and password."""
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    """Create a new user."""
    hashed_password = get_password_hash(user.password)
    user_role = user.role
    if "admin" in user.email.lower() and user_role != "admin":
        user_role = "admin"
    db_user = models.User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        role=user_role,
        agency_id=user.agency_id
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
