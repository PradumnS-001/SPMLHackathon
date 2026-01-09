"""
Routers package initialization.
"""
from .cases import router as cases_router
from .agencies import router as agencies_router
from .analytics import router as analytics_router
from .compliance import router as compliance_router
from .auth import router as auth_router

__all__ = [
    "cases_router",
    "agencies_router", 
    "analytics_router",
    "compliance_router",
    "auth_router"
]
