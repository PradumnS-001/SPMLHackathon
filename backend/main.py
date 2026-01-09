"""
FedEx DCA Management System - FastAPI Backend
MVP Version with AI-powered case assignment and compliance monitoring.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.database import Base, engine, init_db
from app.routers import (
    cases_router,
    agencies_router,
    analytics_router,
    compliance_router,
    auth_router
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting DCA Management System...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")
    
    yield
    
    # Shutdown
    logger.info("Shutting down DCA Management System...")


# Create FastAPI app
app = FastAPI(
    title="FedEx DCA Management System",
    description="""
    AI-powered Debt Collection Agency management platform.
    
    ## Features
    - **Case Management**: Ingest, track, and resolve debt cases
    - **Intelligent Assignment**: AI-powered case routing with fallback
    - **Agency Management**: Track agency performance and capacity
    - **Compliance Monitoring**: NLP-based transcript analysis
    - **Analytics Dashboard**: Real-time KPIs and reporting
    """,
    version="1.0.0-mvp",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(cases_router, prefix="/api/v1")
app.include_router(agencies_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")
app.include_router(compliance_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "FedEx DCA Management System",
        "version": "1.0.0-mvp",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "auth": "/api/v1/auth",
            "cases": "/api/v1/cases",
            "agencies": "/api/v1/agencies",
            "analytics": "/api/v1/analytics",
            "compliance": "/api/v1/compliance"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
