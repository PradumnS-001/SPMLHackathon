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
    auth_router,
    chatbot_router
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

    # Auto-seed database if empty
    try:
        from app.database import SessionLocal
        from app import models
        from seed_data import seed_agencies, seed_cases, seed_violations, seed_users
        
        db = SessionLocal()
        if db.query(models.Agency).count() == 0:
            logger.info("🌱 Database is empty. Auto-seeding initial demo data...")
            agencies = seed_agencies(db)
            cases = seed_cases(db, agencies)
            violations = seed_violations(db, cases, agencies)
            users = seed_users(db, agencies)
            logger.info("✨ Auto-seeding completed successfully!")
        db.close()
    except Exception as e:
        logger.error(f"Error during auto-seeding: {e}")

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
import os

cors_origins_env = os.getenv("CORS_ORIGINS", "")
allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://dca-frontend.onrender.com",
]

if cors_origins_env and cors_origins_env != "*":
    for origin in cors_origins_env.split(","):
        o = origin.strip()
        if o and o not in allowed_origins:
            allowed_origins.append(o)

# If CORS_ORIGINS is explicitly set to '*', allow all origins (no credentials).
# When using wildcard origins, browsers disallow Access-Control-Allow-Credentials,
# so we disable credentials in that case. For stricter setups provide a comma
# separated list of allowed origins via the CORS_ORIGINS env var.
use_wildcard = cors_origins_env == "*"
if use_wildcard:
    cors_allow_origins = ["*"]
    cors_allow_credentials = False
else:
    cors_allow_origins = allowed_origins
    cors_allow_credentials = True

logger.info(f"CORS configuration: allow_origins={cors_allow_origins}, allow_credentials={cors_allow_credentials}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(cases_router, prefix="/api/v1")
app.include_router(agencies_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")
app.include_router(compliance_router, prefix="/api/v1")
app.include_router(chatbot_router, prefix="/api/v1")


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
