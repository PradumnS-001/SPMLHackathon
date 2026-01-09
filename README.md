# FedEx DCA Management System - MVP

AI-powered Debt Collection Agency management platform with intelligent case assignment, compliance monitoring, and real-time analytics.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Seed the database with sample data
python seed_data.py

# Start the backend server
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📁 Project Structure

```
SPMLHackathon/
├── backend/
│   ├── app/
│   │   ├── routers/        # API endpoints
│   │   ├── services/       # AI logic (P2P scorer, case router, compliance)
│   │   ├── models.py       # Database models
│   │   ├── schemas.py      # Pydantic schemas
│   │   └── auth.py         # JWT authentication
│   ├── main.py             # FastAPI application
│   ├── seed_data.py        # Sample data generator
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── components/     # Sidebar, Header
    │   ├── pages/          # Dashboard, Cases, Agencies, Compliance, Portal
    │   └── services/       # API client
    └── package.json
```

## 🎯 Features

### 1. Intelligent Case Assignment
- **AI-powered routing** using Fit Score = (Performance × 0.7) + (Compliance × 0.3)
- **Rule-based fallback** when ML models are unavailable
- **Category matching** (Retail, Commercial, International)
- **Capacity management** with automatic load balancing

### 2. P2P (Probability to Pay) Scoring
- **Factors**: Days overdue, debt amount, payment history, disputes
- **ML-ready** architecture with rule-based fallback
- **Real-time scoring** on case ingestion

### 3. Compliance Monitoring
- **Aggressive language detection** via keyword matching
- **Disclosure verification** using regex patterns
- **Contact time validation** (FDCPA compliance)
- **Violation tracking** with severity levels

### 4. Analytics Dashboard
- **Real-time KPIs**: Recovery rate, SLA compliance, days overdue
- **Agency leaderboard** with performance rankings
- **Case distribution** charts
- **Recovery trends**

## 🔐 Demo Credentials

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@fedex.com | admin123 |
| Viewer | viewer@fedex.com | viewer123 |
| Agency | agent1@recovermaxsolutions.com | agent123 |

## 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/cases` | Create new case |
| `POST /api/v1/cases/assign` | Trigger auto-assignment |
| `GET /api/v1/analytics/dashboard` | Dashboard stats |
| `POST /api/v1/compliance/check-transcript` | Check text for violations |
| `GET /api/v1/agencies` | List agencies with stats |

## 🛡️ Fallback Logic

Every AI component has a rule-based fallback:

| Component | Primary | Fallback |
|-----------|---------|----------|
| P2P Scoring | XGBoost Model | Rule-based scorer |
| Case Assignment | Fit Score AI | Smart Round-Robin |
| Compliance | NLP Sentiment | Keyword Blocklist |

## 📊 KPIs

- **Recovery Rate**: Target 75%+
- **Case Allocation Time**: < 1 hour (from 2-3 days manual)
- **SLA Compliance**: 100% visibility
- **Compliance Violations**: Automated detection

## 🧪 Testing the Demo

1. **Start both servers** (backend on 8000, frontend on 3000)
2. **View Dashboard** - See KPIs and agency performance
3. **Go to Cases** - Click "Auto-Assign Cases" to see AI assignment
4. **Check Compliance** - Paste text like "I will sue you" to see violation detection
5. **Agency Portal** - View assigned cases and resolve them

## 📝 License

MIT License - Built for SPML Hackathon
