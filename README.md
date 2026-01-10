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

---

## 🗄️ Database

The app uses **SQLite** - a simple file-based database (no setup required!).

```
backend/
├── dca_management.db   ← Created automatically by seed_data.py
└── seed_data.py        ← Run this to create DB + sample data
```

### What `seed_data.py` creates:
- **100 sample cases** with realistic P2P scores
- **5 agencies** (RecoverMax, Commercial Collections, etc.)
- **Demo users** (admin, viewer, agency agents)
- **10 sample violations**

### Reset Database
```bash
del backend\dca_management.db   # Windows
# rm backend/dca_management.db  # Linux/Mac
python seed_data.py
```

> **Note**: The database file is created when you first run `seed_data.py`. It doesn't exist until then!

---

## � Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   INGEST     │────▶│  P2P SCORE   │────▶│   ASSIGN     │────▶│   COLLECT    │
│   Case       │     │  ML Model    │     │  AI Routing  │     │   Agency     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │                    │                    │
                            ▼                    ▼                    ▼
                     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
                     │  Rule-based  │     │  Round-Robin │     │  Compliance  │
                     │  Fallback    │     │  Fallback    │     │  Monitor     │
                     └──────────────┘     └──────────────┘     └──────────────┘
```

| Stage | Component | Location |
|-------|-----------|----------|
| **P2P Scoring** | PyTorch Neural Network | `app/ml/models.py` |
| **Case Assignment** | Fit Score Algorithm | `app/services/case_router.py` |
| **Compliance** | Keyword/Regex Detection | `app/services/compliance_checker.py` |
| **Fallback** | Rule-based logic | Built into each service |

---

## �📁 Project Structure

```
SPMLHackathon/
├── backend/
│   ├── app/
│   │   ├── ml/             # ML model definitions
│   │   ├── routers/        # API endpoints
│   │   ├── services/       # AI logic (P2P, router, compliance)
│   │   └── models.py       # Database models
│   ├── models/             # Trained .pth models
│   ├── main.py
│   └── seed_data.py
│
└── frontend/
    ├── src/
    │   ├── components/
    │   ├── pages/
    │   └── services/
    └── package.json
```

---

## 🎯 Features

### 1. Intelligent Case Assignment
- **AI-powered routing** using Fit Score algorithm
- **PyTorch model support** for custom ML
- **Rule-based fallback** when models unavailable

### 2. P2P (Probability to Pay) Scoring
- **8-feature neural network** (or rule-based fallback)
- **Real-time scoring** on case ingestion
- **Train on Kaggle** with your own data

### 3. Compliance Monitoring
- **Aggressive language detection**
- **Disclosure verification** (FDCPA)
- **Contact time validation**

### 4. Analytics Dashboard
- Real-time KPIs
- Agency leaderboard
- Recovery trends

---

## 🔐 Demo Credentials

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@fedex.com | admin123 |
| Viewer | viewer@fedex.com | viewer123 |
| Agency | agent1@recovermaxsolutions.com | agent123 |

---

## 🛡️ Fallback Logic

Every AI component has a rule-based fallback:

| Component | Primary | Fallback |
|-----------|---------|----------|
| P2P Scoring | PyTorch P2PNet | Rule-based scorer |
| Case Assignment | Fit Score AI | Smart Round-Robin |
| Compliance | LSTM Model | Keyword Blocklist |

---

## � API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/cases` | Create new case |
| `POST /api/v1/cases/assign` | Trigger auto-assignment |
| `GET /api/v1/analytics/dashboard` | Dashboard stats |
| `POST /api/v1/compliance/check-transcript` | Check for violations |
| `GET /api/v1/agencies` | List agencies with stats |

---

## 🧪 Testing the Demo

1. **Start both servers** (backend: 8000, frontend: 3000)
2. **Dashboard** - View KPIs and agency performance
3. **Cases** → Click **"Auto-Assign Cases"**
4. **Compliance** → Paste `"I will sue you"` to see detection
5. **Agency Portal** - View and resolve cases

---

## 📝 License

MIT License - Built for SPML Hackathon
