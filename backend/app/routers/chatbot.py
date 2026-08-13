"""
Chatbot API Router
Endpoints for AI Assistant chat interactions and suggestions.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from ..database import get_db
from ..auth import get_current_user
from ..models import User
from ..services.chatbot_agent import DCAChatbotAgent

router = APIRouter(
    prefix="/chatbot",
    tags=["Chatbot Agent"],
    dependencies=[Depends(get_current_user)]
)


class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


@router.post("/chat")
async def chat_with_agent(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Send message to DCA AI Chatbot Agent.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    agent = DCAChatbotAgent(db=db, user=current_user)
    response = await agent.process_message(
        user_message=request.message,
        context=request.context
    )
    return response


@router.get("/suggestions")
async def get_chat_suggestions(
    current_user: User = Depends(get_current_user)
):
    """
    Get role-aware starter prompt suggestions.
    """
    if current_user.role == "admin":
        suggestions = [
            "⚡ Auto-assign pending cases",
            "🛡️ Check transcript: 'Pay now or we will sue you'",
            "📋 Show unassigned high-priority cases",
            "🏢 Show agency workloads & leaderboard",
            "📊 Summarize system recovery stats"
        ]
    else: # Agency user
        suggestions = [
            "📋 Show my assigned cases",
            "⚡ Recommended recovery strategy for invoice",
            "🛡️ Check email compliance draft",
            "📝 Draft a settlement offer email"
        ]

    return {"suggestions": suggestions}
