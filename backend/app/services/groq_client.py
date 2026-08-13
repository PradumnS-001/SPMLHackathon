"""
Groq API Async Client
Ultra-low-latency LLM inference engine using Groq LPU platform.
"""
import os
import httpx
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient:
    """Async client for Groq Chat Completions API."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables. Falling back to local agent engine.")
    
    @property
    def is_configured(self) -> bool:
        """Check if Groq API key is set."""
        return bool(self.api_key and not self.api_key.startswith("your-"))
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Send a chat completion request to Groq API.
        
        Args:
            messages: List of message objects [{"role": "system"/"user"/"assistant", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Optional format e.g. {"type": "json_object"}
            
        Returns:
            Generated content string or None if request failed/not configured
        """
        if not self.is_configured:
            return None
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            payload["response_format"] = response_format
            
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(GROQ_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                content = data["choices"][0]["message"]["content"]
                return content
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return None


# Singleton instance
_groq_client: Optional[GroqClient] = None


def get_groq_client() -> GroqClient:
    """Get or create singleton Groq client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client
