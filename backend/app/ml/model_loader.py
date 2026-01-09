"""
ML Model Loader
Loads .pth PyTorch models trained externally (e.g., on Kaggle)
"""
import os
import torch
import logging
from typing import Optional, Dict, Any

from .models import P2PNet, ComplianceNet, AgencyFitNet

logger = logging.getLogger(__name__)

# Model paths - place your trained .pth files here
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

MODEL_PATHS = {
    "p2p": os.path.join(MODELS_DIR, "p2p_model.pth"),
    "compliance": os.path.join(MODELS_DIR, "compliance_model.pth"),
    "agency_fit": os.path.join(MODELS_DIR, "agency_fit_model.pth"),
}


class ModelLoader:
    """Singleton loader for ML models"""
    
    _instance = None
    _models: Dict[str, Any] = {}
    _loaded: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._loaded:
            self._load_all_models()
            self._loaded = True
    
    def _load_all_models(self):
        """Load all available models"""
        # Ensure models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Try to load each model
        self._load_p2p_model()
        self._load_compliance_model()
        self._load_agency_fit_model()
    
    def _load_p2p_model(self):
        """Load P2P scoring model"""
        path = MODEL_PATHS["p2p"]
        if os.path.exists(path):
            try:
                model = P2PNet()
                model.load_state_dict(torch.load(path, map_location="cpu"))
                model.eval()
                self._models["p2p"] = model
                logger.info(f"✅ Loaded P2P model from {path}")
            except Exception as e:
                logger.warning(f"❌ Failed to load P2P model: {e}")
                self._models["p2p"] = None
        else:
            logger.info(f"⚠️ P2P model not found at {path} - using fallback")
            self._models["p2p"] = None
    
    def _load_compliance_model(self):
        """Load compliance detection model"""
        path = MODEL_PATHS["compliance"]
        if os.path.exists(path):
            try:
                model = ComplianceNet()
                model.load_state_dict(torch.load(path, map_location="cpu"))
                model.eval()
                self._models["compliance"] = model
                logger.info(f"✅ Loaded Compliance model from {path}")
            except Exception as e:
                logger.warning(f"❌ Failed to load Compliance model: {e}")
                self._models["compliance"] = None
        else:
            logger.info(f"⚠️ Compliance model not found at {path} - using fallback")
            self._models["compliance"] = None
    
    def _load_agency_fit_model(self):
        """Load agency fit model"""
        path = MODEL_PATHS["agency_fit"]
        if os.path.exists(path):
            try:
                model = AgencyFitNet()
                model.load_state_dict(torch.load(path, map_location="cpu"))
                model.eval()
                self._models["agency_fit"] = model
                logger.info(f"✅ Loaded Agency Fit model from {path}")
            except Exception as e:
                logger.warning(f"❌ Failed to load Agency Fit model: {e}")
                self._models["agency_fit"] = None
        else:
            logger.info(f"⚠️ Agency Fit model not found at {path} - using fallback")
            self._models["agency_fit"] = None
    
    def get_model(self, name: str) -> Optional[torch.nn.Module]:
        """Get a loaded model by name"""
        return self._models.get(name)
    
    def is_model_available(self, name: str) -> bool:
        """Check if a model is loaded and available"""
        return self._models.get(name) is not None
    
    def reload_model(self, name: str) -> bool:
        """Reload a specific model (useful for hot-swapping after training)"""
        if name == "p2p":
            self._load_p2p_model()
        elif name == "compliance":
            self._load_compliance_model()
        elif name == "agency_fit":
            self._load_agency_fit_model()
        else:
            logger.warning(f"Unknown model name: {name}")
            return False
        return self.is_model_available(name)


# Singleton instance
_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get the singleton model loader"""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader


def predict_p2p(features: torch.Tensor) -> Optional[float]:
    """
    Predict P2P score using the ML model
    
    Args:
        features: Tensor of shape (8,) with normalized features
        
    Returns:
        P2P score (0-1) or None if model not available
    """
    loader = get_model_loader()
    model = loader.get_model("p2p")
    
    if model is None:
        return None
    
    with torch.no_grad():
        # Add batch dimension
        if features.dim() == 1:
            features = features.unsqueeze(0)
        output = model(features)
        return output.item()


def predict_compliance_violation(embeddings: torch.Tensor) -> Optional[float]:
    """
    Predict compliance violation probability
    
    Args:
        embeddings: Tensor of shape (seq_len, embedding_dim) or (batch, seq_len, embedding_dim)
        
    Returns:
        Violation probability (0-1) or None if model not available
    """
    loader = get_model_loader()
    model = loader.get_model("compliance")
    
    if model is None:
        return None
    
    with torch.no_grad():
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        output = model(embeddings)
        return output.item()


def predict_agency_fit(features: torch.Tensor) -> Optional[float]:
    """
    Predict agency-case fit score
    
    Args:
        features: Tensor of shape (12,) with agency-case features
        
    Returns:
        Fit score (0-1) or None if model not available
    """
    loader = get_model_loader()
    model = loader.get_model("agency_fit")
    
    if model is None:
        return None
    
    with torch.no_grad():
        if features.dim() == 1:
            features = features.unsqueeze(0)
        output = model(features)
        return output.item()
