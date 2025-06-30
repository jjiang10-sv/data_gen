import os
import time
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from starfish.common.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

# Global variable to track startup time
startup_time = datetime.now()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "starfish-backend", "timestamp": datetime.now().isoformat(), "uptime": str(datetime.now() - startup_time)}


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with dependency status"""
    try:
        # Check environment variables
        env_status = {
            "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        }

        # Check storage (if available)
        storage_status = "unknown"
        try:
            # Add your storage check logic here
            storage_status = "healthy"
        except Exception as e:
            storage_status = f"error: {str(e)}"

        return {
            "status": "healthy",
            "service": "starfish-backend",
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - startup_time),
            "environment": env_status,
            "storage": storage_status,
            "version": "1.0.0",
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for Kubernetes"""
    try:
        # Add your readiness checks here
        # For example, check database connections, external services, etc.

        return {"status": "ready", "service": "starfish-backend", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check for Kubernetes"""
    return {"status": "alive", "service": "starfish-backend", "timestamp": datetime.now().isoformat()}
