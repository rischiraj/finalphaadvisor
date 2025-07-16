"""
API endpoints for multi-turn conversation functionality.
These endpoints are SEPARATE from existing API endpoints and won't break them.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse

# Import existing dependencies - these are already working
from api.dependencies import (
    SupervisorDep, 
    SettingsDep, 
    LoggerDep,
    ValidatedSettingsDep
)

# Import new conversation models
from core.conversation_models import (
    ConversationRequest,
    ConversationResponse, 
    StartConversationRequest,
    ConversationHistoryResponse,
    AnalysisToConversationAdapter
)

# Import conversation workflow
from agents.conversation_workflow import start_conversation_with_query, continue_conversation
from agents.conversation_manager import conversation_manager

# Import existing models for integration
from core.models import AnalysisResponse


# Create separate router for conversation endpoints
conversation_router = APIRouter(prefix="/conversation", tags=["Multi-Turn Conversation"])
logger = logging.getLogger(__name__)


@conversation_router.post(
    "/start",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
    summary="Start a new multi-turn conversation",
    description="Start a new conversation session, optionally with context from previous analysis"
)
async def start_conversation(
    request: StartConversationRequest,
    logger: LoggerDep = None
) -> ConversationResponse:
    """
    Start a new multi-turn conversation session.
    
    This endpoint creates a new conversation and processes the initial query.
    It can optionally include context from a previous analysis.
    """
    try:
        start_time = time.time()
        
        logger.info(f"Starting new conversation with query: {request.initial_query[:100]}...")
        
        # Start conversation using workflow
        result = await start_conversation_with_query(
            initial_query=request.initial_query,
            analysis_context=request.analysis_context,
            user_id=request.user_id
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start conversation: {result.get('error', 'Unknown error')}"
            )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return ConversationResponse(
            session_id=result["session_id"],
            response=result["response"],
            message_count=result["message_count"],
            total_tokens=result["total_tokens"],
            processing_time_ms=processing_time,
            metadata={
                "conversation_started": True,
                "has_analysis_context": bool(request.analysis_context)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start conversation: {str(e)}"
        )


@conversation_router.post(
    "/{session_id}/message",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
    summary="Continue an existing conversation",
    description="Send a message to continue an existing conversation session"
)
async def send_message(
    session_id: str,
    request: ConversationRequest,
    logger: LoggerDep = None
) -> ConversationResponse:
    """
    Continue an existing conversation with a new message.
    
    This endpoint processes a user message within an existing conversation context.
    """
    try:
        start_time = time.time()
        
        logger.info(f"Processing message for session {session_id}: {request.message[:100]}...")
        
        # Continue conversation using workflow
        result = await continue_conversation(session_id, request.message)
        
        if not result["success"]:
            if "not found" in result.get("error", "").lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation session not found or expired"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to process message: {result.get('error', 'Unknown error')}"
                )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return ConversationResponse(
            session_id=result["session_id"],
            response=result["response"],
            message_count=result["message_count"],
            total_tokens=result["total_tokens"],
            processing_time_ms=processing_time,
            metadata=request.metadata or {}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )


@conversation_router.get(
    "/{session_id}/history",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history",
    description="Retrieve the complete history of a conversation session"
)
async def get_conversation_history(
    session_id: str,
    logger: LoggerDep = None
) -> ConversationHistoryResponse:
    """
    Get the complete history of a conversation session.
    """
    try:
        session = conversation_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation session not found or expired"
            )
        
        return ConversationHistoryResponse(
            session_id=session.session_id,
            messages=session.messages,
            analysis_context=session.analysis_context,
            created_at=session.created_at,
            last_active=session.last_active,
            total_tokens=session.get_total_tokens(),
            message_count=session.get_message_count()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation history: {str(e)}"
        )


@conversation_router.delete(
    "/{session_id}",
    summary="End conversation session",
    description="Manually end a conversation session and clean up resources"
)
async def end_conversation(
    session_id: str,
    logger: LoggerDep = None
) -> Dict[str, str]:
    """
    End a conversation session manually.
    """
    try:
        success = conversation_manager.end_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation session not found"
            )
        
        logger.info(f"Ended conversation session: {session_id}")
        return {"message": "Conversation ended successfully", "session_id": session_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end conversation: {str(e)}"
        )


@conversation_router.get(
    "/sessions/active",
    summary="List active conversation sessions",
    description="Get a list of all currently active conversation sessions (admin endpoint)"
)
async def list_active_sessions(
    logger: LoggerDep = None
) -> Dict[str, Any]:
    """
    List all active conversation sessions.
    This is primarily for monitoring and debugging.
    """
    try:
        sessions = conversation_manager.list_active_sessions()
        stats = conversation_manager.get_stats()
        
        return {
            "active_sessions": sessions,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}"
        )


# Integration endpoint: Analyze file and start conversation
@conversation_router.post(
    "/analyze-and-chat",
    response_model=Dict[str, Any],
    summary="Analyze file and start conversation",
    description="Upload a file for analysis and immediately start a conversation about the results"
)
async def analyze_and_start_conversation(
    uploaded_file: UploadFile = File(..., description="CSV or Excel file for analysis"),
    method: str = Form("rolling-iqr", description="Anomaly detection method"),
    threshold: float = Form(1.5, description="Detection threshold"),
    initial_query: str = Form("Analyze these results for trading opportunities", description="Initial conversation query"),
    supervisor: SupervisorDep = None,
    settings: ValidatedSettingsDep = None,
    logger: LoggerDep = None
) -> Dict[str, Any]:
    """
    Analyze an uploaded file and immediately start a conversation about the results.
    
    This endpoint combines the existing analysis functionality with conversation startup.
    It's a convenience endpoint that integrates both workflows.
    """
    try:
        start_time = time.time()
        
        logger.info(f"Starting analyze-and-chat workflow for file: {uploaded_file.filename}")
        
        # Step 1: Save uploaded file (reuse existing logic)
        from pathlib import Path
        import uuid
        
        file_id = str(uuid.uuid4())[:8]
        file_extension = Path(uploaded_file.filename).suffix
        filename = f"{file_id}_{uploaded_file.filename}"
        file_path = settings.data_dir / filename
        
        # Save file
        content = await uploaded_file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Step 2: Run analysis using existing supervisor
        from core.models import AnomalyDetectionRequest
        
        analysis_request = AnomalyDetectionRequest(
            file_path=str(file_path),
            method=method,
            threshold=threshold,
            query=initial_query
        )
        
        # Use existing analysis workflow
        analysis_result = await supervisor.analyze(analysis_request)
        
        # Step 3: Create conversation context from analysis
        analysis_context = AnalysisToConversationAdapter.create_context_from_analysis(
            analysis_result, str(file_path)
        )
        
        # Step 4: Start conversation with analysis context
        conversation_result = await start_conversation_with_query(
            initial_query=initial_query,
            analysis_context=analysis_context
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "analysis_result": {
                "anomaly_count": analysis_result.anomaly_result.anomaly_count,
                "method_used": analysis_result.anomaly_result.method_used,
                "total_points": analysis_result.anomaly_result.total_points,
                "anomaly_percentage": analysis_result.anomaly_result.anomaly_percentage,
                "visualization": {
                    "plot_path": analysis_result.visualization.plot_path,
                    "plot_description": analysis_result.visualization.plot_description,
                    "plot_type": analysis_result.visualization.plot_type
                }
            },
            "conversation": {
                "session_id": conversation_result["session_id"],
                "initial_response": conversation_result["response"],
                "message_count": conversation_result["message_count"]
            },
            "file_info": {
                "original_name": uploaded_file.filename,
                "saved_path": str(file_path),
                "file_size": len(content)
            },
            "processing_time_ms": processing_time,
            "success": True
        }
    
    except Exception as e:
        logger.error(f"Error in analyze-and-chat workflow: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analyze-and-chat workflow failed: {str(e)}"
        )


# Health check for conversation system
@conversation_router.get(
    "/health",
    summary="Conversation system health check",
    description="Check if the conversation system is working properly"
)
async def conversation_health_check() -> Dict[str, Any]:
    """
    Health check for the conversation system.
    """
    try:
        stats = conversation_manager.get_stats()
        
        return {
            "status": "healthy",
            "conversation_system": "operational", 
            "active_sessions": stats["total_sessions"],
            "max_sessions": stats["max_concurrent_sessions"],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }