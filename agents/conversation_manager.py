"""
Conversation session manager for multi-turn chat functionality.
This is completely separate from existing agent functionality.
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
import asyncio
import logging
import yaml
from pathlib import Path

from core.conversation_models import ConversationSession, ConversationMessage, AnalysisToConversationAdapter
from core.models import AnalysisResponse


class ConversationManager:
    """
    Manages conversation sessions for multi-turn chat.
    Completely independent of existing CLI/API functionality.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.sessions: Dict[str, ConversationSession] = {}
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Session settings from config
        session_config = self.config.get('conversation_settings', {}).get('session_management', {})
        self.session_timeout = timedelta(minutes=session_config.get('timeout_minutes', 120))
        self.max_concurrent_sessions = session_config.get('max_concurrent_sessions', 100)
        self.cleanup_interval = timedelta(minutes=session_config.get('cleanup_interval_minutes', 15))
        
        # Token settings from config
        token_config = self.config.get('conversation_settings', {}).get('token_management', {})
        self.default_max_context_tokens = token_config.get('optimization', {}).get('max_message_length', 400) * 3  # ~3 messages worth
        self.default_max_message_tokens = token_config.get('optimization', {}).get('max_message_length', 400)
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load conversation configuration."""
        if config_path is None:
            config_path = "config/conversation_config.yaml"
        
        config_file = Path(config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                self.logger.warning(f"Failed to load conversation config: {e}")
                return {}
        else:
            self.logger.info(f"Conversation config not found at {config_path}, using defaults")
            return {}
    
    def create_session(
        self, 
        analysis_context: Optional[Dict] = None,
        user_id: Optional[str] = None,
        session_config: Optional[Dict] = None
    ) -> str:
        """
        Create a new conversation session.
        
        Args:
            analysis_context: Context from previous analysis (optional)
            user_id: User identifier (optional)
            session_config: Override default session settings (optional)
            
        Returns:
            session_id: Unique identifier for the session
        """
        # Check session limit
        if len(self.sessions) >= self.max_concurrent_sessions:
            self._cleanup_expired_sessions()
            if len(self.sessions) >= self.max_concurrent_sessions:
                raise RuntimeError("Maximum concurrent sessions reached")
        
        # Create session with configuration
        session = ConversationSession(
            analysis_context=analysis_context,
            user_id=user_id,
            max_context_tokens=session_config.get('max_context_tokens', self.default_max_context_tokens) if session_config else self.default_max_context_tokens,
            max_message_tokens=session_config.get('max_message_tokens', self.default_max_message_tokens) if session_config else self.default_max_message_tokens
        )
        
        self.sessions[session.session_id] = session
        
        self.logger.info(f"Created conversation session: {session.session_id}")
        return session.session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get an existing session if it exists and is not expired.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationSession or None if not found/expired
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session is expired
        if session.is_expired(timeout_minutes=int(self.session_timeout.total_seconds() // 60)):
            self.logger.info(f"Session expired: {session_id}")
            del self.sessions[session_id]
            return None
        
        return session
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a message to an existing session.
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            bool: True if message was added successfully
        """
        session = self.get_session(session_id)
        if not session:
            self.logger.warning(f"Attempted to add message to non-existent session: {session_id}")
            return False
        
        session.add_message(role, content, metadata)
        self.logger.debug(f"Added {role} message to session {session_id}")
        return True
    
    def get_conversation_context(
        self, 
        session_id: str, 
        include_analysis: bool = True
    ) -> Optional[str]:
        """
        Get formatted conversation context for LLM input.
        
        Args:
            session_id: Session identifier
            include_analysis: Whether to include analysis context
            
        Returns:
            Formatted context string or None if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        return session.get_context_for_llm(include_analysis=include_analysis)
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get session information for monitoring/debugging.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session info dict or None if not found
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "last_active": session.last_active,
            "message_count": session.get_message_count(),
            "total_tokens": session.get_total_tokens(),
            "has_analysis_context": session.analysis_context is not None,
            "user_id": session.user_id
        }
    
    def list_active_sessions(self) -> List[Dict]:
        """
        List all active sessions with basic info.
        
        Returns:
            List of session info dicts
        """
        sessions_info = []
        for session_id in list(self.sessions.keys()):  # Create copy to avoid modification during iteration
            info = self.get_session_info(session_id)
            if info:  # Session still exists after get_session_info call
                sessions_info.append(info)
        
        return sessions_info
    
    def end_session(self, session_id: str) -> bool:
        """
        Manually end a conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if session was ended successfully
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"Ended conversation session: {session_id}")
            return True
        return False
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_active > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            self.logger.debug(f"Cleaned up expired session: {session_id}")
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval.total_seconds())
                    self._cleanup_expired_sessions()
                except Exception as e:
                    self.logger.error(f"Error in cleanup task: {e}")
        
        # Only start if not already running and there's a running event loop
        try:
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(cleanup_loop())
        except RuntimeError:
            # No running event loop - cleanup will happen manually
            self.logger.debug("No running event loop, skipping background cleanup task")
    
    def stop_cleanup_task(self):
        """Stop background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
    
    def get_stats(self) -> Dict:
        """Get conversation manager statistics."""
        return {
            "total_sessions": len(self.sessions),
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "session_timeout_minutes": int(self.session_timeout.total_seconds() // 60),
            "cleanup_interval_minutes": int(self.cleanup_interval.total_seconds() // 60),
            "default_max_context_tokens": self.default_max_context_tokens,
            "default_max_message_tokens": self.default_max_message_tokens
        }
    
    # Integration methods for existing codebase
    def create_session_from_analysis(
        self, 
        analysis_response: AnalysisResponse, 
        file_path: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Create a conversation session from an existing analysis result.
        This integrates with the existing analysis workflow.
        
        Args:
            analysis_response: Result from existing analysis
            file_path: Path to the analyzed file
            user_id: User identifier
            
        Returns:
            session_id: New session identifier
        """
        # Convert analysis to conversation context
        analysis_context = AnalysisToConversationAdapter.create_context_from_analysis(
            analysis_response, file_path
        )
        
        # Create session
        session_id = self.create_session(
            analysis_context=analysis_context,
            user_id=user_id
        )
        
        self.logger.info(f"Created conversation session from analysis: {session_id}")
        return session_id


# Global instance for use across the application
# This won't interfere with existing functionality
conversation_manager = ConversationManager()


# Configuration helper
def get_conversation_config() -> Dict:
    """Get conversation configuration for other modules."""
    return conversation_manager.config