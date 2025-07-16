"""
Conversation models for multi-turn chat functionality.
These models are SEPARATE from existing core models and don't modify them.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid

# Import existing models for integration (read-only)
from core.models import AnomalyResult, TimeSeriesData, AnalysisResponse


class ConversationMessage(BaseModel):
    """Individual message in a conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    token_count: int = Field(0, description="Estimated token count")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be user, assistant, or system')
        return v
    
    @validator('content')
    def estimate_tokens(cls, v):
        # Simple token estimation: ~4 characters per token
        return v
    
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = max(1, len(self.content) // 4)


class ConversationSession(BaseModel):
    """Multi-turn conversation session."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    
    # Link to existing analysis (optional)
    analysis_context: Optional[Dict[str, Any]] = Field(None, description="Context from original analysis")
    original_analysis_id: Optional[str] = Field(None, description="ID of original analysis")
    file_path: Optional[str] = Field(None, description="Path to analyzed file")
    
    # Token management settings (configurable)
    max_context_tokens: int = Field(1200, description="Max tokens for conversation context")
    max_message_tokens: int = Field(400, description="Max tokens per message")
    max_messages: int = Field(10, description="Max messages to retain")
    
    # Session metadata
    user_id: Optional[str] = Field(None, description="User identifier if available")
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> ConversationMessage:
        """Add a message to the conversation with automatic token management."""
        
        # Truncate if message is too long
        if len(content) > self.max_message_tokens * 4:
            content = content[:self.max_message_tokens * 4] + "... [truncated for context efficiency]"
        
        # Create message
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        self.last_active = datetime.now()
        
        # Apply token management
        self._manage_conversation_length()
        
        return message
    
    def _manage_conversation_length(self):
        """Keep conversation within token and message limits."""
        # Remove old messages if we exceed limits
        total_tokens = sum(msg.token_count for msg in self.messages)
        
        while (total_tokens > self.max_context_tokens or 
               len(self.messages) > self.max_messages) and len(self.messages) > 2:
            removed_msg = self.messages.pop(0)  # Remove oldest message
            total_tokens -= removed_msg.token_count
    
    def get_context_for_llm(self, include_analysis: bool = True) -> str:
        """Build optimized context string for LLM input."""
        context_parts = []
        
        # Include analysis context if available and requested
        if include_analysis and self.analysis_context:
            analysis_summary = self._format_analysis_context()
            if analysis_summary:
                context_parts.append("Previous Analysis Context:")
                context_parts.append(analysis_summary)
                context_parts.append("")
        
        # Include conversation history
        if self.messages:
            # For longer conversations, use smart summarization
            if len(self.messages) > 6:
                context_parts.extend(self._get_summarized_context())
            else:
                context_parts.append("Conversation History:")
                for msg in self.messages:
                    context_parts.append(f"{msg.role.title()}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def _format_analysis_context(self) -> str:
        """Format analysis context in a compact way."""
        if not self.analysis_context:
            return ""
        
        parts = []
        if company := self.analysis_context.get('company'):
            parts.append(f"Company: {company}")
        if method := self.analysis_context.get('method'):
            parts.append(f"Method: {method}")
        if count := self.analysis_context.get('anomaly_count'):
            parts.append(f"Anomalies Found: {count}")
        if file_name := self.analysis_context.get('file_name'):
            parts.append(f"Data Source: {file_name}")
        
        return " | ".join(parts) if parts else ""
    
    def _get_summarized_context(self) -> List[str]:
        """Get summarized context for longer conversations."""
        context_parts = []
        
        # Take last 4 messages for detailed context
        recent_messages = self.messages[-4:]
        older_messages = self.messages[:-4]
        
        # Simple summarization of older messages
        if older_messages:
            topics = self._extract_conversation_topics(older_messages)
            if topics:
                context_parts.append(f"Earlier Discussion: {', '.join(topics)}")
                context_parts.append("")
        
        # Detailed recent context
        context_parts.append("Recent Conversation:")
        for msg in recent_messages:
            context_parts.append(f"{msg.role.title()}: {msg.content}")
        
        return context_parts
    
    def _extract_conversation_topics(self, messages: List[ConversationMessage]) -> List[str]:
        """Extract key topics from conversation messages."""
        topics = set()
        
        for msg in messages:
            if msg.role == 'user':
                content_lower = msg.content.lower()
                
                # Financial topic detection
                if any(word in content_lower for word in ['risk', 'risks']):
                    topics.add('risk analysis')
                if any(word in content_lower for word in ['buy', 'sell', 'trade']):
                    topics.add('trading decisions')
                if any(word in content_lower for word in ['price', 'target', 'valuation']):
                    topics.add('price analysis')
                if any(word in content_lower for word in ['earnings', 'revenue', 'profit']):
                    topics.add('earnings discussion')
                if any(word in content_lower for word in ['correlation', 'correlate']):
                    topics.add('correlation analysis')
        
        return list(topics)[:3]  # Return max 3 topics
    
    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)
    
    def get_total_tokens(self) -> int:
        """Get estimated total tokens in conversation."""
        return sum(msg.token_count for msg in self.messages)
    
    def is_expired(self, timeout_minutes: int = 120) -> bool:
        """Check if session is expired."""
        from datetime import timedelta
        timeout = timedelta(minutes=timeout_minutes)
        return datetime.now() - self.last_active > timeout


class ConversationState(BaseModel):
    """LangGraph state for conversation workflow."""
    session_id: str
    current_message: str
    session: Optional[ConversationSession] = None
    context: Optional[str] = None
    llm_response: Optional[str] = None
    error: Optional[str] = None
    analysis_context: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    is_initial_analysis: bool = Field(False, description="Flag indicating this is first turn with full analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationRequest(BaseModel):
    """Request model for conversation API."""
    message: str = Field(..., description="User message")
    include_analysis_context: bool = Field(True, description="Include previous analysis context")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class ConversationResponse(BaseModel):
    """Response model for conversation API."""
    session_id: str
    response: str
    message_count: int
    total_tokens: int
    processing_time_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StartConversationRequest(BaseModel):
    """Request to start a new conversation."""
    initial_query: str = Field(..., description="Initial user query")
    analysis_context: Optional[Dict[str, Any]] = Field(None, description="Context from previous analysis")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_config: Optional[Dict[str, Any]] = Field(None, description="Session configuration overrides")


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    session_id: str
    messages: List[ConversationMessage]
    analysis_context: Optional[Dict[str, Any]]
    created_at: datetime
    last_active: datetime
    total_tokens: int
    message_count: int


# Integration helpers for existing codebase
class AnalysisToConversationAdapter:
    """Adapter to convert existing analysis results to conversation context."""
    
    @staticmethod
    def create_context_from_analysis(analysis_response: AnalysisResponse, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Convert AnalysisResponse to conversation context."""
        context = {
            "anomaly_count": analysis_response.anomaly_result.anomaly_count,
            "method": analysis_response.anomaly_result.method_used,
            "threshold": analysis_response.anomaly_result.threshold_used,
            "total_points": analysis_response.anomaly_result.total_points,
            "anomaly_percentage": analysis_response.anomaly_result.anomaly_percentage,
            "processing_time": analysis_response.processing_time,
        }
        
        if file_path:
            context["file_path"] = file_path
            context["file_name"] = file_path.split("/")[-1] if "/" in file_path else file_path
            # Extract company name from filename
            context["company"] = AnalysisToConversationAdapter._extract_company_name(context["file_name"])
        
        if analysis_response.insights:
            # Store full insights object for conversation use
            context["insights_summary"] = analysis_response.insights.summary
            context["full_insights"] = {
                "summary": analysis_response.insights.summary,
                "anomaly_explanations": analysis_response.insights.anomaly_explanations,
                "recommendations": analysis_response.insights.recommendations,
                "root_causes": analysis_response.insights.root_causes,
                "confidence_score": analysis_response.insights.confidence_score
            }
            # Also store any additional insights fields
            if hasattr(analysis_response.insights, 'actionable_insights'):
                context["full_insights"]["actionable_insights"] = analysis_response.insights.actionable_insights
            if hasattr(analysis_response.insights, 'trading_signals'):
                context["full_insights"]["trading_signals"] = analysis_response.insights.trading_signals
            if hasattr(analysis_response.insights, 'risk_assessment'):
                context["full_insights"]["risk_assessment"] = analysis_response.insights.risk_assessment
        
        return context
    
    @staticmethod
    def _extract_company_name(filename: str) -> str:
        """Extract company name from filename."""
        filename_upper = filename.upper()
        
        # Common stock symbols
        symbols = {
            'NVDA': 'NVIDIA',
            'NVIDIA': 'NVIDIA',
            'AAPL': 'Apple',
            'APPLE': 'Apple',
            'MSFT': 'Microsoft',
            'MICROSOFT': 'Microsoft',
            'TSLA': 'Tesla',
            'TESLA': 'Tesla',
            'AMZN': 'Amazon',
            'AMAZON': 'Amazon',
            'GOOGL': 'Google',
            'GOOGLE': 'Google',
            'META': 'Meta',
            'FB': 'Meta'
        }
        
        for symbol, company in symbols.items():
            if symbol in filename_upper:
                return company
        
        return "Unknown Company"