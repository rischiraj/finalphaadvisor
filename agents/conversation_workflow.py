"""
LangGraph workflow for multi-turn conversations.
This workflow is separate from existing agent workflows and can coexist safely.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

from core.conversation_models import ConversationState
from agents.conversation_manager import conversation_manager
from agents.tools.intelligent_insight_generator import IntelligentInsightGenerator
from agents.enhanced_suggestion_agent import EnhancedSuggestionAgent
from core.config import get_settings


class ConversationWorkflow:
    """
    Multi-turn conversation workflow using LangGraph.
    Completely separate from existing agent workflows.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        
        # Use existing enhanced suggestion agent instead of separate LLM
        try:
            self.enhanced_agent = EnhancedSuggestionAgent()
            self.logger.info("Initialized conversation system with existing enhanced_suggestion_agent")
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced suggestion agent: {e}")
            self.enhanced_agent = None
        
        # Create workflow graph
        self.graph = self._create_conversation_graph()
    
    def _create_conversation_graph(self):
        """Create the LangGraph workflow for conversations."""
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("save_context", self._save_context_node)
        
        # Add edges
        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "save_context")
        workflow.add_edge("save_context", END)
        
        return workflow.compile()
    
    async def _retrieve_context_node(self, state: ConversationState) -> ConversationState:
        """
        Retrieve conversation context from session.
        This node gets the conversation history and analysis context.
        """
        try:
            session = conversation_manager.get_session(state.session_id)
            
            if session:
                state.session = session
                state.context = session.get_context_for_llm(include_analysis=True)
                
                # Extract analysis context and conversation history for enhanced agent
                state.analysis_context = session.analysis_context
                state.conversation_history = [
                    {"role": msg.role, "content": msg.content} 
                    for msg in session.messages
                ]
                
                # Log context info for debugging
                self.logger.debug(f"Retrieved context for session {state.session_id}: {len(state.context)} chars, {len(state.conversation_history or [])} messages")
            else:
                state.context = ""
                state.error = "Session not found or expired"
                self.logger.warning(f"Session not found: {state.session_id}")
        
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            state.error = f"Failed to retrieve context: {str(e)}"
            state.context = ""
        
        return state
    
    async def _generate_response_node(self, state: ConversationState) -> ConversationState:
        """
        Generate response using existing enhanced_suggestion_agent.
        This leverages the proven, tested agent system.
        """
        try:
            # If there was an error in previous node, handle it
            if state.error:
                state.llm_response = "I apologize, but I encountered an issue accessing our conversation history. Could you please rephrase your question?"
                return state
            
            # Use existing enhanced suggestion agent
            if not self.enhanced_agent:
                state.llm_response = "I'm currently unable to process your request. Please try again later."
                return state
            
            # Set session context for LLM logging continuity
            if hasattr(self.enhanced_agent, 'set_session_context'):
                self.enhanced_agent.set_session_context(state.session_id)
            
            # Use the existing agent's conversation capability
            # This will use the same system prompts and LLM setup as the proven system
            response = await self.enhanced_agent.generate_conversation_response(
                query=state.current_message,
                context=state.context,
                conversation_history=state.conversation_history or [],
                analysis_context=state.analysis_context
            )
            
            state.llm_response = response
            
            # Apply intelligent turn-aware token management
            is_first_turn = len(state.conversation_history or []) == 0
            is_json_response = self._is_json_analysis_response(response)
            
            if is_first_turn and is_json_response:
                # Turn 1: Full comprehensive analysis - no truncation for JSON to preserve validity
                max_response_length = None  # No limit for JSON responses
                state.is_initial_analysis = True
                self.logger.debug(f"Turn 1 - Full analysis: No truncation for JSON")
            elif is_json_response:
                # Turn 2+: Focused JSON - no truncation for JSON to preserve validity
                max_response_length = None  # No limit for JSON responses
                state.is_initial_analysis = True  # Still use JSON viewer
                self.logger.debug(f"Turn 2+ - Focused analysis: No truncation for JSON")
            else:
                # Non-JSON fallback: Standard conversational (1000 tokens ~= 4000 chars)
                max_response_length = 4000
                state.is_initial_analysis = False
                self.logger.debug(f"Non-JSON response: 1000 token limit")
            
            # Apply truncation only for non-JSON responses
            if max_response_length and len(state.llm_response) > max_response_length:
                state.llm_response = state.llm_response[:max_response_length] + "\\n\\n[Response truncated for efficiency. Please ask for specific details if needed.]"
            
            self.logger.debug(f"Generated response for session {state.session_id}: {len(state.llm_response)} chars, first_turn: {is_first_turn}")
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            state.llm_response = "I encountered an error processing your request. Please try rephrasing your question about financial analysis."
            state.error = f"Response generation failed: {str(e)}"
        
        return state
    
    async def _save_context_node(self, state: ConversationState) -> ConversationState:
        """
        Save the conversation turn (user message + assistant response).
        This maintains the conversation history for future turns.
        """
        try:
            if state.session and not state.error:
                # Add user message
                conversation_manager.add_message(
                    state.session_id, 
                    "user", 
                    state.current_message
                )
                
                # Add assistant response
                conversation_manager.add_message(
                    state.session_id, 
                    "assistant", 
                    state.llm_response
                )
                
                self.logger.debug(f"Saved conversation turn for session {state.session_id}")
        
        except Exception as e:
            self.logger.error(f"Error saving context: {e}")
            state.error = f"Failed to save conversation: {str(e)}"
        
        return state
    
    def _is_json_analysis_response(self, response: str) -> bool:
        """
        Detect if response is a JSON analysis structure.
        
        Args:
            response (str): LLM response text
            
        Returns:
            bool: True if response appears to be JSON analysis format
        """
        if not response:
            return False
            
        response_clean = response.strip()
        
        # Check for JSON structure indicators
        json_indicators = [
            response_clean.startswith('{') or response_clean.startswith('```json'),  # Handle markdown code blocks
            '"disclaimer"' in response,
            '"executive_summary"' in response,
            '"anomaly_analysis"' in response,
            '"actionable_recommendations"' in response,
            '"confidence_score"' in response
        ]
        
        # DEBUG: Log the detection process
        self.logger.info(f"JSON Detection Debug:")
        self.logger.info(f"Response length: {len(response)}")
        self.logger.info(f"Response starts with {{: {response_clean.startswith('{')}")
        self.logger.info(f"Response starts with ```json: {response_clean.startswith('```json')}")
        self.logger.info(f"Contains disclaimer: {'\"disclaimer\"' in response}")
        self.logger.info(f"Contains executive_summary: {'\"executive_summary\"' in response}")
        self.logger.info(f"Contains anomaly_analysis: {'\"anomaly_analysis\"' in response}")
        self.logger.info(f"Contains actionable_recommendations: {'\"actionable_recommendations\"' in response}")
        self.logger.info(f"Contains confidence_score: {'\"confidence_score\"' in response}")
        self.logger.info(f"JSON indicators: {json_indicators}")
        self.logger.info(f"Sum of field indicators: {sum(json_indicators[1:])}")
        
        # Consider it JSON if it starts with { or ```json and has at least 3 key fields
        result = json_indicators[0] and sum(json_indicators[1:]) >= 3
        self.logger.info(f"Final JSON detection result: {result}")
        return result
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for conversation mode.
        
        NOTE: This method is deprecated. The enhanced_suggestion_agent now uses
        centralized prompts from prompts.yaml for all conversation interactions.
        """
        # This method is no longer used - all prompts come from prompts.yaml
        return ""
    
    def _build_human_prompt(self, current_message: str, context: str) -> str:
        """Build human prompt with context and current message.
        
        NOTE: This method is deprecated. The enhanced_suggestion_agent now handles
        all prompt building using centralized prompts from prompts.yaml.
        """
        # This method is no longer used - enhanced_suggestion_agent handles all prompting
        return ""
    
    async def process_conversation_turn(
        self, 
        session_id: str, 
        user_message: str
    ) -> Dict[str, Any]:
        """
        Process a single conversation turn.
        
        Args:
            session_id: Conversation session ID
            user_message: User's message
            
        Returns:
            Dict containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Create initial state
            state = ConversationState(
                session_id=session_id,
                current_message=user_message
            )
            
            # Run through workflow
            result = await self.graph.ainvoke(state)
            
            processing_time = int((time.time() - start_time) * 1000)  # ms
            
            # Get session info for response metadata
            session_info = conversation_manager.get_session_info(session_id)
            
            return {
                "response": result["llm_response"],
                "session_id": session_id,
                "message_count": session_info["message_count"] if session_info else 0,
                "total_tokens": session_info["total_tokens"] if session_info else 0,
                "processing_time_ms": processing_time,
                "is_initial_analysis": result.get("is_initial_analysis", False),
                "error": result.get("error"),
                "success": not bool(result.get("error"))
            }
        
        except Exception as e:
            self.logger.error(f"Error processing conversation turn: {e}")
            return {
                "response": "I encountered an error processing your request. Please try again.",
                "session_id": session_id,
                "message_count": 0,
                "total_tokens": 0,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "error": str(e),
                "success": False
            }


# Global workflow instance
conversation_workflow = ConversationWorkflow()


# Convenience functions for integration
async def start_conversation_with_query(
    initial_query: str, 
    analysis_context: Optional[Dict] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Start a new conversation with an initial query.
    This is the main entry point for new conversations.
    
    Args:
        initial_query: User's first message
        analysis_context: Optional context from previous analysis
        user_id: Optional user identifier
        
    Returns:
        Dict with response and session info
    """
    try:
        # Create new session
        session_id = conversation_manager.create_session(
            analysis_context=analysis_context,
            user_id=user_id
        )
        
        # Process initial query
        result = await conversation_workflow.process_conversation_turn(
            session_id, initial_query
        )
        
        return result
    
    except Exception as e:
        logging.getLogger(__name__).error(f"Error starting conversation: {e}")
        return {
            "response": "Failed to start conversation. Please try again.",
            "session_id": None,
            "message_count": 0,
            "total_tokens": 0,
            "processing_time_ms": 0,
            "error": str(e),
            "success": False
        }


async def continue_conversation(session_id: str, user_message: str) -> Dict[str, Any]:
    """
    Continue an existing conversation.
    
    Args:
        session_id: Existing session ID
        user_message: User's message
        
    Returns:
        Dict with response and session info
    """
    return await conversation_workflow.process_conversation_turn(session_id, user_message)