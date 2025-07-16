"""
Custom LangChain callback for detailed LLM logging.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


class LLMDebugCallback(BaseCallbackHandler):
    """Custom callback handler to log all LLM interactions in detail."""
    
    def __init__(self, logger_name: str = "llm_debug"):
        """Initialize the callback handler."""
        self.logger = logging.getLogger(logger_name)
        self.call_count = 0
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Log when LLM starts processing."""
        self.call_count += 1
        self.logger.debug(f"\n{'='*100}")
        self.logger.debug(f"[LLM] CALL #{self.call_count} STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.debug(f"{'='*100}")
        
        # Log the model info
        model_name = serialized.get("name", "Unknown")
        self.logger.debug(f"[MODEL] {model_name}")
        
        # Log invocation parameters if present
        if kwargs.get('invocation_params'):
            params = kwargs['invocation_params']
            self.logger.debug(f"[PARAM] Temperature: {params.get('temperature', 'N/A')}")
            self.logger.debug(f"[PARAM] Max Tokens: {params.get('max_tokens', 'N/A')}")
        
        # Log kwargs if present
        if kwargs:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['invocation_params']}
            if filtered_kwargs:
                self.logger.debug(f"[PARAM] Parameters: {filtered_kwargs}")
        
        # Log all prompts with detailed formatting
        for i, prompt in enumerate(prompts):
            self.logger.debug(f"\n[PROMPT] FULL PROMPT #{i+1}:")
            self.logger.debug(f"{'-'*60}")
            self.logger.debug(f"PROMPT LENGTH: {len(prompt)} characters")
            self.logger.debug(f"{'-'*60}")
            self.logger.debug(prompt)
            self.logger.debug(f"{'-'*60}")
            self.logger.debug(f"END PROMPT #{i+1}")
            self.logger.debug(f"{'-'*60}")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log when LLM finishes processing."""
        self.logger.debug(f"\n[RESPONSE] LLM RESPONSE:")
        self.logger.debug(f"{'-'*80}")
        
        # Log each generation with detailed formatting
        for i, generation in enumerate(response.generations):
            for j, gen in enumerate(generation):
                self.logger.debug(f"\n[RESPONSE] {i+1}.{j+1}:")
                self.logger.debug(f"{'-'*60}")
                self.logger.debug(f"RESPONSE LENGTH: {len(gen.text)} characters")
                self.logger.debug(f"{'-'*60}")
                self.logger.debug(gen.text)
                self.logger.debug(f"{'-'*60}")
                self.logger.debug(f"END RESPONSE {i+1}.{j+1}")
                self.logger.debug(f"{'-'*60}")
        
        # Log token usage if available
        if response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            if usage:
                self.logger.debug(f"\n[USAGE] TOKEN USAGE:")
                for key, value in usage.items():
                    self.logger.debug(f"   {key}: {value}")
            
            # Log other LLM output info
            other_info = {k: v for k, v in response.llm_output.items() if k != 'token_usage'}
            if other_info:
                self.logger.debug(f"\n[OUTPUT] LLM OUTPUT INFO:")
                for key, value in other_info.items():
                    self.logger.debug(f"   {key}: {value}")
        
        self.logger.debug(f"\n[COMPLETE] LLM CALL #{self.call_count} COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.debug(f"{'='*100}\n")
    
    def on_llm_error(
        self, 
        error: Union[Exception, KeyboardInterrupt], 
        **kwargs: Any
    ) -> None:
        """Log when LLM encounters an error."""
        self.logger.error(f"\n[ERROR] LLM CALL #{self.call_count} FAILED")
        self.logger.error(f"{'='*80}")
        self.logger.error(f"Error: {str(error)}")
        self.logger.error(f"{'='*80}\n")
    
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Log when chat model starts processing."""
        self.call_count += 1
        self.logger.debug(f"\n{'='*80}")
        self.logger.debug(f"[CHAT] MODEL CALL #{self.call_count} STARTED")
        self.logger.debug(f"{'='*80}")
        
        # Log the model info
        model_name = serialized.get("name", "Unknown")
        self.logger.debug(f"[MODEL] {model_name}")
        
        # Log all message conversations
        for i, message_list in enumerate(messages):
            self.logger.debug(f"\n[CHAT] CONVERSATION #{i+1}:")
            self.logger.debug(f"{'-'*40}")
            for msg in message_list:
                role = msg.__class__.__name__.replace("Message", "").lower()
                # Handle Unicode characters that can't be encoded in Windows console
                try:
                    # Replace problematic Unicode characters
                    content = msg.content.replace('→', '->').replace('←', '<-').replace('↓', 'v').replace('↑', '^')
                    # Additional safety: encode/decode to handle any remaining issues
                    content = content.encode('ascii', errors='replace').decode('ascii')
                    self.logger.debug(f"[{role.upper()}]: {content}")
                except Exception as e:
                    # Ultimate fallback: just log that there was content
                    self.logger.debug(f"[{role.upper()}]: <Content with encoding issues - {len(msg.content)} characters>")
            self.logger.debug(f"{'-'*40}")
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Log when a tool starts executing."""
        tool_name = serialized.get("name", "Unknown Tool")
        self.logger.debug(f"\n[TOOL] CALL: {tool_name}")
        # Handle Unicode in tool input
        try:
            safe_input = input_str.encode('utf-8', errors='replace').decode('utf-8')
            self.logger.debug(f"[INPUT] {safe_input}")
        except (UnicodeEncodeError, UnicodeDecodeError):
            safe_input = input_str.encode('ascii', errors='replace').decode('ascii')
            self.logger.debug(f"[INPUT] {safe_input}")
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Log when a tool finishes executing."""
        # Handle Unicode in tool output
        try:
            safe_output = output.encode('utf-8', errors='replace').decode('utf-8')
            self.logger.debug(f"[OUTPUT] {safe_output}")
        except (UnicodeEncodeError, UnicodeDecodeError):
            safe_output = output.encode('ascii', errors='replace').decode('ascii')
            self.logger.debug(f"[OUTPUT] {safe_output}")
        self.logger.debug(f"[COMPLETE] Tool completed\n")
    
    def on_tool_error(
        self, 
        error: Union[Exception, KeyboardInterrupt], 
        **kwargs: Any
    ) -> None:
        """Log when a tool encounters an error."""
        self.logger.error(f"[ERROR] Tool failed: {str(error)}\n")
    
    def on_agent_action(self, action, **kwargs: Any) -> None:
        """Log agent actions."""
        self.logger.debug(f"\n[AGENT] ACTION: {action.tool}")
        self.logger.debug(f"[INPUT] {action.tool_input}")
        if action.log:
            self.logger.debug(f"[REASONING] {action.log}")
    
    def on_agent_finish(self, finish, **kwargs: Any) -> None:
        """Log when agent finishes."""
        self.logger.debug(f"\n[AGENT] FINISHED")
        self.logger.debug(f"[OUTPUT] Final output: {finish.return_values}")