"""
Centralized prompt management system for LLM interactions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from core.config import get_settings
from core.exceptions import ConfigurationError


logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages LLM prompts loaded from YAML configuration files.
    
    Provides centralized access to all system prompts with template
    variable substitution and validation.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the prompt manager.
        
        Args:
            config_path: Path to prompts configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Set default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        
        self.config_path = Path(config_path)
        self.prompts_config = {}
        self.load_prompts()
    
    def load_prompts(self):
        """
        Load prompts configuration from YAML file.
        
        Raises:
            ConfigurationError: If config file cannot be loaded
        """
        try:
            if not self.config_path.exists():
                raise ConfigurationError(f"Prompts config file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.prompts_config = yaml.safe_load(f)
            
            self.logger.info(f"Loaded prompts configuration from {self.config_path}")
            self._validate_config()
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in prompts config: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load prompts config: {str(e)}")
    
    def _validate_config(self):
        """
        Validate the loaded prompts configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_sections = ['agents', 'tools']
        
        for section in required_sections:
            if section not in self.prompts_config:
                raise ConfigurationError(f"Missing required section '{section}' in prompts config")
        
        # Validate agent prompts
        required_agents = ['anomaly_agent', 'enhanced_suggestion_agent']
        for agent in required_agents:
            if agent not in self.prompts_config['agents']:
                raise ConfigurationError(f"Missing agent '{agent}' in prompts config")
            
            if 'system_prompt' not in self.prompts_config['agents'][agent]:
                raise ConfigurationError(f"Missing system_prompt for agent '{agent}'")
        
        # Validate tool prompts
        required_tools = ['intelligent_insight_generator']
        for tool in required_tools:
            if tool not in self.prompts_config['tools']:
                raise ConfigurationError(f"Missing tool '{tool}' in prompts config")
            
            tool_config = self.prompts_config['tools'][tool]
            if 'system_prompt' not in tool_config:
                raise ConfigurationError(f"Missing system_prompt for tool '{tool}'")
            
            if 'human_prompt_template' not in tool_config:
                raise ConfigurationError(f"Missing human_prompt_template for tool '{tool}'")
        
        self.logger.debug("Prompts configuration validation passed")
    
    def get_agent_system_prompt(self, agent_name: str) -> str:
        """
        Get system prompt for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            str: System prompt text
            
        Raises:
            ConfigurationError: If agent or prompt not found
        """
        try:
            return self.prompts_config['agents'][agent_name]['system_prompt'].strip()
        except KeyError:
            raise ConfigurationError(f"System prompt not found for agent '{agent_name}'")
    
    def get_tool_system_prompt(self, tool_name: str) -> str:
        """
        Get system prompt for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            str: System prompt text
            
        Raises:
            ConfigurationError: If tool or prompt not found
        """
        try:
            return self.prompts_config['tools'][tool_name]['system_prompt'].strip()
        except KeyError:
            raise ConfigurationError(f"System prompt not found for tool '{tool_name}'")
    
    def get_tool_human_prompt(self, tool_name: str, variables: Dict[str, Any]) -> str:
        """
        Get human prompt template for a tool with variable substitution.
        
        Args:
            tool_name: Name of the tool
            variables: Dictionary of variables to substitute in template
            
        Returns:
            str: Human prompt with variables substituted
            
        Raises:
            ConfigurationError: If tool or prompt not found
            ValueError: If required variables are missing
        """
        try:
            template_str = self.prompts_config['tools'][tool_name]['human_prompt_template']
            
            # Use str.format for variable substitution (supports {variable} syntax)
            try:
                return template_str.format(**variables).strip()
            except KeyError as e:
                missing_var = str(e).strip("'")
                raise ValueError(f"Missing required variable '{missing_var}' for tool '{tool_name}'")
                
        except KeyError:
            raise ConfigurationError(f"Human prompt template not found for tool '{tool_name}'")
    
    def get_formatting_config(self) -> Dict[str, Any]:
        """
        Get formatting configuration for prompts.
        
        Returns:
            Dict[str, Any]: Formatting configuration
        """
        return self.prompts_config.get('formatting', {})
    
    def get_constraints_config(self) -> Dict[str, Any]:
        """
        Get constraints configuration for prompts.
        
        Returns:
            Dict[str, Any]: Constraints configuration
        """
        return self.prompts_config.get('constraints', {})
    
    def get_variables_config(self, category: str = 'common') -> list:
        """
        Get list of supported variables for a category.
        
        Args:
            category: Variable category ('common', 'financial', etc.)
            
        Returns:
            list: List of supported variable names
        """
        variables_config = self.prompts_config.get('variables', {})
        return variables_config.get(category, [])
    
    def validate_prompt_length(self, prompt: str, prompt_type: str = 'general') -> bool:
        """
        Validate prompt length against configured constraints.
        
        Args:
            prompt: Prompt text to validate
            prompt_type: Type of prompt for specific length limits
            
        Returns:
            bool: True if prompt length is valid
        """
        constraints = self.get_constraints_config()
        max_length = constraints.get('max_prompt_length', 8000)
        
        if len(prompt) > max_length:
            self.logger.warning(f"Prompt length ({len(prompt)}) exceeds maximum ({max_length})")
            return False
        
        return True
    
    def format_confidence_score(self, score: Union[int, float]) -> str:
        """
        Format confidence score according to configuration.
        
        Args:
            score: Confidence score to format
            
        Returns:
            str: Formatted confidence score
        """
        constraints = self.get_constraints_config()
        min_conf, max_conf = constraints.get('confidence_range', [1, 10])
        
        # Clamp score to valid range
        score = max(min_conf, min(max_conf, score))
        
        formatting = self.get_formatting_config()
        scale_info = formatting.get('confidence_scale', '1-10')
        
        return f"{score} (scale: {scale_info})"
    
    def get_valid_risk_levels(self) -> list:
        """
        Get list of valid risk levels.
        
        Returns:
            list: Valid risk level values
        """
        constraints = self.get_constraints_config()
        return constraints.get('supported_risk_levels', ['low', 'medium', 'high', 'critical'])
    
    def get_valid_signal_types(self) -> list:
        """
        Get list of valid trading signal types.
        
        Returns:
            list: Valid signal type values
        """
        formatting = self.get_formatting_config()
        return formatting.get('signal_types', ['buy', 'sell', 'hold'])
    
    def reload_prompts(self):
        """
        Reload prompts configuration from file.
        
        Useful for development or configuration updates without restart.
        """
        self.logger.info("Reloading prompts configuration")
        self.load_prompts()
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded configuration.
        
        Returns:
            Dict[str, Any]: Configuration metadata
        """
        return {
            'config_path': str(self.config_path),
            'version': self.prompts_config.get('version', 'unknown'),
            'last_updated': self.prompts_config.get('last_updated', 'unknown'),
            'description': self.prompts_config.get('description', ''),
            'agents_count': len(self.prompts_config.get('agents', {})),
            'tools_count': len(self.prompts_config.get('tools', {}))
        }


# Global prompt manager instance
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """
    Get the global prompt manager instance.
    
    Returns:
        PromptManager: Global prompt manager
    """
    global _prompt_manager
    
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    
    return _prompt_manager


def reload_prompts():
    """
    Reload prompts configuration in the global manager.
    """
    global _prompt_manager
    
    if _prompt_manager is not None:
        _prompt_manager.reload_prompts()
    else:
        _prompt_manager = PromptManager()