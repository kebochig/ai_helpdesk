"""
Configuration settings for the AI Help Desk system.
"""
import os
from typing import Optional
from dotenv import load_dotenv
from src.models.schemas import SystemConfig

# Load environment variables
load_dotenv()


class Settings:
    """Application settings and configuration."""
    
    def __init__(self):
        self.system_config = SystemConfig()
        self._load_env_variables()
    
    def _load_env_variables(self):
        """Load environment variables."""
        self.google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        
        # Vector store settings
        self.system_config.vector_store.persist_directory = os.getenv(
            "VECTOR_STORE_PATH", "./data/vector_store"
        )
        
        # LLM settings
        self.system_config.llm.model_name = os.getenv(
            "GEMINI_MODEL", "gemini-1.5-flash"
        )
        self.system_config.llm.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.system_config.llm.max_tokens = int(os.getenv("MAX_TOKENS", "2000"))
        
        # Knowledge base path
        self.system_config.knowledge_base_path = os.getenv(
            "KNOWLEDGE_BASE_PATH", "./data/knowledge_base"
        )
        
        # Thresholds
        self.system_config.escalation_threshold = float(
            os.getenv("ESCALATION_THRESHOLD", "0.7")
        )
        self.system_config.min_confidence_threshold = float(
            os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.5")
        )
    
    def validate_settings(self) -> bool:
        """Validate that all required settings are present."""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required but not set")
        
        if not os.path.exists(self.system_config.knowledge_base_path):
            os.makedirs(self.system_config.knowledge_base_path, exist_ok=True)
        
        if not os.path.exists(self.system_config.vector_store.persist_directory):
            os.makedirs(self.system_config.vector_store.persist_directory, exist_ok=True)
        
        return True


# Global settings instance
settings = Settings()

# Category-specific escalation rules
ESCALATION_RULES = {
    "security_incident": {
        "auto_escalate": True,
        "priority": "critical",
        "department": "security_team",
        "response_time": "immediate"
    },
    "hardware_failure": {
        "auto_escalate": False,
        "priority": "medium",
        "department": "it_support",
        "response_time": "4_hours"
    },
    "password_reset": {
        "auto_escalate": False,
        "priority": "low",
        "department": "it_support",
        "response_time": "24_hours"
    },
    "network_connectivity": {
        "auto_escalate": False,
        "priority": "high",
        "department": "network_team",
        "response_time": "2_hours"
    },
    "software_installation": {
        "auto_escalate": False,
        "priority": "medium",
        "department": "it_support",
        "response_time": "8_hours"
    },
    "email_configuration": {
        "auto_escalate": False,
        "priority": "medium",
        "department": "it_support",
        "response_time": "8_hours"
    },
    "policy_question": {
        "auto_escalate": False,
        "priority": "low",
        "department": "hr_department",
        "response_time": "24_hours"
    }
}

# Classification prompts and examples
CLASSIFICATION_EXAMPLES = {
    "password_reset": [
        "I forgot my password",
        "Can't log into my account",
        "Password expired",
        "Account locked out",
        "Reset password"
    ],
    "software_installation": [
        "Install new software",
        "Need Microsoft Office",
        "Can't install application",
        "Software license",
        "Download program"
    ],
    "hardware_failure": [
        "Computer won't start",
        "Printer not working",
        "Monitor issues",
        "Keyboard broken",
        "Hardware malfunction"
    ],
    "network_connectivity": [
        "Internet not working",
        "Can't connect to WiFi",
        "Network down",
        "VPN issues",
        "Connection problems"
    ],
    "email_configuration": [
        "Email not working",
        "Outlook setup",
        "Can't send emails",
        "Email server settings",
        "Mail configuration"
    ],
    "security_incident": [
        "Suspicious email",
        "Malware detected",
        "Security breach",
        "Phishing attempt",
        "Unauthorized access"
    ],
    "policy_question": [
        "Company policy",
        "IT guidelines",
        "What's allowed",
        "Procedure question",
        "Compliance issue"
    ]
}