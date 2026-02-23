"""
K-IFRS 1019 DBO Validation System - Configuration Management
============================================================
Environment variable management using Pydantic Settings

Usage:
    from config import settings
    print(settings.SUPABASE_URL)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables
    """

    # ==========================================================================
    # Supabase Configuration
    # ==========================================================================
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""  # anon/public key
    SUPABASE_SERVICE_KEY: Optional[str] = None  # for admin operations
    SUPABASE_STORAGE_BUCKET: str = "rule-files"

    # ==========================================================================
    # Application Settings
    # ==========================================================================
    APP_NAME: str = "K-IFRS 1019 DBO Validator"
    APP_VERSION: str = "2.0.0"

    # ==========================================================================
    # AI Configuration
    # ==========================================================================
    AI_MODEL_VERSION: str = "local-parser"
    ANTHROPIC_API_KEY: Optional[str] = None

    # ==========================================================================
    # Feature Flags
    # ==========================================================================
    ENABLE_AI_CACHING: bool = True
    ENABLE_LEARNING_DATA: bool = True

    # ==========================================================================
    # Pydantic Settings Configuration
    # ==========================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra fields in .env
    )

    def is_supabase_configured(self) -> bool:
        """Check if Supabase is properly configured"""
        return bool(self.SUPABASE_URL and self.SUPABASE_KEY)

    def is_ai_enabled(self) -> bool:
        """Check if AI features are enabled"""
        return bool(self.ANTHROPIC_API_KEY)


# =============================================================================
# Global Settings Instance
# =============================================================================
settings = Settings()


# =============================================================================
# Validation on Import
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Configuration Status")
    print("=" * 70)
    print(f"App Name: {settings.APP_NAME}")
    print(f"App Version: {settings.APP_VERSION}")
    print(f"Supabase Configured: {settings.is_supabase_configured()}")
    print(f"Supabase URL: {settings.SUPABASE_URL[:50]}..." if settings.SUPABASE_URL else "Not set")
    print(f"AI Caching Enabled: {settings.ENABLE_AI_CACHING}")
    print(f"Learning Data Enabled: {settings.ENABLE_LEARNING_DATA}")
    print(f"AI Model: {settings.AI_MODEL_VERSION}")
    print("=" * 70)
