"""
Centralized configuration for the card scanner service.
Uses pydantic-settings for environment variable support.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Any
from datetime import time as dt_time


def parse_comma_list(value: Any) -> List[str]:
    """Parse comma-separated string into list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if not value.strip():
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def parse_int_list(value: Any, default: List[int] | None = None) -> List[int]:
    """Parse comma-separated integers into list."""
    if value is None:
        return default or []
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, str):
        if not value.strip():
            return default or []
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return default or []


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All settings can be overridden with CARD_SCANNER_ prefix.
    e.g., CARD_SCANNER_DATABASE_PATH, CARD_SCANNER_API_KEYS

    List fields (api_keys, cors_origins, default_categories) support
    comma-separated values: CARD_SCANNER_API_KEYS=key1,key2,key3
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CARD_SCANNER_",
        extra="ignore",
    )

    # Database settings
    database_path: str = "database.db"
    csv_path: str = "categories/"

    # Authentication - stored as string, parsed to list
    api_keys_str: str = Field(default="", validation_alias="CARD_SCANNER_API_KEYS")

    # CORS settings - stored as string, parsed to list
    cors_origins_str: str = Field(default="*", validation_alias="CARD_SCANNER_CORS_ORIGINS")

    # File upload limits
    max_file_size: int = 10 * 1024 * 1024  # 10MB

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # External service URLs
    tcg_csv_url: str = "https://tcgcsv.com/tcgplayer/"
    vectors_repo_url: str = "https://github.com/card-sorter/vectors.git"
    vectors_repo_path: str = "vectors"

    # Timeouts
    http_timeout: float = 30.0
    yolo_timeout: float = 30.0
    git_timeout: float = 120.0

    # Default categories - stored as string, parsed to list
    default_categories_str: str = Field(
        default="3", validation_alias="CARD_SCANNER_DEFAULT_CATEGORIES"
    )

    # Scheduled update times (hour, minute)
    db_update_hour: int = 3
    db_update_minute: int = 0
    vectors_update_hour: int = 4
    vectors_update_minute: int = 0

    # Concurrency limits
    max_concurrent_downloads: int = 5

    # Geometric verification (RANSAC re-rank) for search_verified()
    # Reference SIFT features ship in the vectors repository
    # (features/{category}/{product_id}.npz, written by regen_features.py)
    # and are LRU-cached in memory once loaded.
    rerank_candidates: int = 10
    rerank_lowe_ratio: float = 0.75
    rerank_ransac_reproj: float = 5.0
    ref_features_cache_entries: int = 2048

    # Retry settings
    retry_attempts: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0

    # Rate limiting (requests per minute)
    rate_limit_scan: int = 10
    rate_limit_identify: int = 30
    rate_limit_price: int = 60
    rate_limit_update: int = 1

    # Logging
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def api_keys(self) -> List[str]:
        """Get API keys as a list."""
        return parse_comma_list(self.api_keys_str)

    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins as a list."""
        result = parse_comma_list(self.cors_origins_str)
        return result if result else ["*"]

    @property
    def default_categories(self) -> List[int]:
        """Get default categories as a list of integers."""
        return parse_int_list(self.default_categories_str, [3])

    @property
    def db_update_time(self) -> dt_time:
        """Return the database update time as a time object."""
        return dt_time(hour=self.db_update_hour, minute=self.db_update_minute)

    @property
    def vectors_update_time(self) -> dt_time:
        """Return the vectors update time as a time object."""
        return dt_time(hour=self.vectors_update_hour, minute=self.vectors_update_minute)

    @property
    def auth_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return len(self.api_keys) > 0


# Global settings instance
settings = Settings()
