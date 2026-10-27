"""Central, environment-driven configuration for PromptFoundry.

Everything is configurable through environment variables (see ``.env.example``).
Defaults are deliberately safe: no network access, no publishing, mock providers,
and a hard weekly spend cap well inside the initial AUD 50-150/week budget.

Money is stored as integer **AUD micros** (1 AUD = 1_000_000 micros) everywhere so
that accounting never suffers floating point drift.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
AUD_MICROS = 1_000_000


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE, # comments) so the MVP needs no extra dep."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = _env(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    """Immutable-ish runtime configuration (call :func:`reload_settings` in tests)."""

    app_name: str = "PromptFoundry"
    version: str = "0.1.0"
    env: str = "local"
    log_level: str = "INFO"
    jurisdiction: str = "AU"
    data_region: str = "ap-southeast-2"

    # --- storage -----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "var")
    database_url: str = ""
    artifacts_dir: Path = field(default_factory=lambda: REPO_ROOT / "var" / "artifacts")
    demo_dir: Path = field(default_factory=lambda: REPO_ROOT / "demo")
    permitted_corpora_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "demo" / "data" / "corpora"
    )

    # --- queue -------------------------------------------------------------
    redis_url: str = ""
    celery_broker_url: str = ""
    celery_task_always_eager: bool = True

    # --- auth --------------------------------------------------------------
    session_ttl_hours: int = 12
    bootstrap_admin_email: str = ""
    bootstrap_admin_password: str = ""
    allow_self_registration: bool = True
    rate_limit_per_minute: int = 240

    # --- embeddings / ML ---------------------------------------------------
    embedding_provider: str = ""  # "" => auto (openai if key else hashing)
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 256
    cluster_min_size: int = 3
    cluster_eps: float = 0.34
    drift_threshold: float = 0.35

    # --- LLM providers -----------------------------------------------------
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    default_model_tier: str = "balanced"  # cheap | balanced | frontier
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # --- budgets (integer AUD micros) --------------------------------------
    aud_per_usd: float = 1.55
    budget_daily_llm_micros: int = 6 * AUD_MICROS
    budget_weekly_llm_micros: int = 30 * AUD_MICROS
    budget_weekly_infra_micros: int = 5 * AUD_MICROS
    budget_alert_ratio: float = 0.8
    budget_enforced: bool = True

    # --- governance --------------------------------------------------------
    require_human_approval: bool = True
    auto_publish_enabled: bool = False
    auto_publish_max_risk: float = 0.25
    ip_block_threshold: float = 0.92
    ip_warn_threshold: float = 0.80
    ip_shingle_overlap_threshold: float = 0.35
    signing_key: str = "dev-only-signing-key-change-me"
    require_signatures: bool = True

    # --- publishing --------------------------------------------------------
    storefront_adapter: str = "mock"  # mock | gumroad | lemon_squeezy
    gumroad_access_token: str = ""
    lemon_squeezy_api_key: str = ""
    lemon_squeezy_store_id: str = ""
    default_currency: str = "AUD"

    # --- ingestion ---------------------------------------------------------
    max_upload_rows: int = 50_000
    connector_timeout_seconds: float = 30.0

    # --- misc --------------------------------------------------------------
    cors_origins: list[str] = field(default_factory=list)
    max_page_size: int = 200

    def __post_init__(self) -> None:
        _load_dotenv(REPO_ROOT / ".env")
        self.env = _env("PF_ENV", self.env)
        self.log_level = _env("PF_LOG_LEVEL", self.log_level).upper()
        self.data_dir = Path(_env("PF_DATA_DIR", str(self.data_dir))).expanduser()
        self.artifacts_dir = Path(
            _env("PF_ARTIFACTS_DIR", str(self.data_dir / "artifacts"))
        ).expanduser()
        self.demo_dir = Path(_env("PF_DEMO_DIR", str(REPO_ROOT / "demo"))).expanduser()
        self.permitted_corpora_dir = Path(
            _env("PF_PERMITTED_CORPORA_DIR", str(self.demo_dir / "data" / "corpora"))
        ).expanduser()
        self.database_url = _env(
            "DATABASE_URL", f"sqlite+pysqlite:///{self.data_dir / 'promptfoundry.db'}"
        )
        self.redis_url = _env("REDIS_URL", "")
        self.celery_broker_url = _env("CELERY_BROKER_URL", self.redis_url)
        self.celery_task_always_eager = _env_bool(
            "CELERY_TASK_ALWAYS_EAGER", not bool(self.celery_broker_url)
        )
        self.session_ttl_hours = _env_int("PF_SESSION_TTL_HOURS", self.session_ttl_hours)
        self.bootstrap_admin_email = _env("PF_ADMIN_EMAIL", "")
        self.bootstrap_admin_password = _env("PF_ADMIN_PASSWORD", "")
        self.allow_self_registration = _env_bool(
            "PF_ALLOW_SELF_REGISTRATION", self.allow_self_registration
        )
        self.rate_limit_per_minute = _env_int(
            "PF_RATE_LIMIT_PER_MINUTE", self.rate_limit_per_minute
        )
        self.embedding_provider = _env("PF_EMBEDDING_PROVIDER", "")
        self.embedding_model = _env("PF_EMBEDDING_MODEL", self.embedding_model)
        self.embedding_dim = _env_int("PF_EMBEDDING_DIM", self.embedding_dim)
        self.cluster_min_size = _env_int("PF_CLUSTER_MIN_SIZE", self.cluster_min_size)
        self.cluster_eps = _env_float("PF_CLUSTER_EPS", self.cluster_eps)
        self.drift_threshold = _env_float("PF_DRIFT_THRESHOLD", self.drift_threshold)
        self.openai_api_key = _env("OPENAI_API_KEY", "")
        self.anthropic_api_key = _env("ANTHROPIC_API_KEY", "")
        self.google_api_key = _env("GOOGLE_API_KEY", "") or _env("GEMINI_API_KEY", "")
        self.default_model_tier = _env("PF_DEFAULT_MODEL_TIER", self.default_model_tier)
        self.llm_timeout_seconds = _env_float("PF_LLM_TIMEOUT_SECONDS", self.llm_timeout_seconds)
        self.llm_max_retries = _env_int("PF_LLM_MAX_RETRIES", self.llm_max_retries)
        self.aud_per_usd = _env_float("PF_AUD_PER_USD", self.aud_per_usd)
        self.budget_daily_llm_micros = _env_int(
            "PF_BUDGET_DAILY_LLM_MICROS", self.budget_daily_llm_micros
        )
        self.budget_weekly_llm_micros = _env_int(
            "PF_BUDGET_WEEKLY_LLM_MICROS", self.budget_weekly_llm_micros
        )
        self.budget_weekly_infra_micros = _env_int(
            "PF_BUDGET_WEEKLY_INFRA_MICROS", self.budget_weekly_infra_micros
        )
        self.budget_enforced = _env_bool("PF_BUDGET_ENFORCED", self.budget_enforced)
        self.require_human_approval = _env_bool(
            "PF_REQUIRE_HUMAN_APPROVAL", self.require_human_approval
        )
        self.auto_publish_enabled = _env_bool("PF_AUTO_PUBLISH_ENABLED", self.auto_publish_enabled)
        self.auto_publish_max_risk = _env_float(
            "PF_AUTO_PUBLISH_MAX_RISK", self.auto_publish_max_risk
        )
        self.ip_block_threshold = _env_float("PF_IP_BLOCK_THRESHOLD", self.ip_block_threshold)
        self.ip_warn_threshold = _env_float("PF_IP_WARN_THRESHOLD", self.ip_warn_threshold)
        self.signing_key = _env("PF_SIGNING_KEY", self.signing_key)
        self.storefront_adapter = _env("PF_STOREFRONT_ADAPTER", self.storefront_adapter)
        self.gumroad_access_token = _env("GUMROAD_ACCESS_TOKEN", "")
        self.lemon_squeezy_api_key = _env("LEMONSQUEEZY_API_KEY", "")
        self.lemon_squeezy_store_id = _env("LEMONSQUEEZY_STORE_ID", "")
        self.default_currency = _env("PF_DEFAULT_CURRENCY", self.default_currency)
        self.max_upload_rows = _env_int("PF_MAX_UPLOAD_ROWS", self.max_upload_rows)
        self.cors_origins = _env_list("PF_CORS_ORIGINS", self.cors_origins)
        self.max_page_size = _env_int("PF_MAX_PAGE_SIZE", self.max_page_size)

    # ------------------------------------------------------------------ utils
    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"prod", "production"}

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key or self.anthropic_api_key or self.google_api_key)

    def any(self) -> dict[str, Any]:
        """Public, secret-free view used by /healthz and the demo banner."""
        return {
            "app": self.app_name,
            "version": self.version,
            "env": self.env,
            "jurisdiction": self.jurisdiction,
            "database": self.database_url.split("://")[0],
            "embedding_provider": self.embedding_provider or "auto",
            "llm_configured": self.llm_configured,
            "require_human_approval": self.require_human_approval,
            "auto_publish_enabled": self.auto_publish_enabled,
            "storefront_adapter": self.storefront_adapter,
            "budget_weekly_llm_aud": round(self.budget_weekly_llm_micros / AUD_MICROS, 2),
            "budget_daily_llm_aud": round(self.budget_daily_llm_micros / AUD_MICROS, 2),
        }

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()


def reload_settings() -> Settings:
    """Re-read the environment (used by tests and by ``make demo`` with a temp DB)."""
    global settings
    settings.__init__()  # type: ignore[misc]
    settings.ensure_dirs()
    return settings


def get_settings() -> Settings:
    return settings
