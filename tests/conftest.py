"""Test fixtures: an isolated database, deterministic settings, and an HTTP client.

Every test module gets its own SQLite file under ``var/test`` so tests never share state
and can run in any order. Settings are re-read per session via ``reload_settings()``.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fresh_env(tmp_root: Path) -> None:
    os.environ.setdefault("PF_ENV", "test")
    os.environ["PF_DATA_DIR"] = str(tmp_root)
    os.environ["PF_ARTIFACTS_DIR"] = str(tmp_root / "artifacts")
    os.environ["PF_DATABASE_URL"] = ""
    os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{tmp_root / 'test.db'}"
    os.environ["PF_SIGNING_KEY"] = "test-signing-key"
    os.environ["PF_REQUIRE_HUMAN_APPROVAL"] = "true"
    os.environ["PF_AUTO_PUBLISH_ENABLED"] = "false"
    os.environ["PF_BUDGET_ENFORCED"] = "true"
    os.environ["PF_BUDGET_DAILY_LLM_MICROS"] = str(2_000_000)  # A$2/day for tests
    os.environ["PF_BUDGET_WEEKLY_LLM_MICROS"] = str(5_000_000)
    os.environ["PF_RATE_LIMIT_PER_MINUTE"] = "10000"
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(key, None)
    os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("promptfoundry")
    _fresh_env(root)
    from core.config import reload_settings

    settings = reload_settings()
    settings.ensure_dirs()
    from db.session import init_db

    init_db()
    return root


@pytest.fixture()
def session(data_dir: Path) -> Iterator[object]:
    """A transactional-ish session with a clean schema per test."""
    from db.base import Base
    from db.session import get_engine, session_scope

    Base.metadata.drop_all(bind=get_engine())
    Base.metadata.create_all(bind=get_engine())
    with session_scope() as db_session:
        yield db_session


@pytest.fixture()
def client(data_dir: Path) -> Iterator[object]:
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def admin_token(client: object) -> str:
    email = f"admin-{uuid.uuid4().hex[:8]}@example.com"
    response = client.post(  # type: ignore[attr-defined]
        "/auth/register",
        json={"email": email, "password": "supersecret123", "display_name": "Test Admin"},
    )
    assert response.status_code == 201, response.text
    return response.json()["token"]


@pytest.fixture()
def auth(admin_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {admin_token}"}
