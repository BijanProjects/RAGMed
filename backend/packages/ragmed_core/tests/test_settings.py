import pytest
from pydantic import ValidationError

from ragmed_core.settings import CoreSettings, Environment


def test_defaults_are_local_and_human_readable() -> None:
    settings = CoreSettings(_env_file=None)

    assert settings.environment is Environment.LOCAL
    assert settings.log_level == "INFO"
    assert settings.use_json_logs is False


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGMED_ENVIRONMENT", "production")
    monkeypatch.setenv("RAGMED_SERVICE_NAME", "diagnosis-agent")
    monkeypatch.setenv("RAGMED_LOG_LEVEL", "DEBUG")

    settings = CoreSettings(_env_file=None)

    assert settings.environment is Environment.PRODUCTION
    assert settings.service_name == "diagnosis-agent"
    assert settings.log_level == "DEBUG"
    assert settings.use_json_logs is True


def test_explicit_log_json_wins_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGMED_ENVIRONMENT", "production")
    monkeypatch.setenv("RAGMED_LOG_JSON", "false")

    assert CoreSettings(_env_file=None).use_json_logs is False


def test_rejects_unknown_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGMED_LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValidationError):
        CoreSettings(_env_file=None)


def test_settings_are_immutable() -> None:
    settings = CoreSettings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.service_name = "other"  # type: ignore[misc]
