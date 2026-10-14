"""Launch profiles: [general] always, the ENGINE_PROFILE table over it, written into the environment."""

import pytest

from inference_server import profiles

TABLE = {"general": {"MAX_BATCH_SIZE": 64, "PREFILL_CHUNK_SIZE": "512"},
         "single_stream": {"MAX_BATCH_SIZE": 1}}


def test_general_applies_without_a_profile():
    assert profiles.resolve(None, TABLE) == {"MAX_BATCH_SIZE": "64", "PREFILL_CHUNK_SIZE": "512"}


def test_a_profile_overrides_general():
    assert profiles.resolve("single_stream", TABLE) == {"MAX_BATCH_SIZE": "1", "PREFILL_CHUNK_SIZE": "512"}


def test_an_unknown_profile_is_an_error():
    with pytest.raises(ValueError, match="no \\[saturated\\] table"):
        profiles.resolve("saturated", TABLE)


def test_apply_overrides_the_launch_env(tmp_path):
    f = tmp_path / "profiles.toml"
    f.write_text('[general]\nMAX_BATCH_SIZE = 64\n[single_stream]\nMAX_BATCH_SIZE = 1\n')
    env = {"MAX_BATCH_SIZE": "32", profiles.ENV: "single_stream"}
    assert profiles.apply(env, f) == {"MAX_BATCH_SIZE": "1"} and env["MAX_BATCH_SIZE"] == "1"


def test_the_shipped_file_parses():
    assert "general" in profiles.load()
