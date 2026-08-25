from cga.config import Settings


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.langsmith_project == "clinical-guideline-assistant"
    assert s.vector_store_dir == "./data/processed/chroma"
