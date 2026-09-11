"""The records committed to knowledge/ and experiments/ must parse.

premerge_check.py calls these same loaders at merge time. A malformed record does not fail the
commit that wrote it — it fails the next person's merge, in a traceback nobody traces back here.
Every other research test builds synthetic records in a tmpdir, so nothing read the real ones.
"""

from inference_server.research.kb import (
    EXPERIMENTS_DIR,
    KNOWLEDGE_DIR,
    load_entries,
    load_experiments,
)


def test_every_knowledge_entry_parses():
    assert len(load_entries()) == len(list(KNOWLEDGE_DIR.glob("*.json")))


def test_every_experiment_parses():
    assert len(load_experiments()) == len(list(EXPERIMENTS_DIR.glob("*.json")))
