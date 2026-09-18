"""Control plane: the seconds clock. Routing and the global prefix index, as pure policy.

Deliberately a sibling of `inference_server`, not a subpackage: the engine knows nothing about
it, and nothing here imports the engine (which needs torch). `tests/test_control_plane.py`
asserts that isolation the way the CI loop lane asserts it for `research/`.
"""
