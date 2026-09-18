"""Control plane: the seconds clock. Routing and the global prefix index, as pure policy.

A sibling of `inference_server`, not a subpackage: nothing here imports the engine, and CI runs
this package in the loop lane, where no torch is installed, so that stays true.
"""
