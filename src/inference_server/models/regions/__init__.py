"""Overrides for stretches of the decode stations (`models/stations.py`). A module here may define
`OVERRIDES = {(first_station, last_station): fn}`: fn(ctx) gets the stretch's inputs and returns its writes, the K/V
pool written in place. Collected once, at the first decode forward."""
