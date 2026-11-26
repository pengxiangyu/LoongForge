"""Checkpoint key and layout conversion, kept in one place.

Model code must not know that a weight used to be called something else. Once a rename lives in a
module's `load_state_dict`, the same-weights precondition of F5 stops meaning anything: the two
paths load what looks like one checkpoint and end up with different tensors in different places.
So every rename and every transpose is declared in `weight_map.yaml` and applied here, and the
report this stage writes is what the structure judge checks for coverage.
"""
