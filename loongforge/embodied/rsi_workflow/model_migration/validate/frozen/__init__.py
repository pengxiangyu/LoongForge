"""The copies that judge, and the hash lock over them.

`freeze.py` copies the judging closure out of `validate/` and records a sha256 per file in
`JUDGE_LOCK.json`. From then on every judgement first verifies the lock: if any frozen file has
changed, the judge refuses to produce a verdict instead of producing a lenient one. Widening a
limit is therefore not a thing a migration can do quietly — it shows up as a broken lock.

The closure is the two judges plus everything they import plus the threshold file:
structure_judge.py, align_gate.py (copied as align_gate_frozen.py), stats.py, thresholds.yaml.
"""
