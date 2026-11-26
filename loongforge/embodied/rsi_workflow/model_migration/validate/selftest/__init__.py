"""Negative self-test: proves the judges reject what they are supposed to reject.

A gate that is never shown a broken case is indistinguishable from a gate that always passes.
Each case here declares the verdict it must receive; the self-test fails if a case marked FAIL
is let through (the gate leaks) or a case marked PASS is rejected (the gate over-rejects).
Nothing may be judged for real until this passes.
"""
