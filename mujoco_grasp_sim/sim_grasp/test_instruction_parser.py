"""Standalone checks for instruction_parser.py -- run directly, no pytest
(this codebase has no automated test suite). No live NIM API calls here:
only the pure JSON-validation logic is exercised."""
import json

from sim_grasp.instruction_parser import Step, _parse_and_validate

VALID_TWO_STEP = json.dumps({"steps": [
    {"step": 1, "pick_target": "blue cube", "place_relation": "left_of", "place_reference": "bin"},
    {"step": 2, "pick_target": "red cube", "place_relation": "near", "place_reference": "blue cube"},
]})

steps = _parse_and_validate(VALID_TWO_STEP)
assert len(steps) == 2
assert steps[0] == Step(step=1, pick_target="blue cube",
                        place_relation="left_of", place_reference="bin")
assert steps[1] == Step(step=2, pick_target="red cube",
                        place_relation="near", place_reference="blue cube")

VALID_NONE_RELATION = json.dumps({"steps": [
    {"step": 1, "pick_target": "yellow block", "place_relation": "none", "place_reference": None},
]})
steps = _parse_and_validate(VALID_NONE_RELATION)
assert steps == [Step(step=1, pick_target="yellow block",
                      place_relation="none", place_reference=None)]

# Not valid JSON at all.
try:
    _parse_and_validate("not json at all")
    assert False, "expected ValueError for unparseable JSON"
except ValueError:
    pass

# Top-level array instead of {"steps": [...]} (the exact quirk the bake-off
# notes hit with response_format=json_object -- must be rejected, not
# silently misinterpreted).
try:
    _parse_and_validate(json.dumps([{"step": 1, "pick_target": "x",
                                     "place_relation": "none", "place_reference": None}]))
    assert False, "expected ValueError for a top-level array"
except ValueError:
    pass

# Missing the "steps" key.
try:
    _parse_and_validate(json.dumps({"wrong_key": []}))
    assert False, "expected ValueError for a missing 'steps' key"
except ValueError:
    pass

# Empty steps array.
try:
    _parse_and_validate(json.dumps({"steps": []}))
    assert False, "expected ValueError for an empty steps array"
except ValueError:
    pass

# A step missing a required key.
try:
    _parse_and_validate(json.dumps({"steps": [
        {"step": 1, "pick_target": "x", "place_relation": "none"}]}))  # no place_reference
    assert False, "expected ValueError for a step missing place_reference"
except ValueError:
    pass

# A step with an extra, unexpected key.
try:
    _parse_and_validate(json.dumps({"steps": [
        {"step": 1, "pick_target": "x", "place_relation": "none",
         "place_reference": None, "extra_key": "unexpected"}]}))
    assert False, "expected ValueError for a step with an extra key"
except ValueError:
    pass

# Wrong type for "pick_target" (must be a string).
try:
    _parse_and_validate(json.dumps({"steps": [
        {"step": 1, "pick_target": 123, "place_relation": "none",
         "place_reference": None}]}))
    assert False, "expected ValueError for a non-string pick_target"
except ValueError:
    pass

# Wrong type for "place_reference" (must be a string or null, not a number).
try:
    _parse_and_validate(json.dumps({"steps": [
        {"step": 1, "pick_target": "x", "place_relation": "near",
         "place_reference": 42}]}))
    assert False, "expected ValueError for a non-string/non-null place_reference"
except ValueError:
    pass

# An invalid place_relation value.
try:
    _parse_and_validate(json.dumps({"steps": [
        {"step": 1, "pick_target": "x", "place_relation": "on_top_of",
         "place_reference": "bin"}]}))
    assert False, "expected ValueError for an invalid place_relation"
except ValueError:
    pass

# Wrong type for "step" (must be an int, not a string).
try:
    _parse_and_validate(json.dumps({"steps": [
        {"step": "1", "pick_target": "x", "place_relation": "none",
         "place_reference": None}]}))
    assert False, "expected ValueError for a non-int step"
except ValueError:
    pass

print('All instruction_parser checks passed.')
