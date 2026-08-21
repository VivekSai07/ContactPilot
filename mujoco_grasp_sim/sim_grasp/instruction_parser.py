"""Instruction parser: turns a free-text pick-and-place instruction into
an ordered list of Steps via NVIDIA's OpenAI-compatible NIM endpoint
(meta/llama-3.1-8b-instruct -- the only bake-off candidate confirmed to
support schema-enforced JSON output, see
docs/research/2026-08-20-reasoning-layer-reflectvlm.md and
docs/research/bakeoff_instruction_parser.py for the empirical validation).

SYSTEM_PROMPT below is kept in sync BY HAND with
docs/research/bakeoff_instruction_parser.py's copy -- that script is
deliberately standalone/stdlib-only (no cgn_torch env needed to rerun the
bake-off), so it cannot import this module without pulling in
sim_grasp/__init__.py's full mujoco/cv2 import chain. Update both files
together if the parsing contract changes.
"""
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-8b-instruct"
VALID_RELATIONS = {"left_of", "right_of", "near", "center", "none"}

SYSTEM_PROMPT = """You are a robotic task planner. Translate a human's \
pick-and-place instruction into a structured execution sequence.

Output ONLY a JSON object, nothing else -- no chat filler, no markdown code \
fences, no explanation. The object must have exactly one key, "steps", whose \
value is a JSON array with one element per pick-and-place action described \
in the instruction (an instruction describing N objects to pick produces N \
elements, in the order they should be executed).

Each element of "steps" must have exactly these keys:
  "step": integer, 1-indexed
  "pick_target": string, a short description of the object to pick
  "place_relation": one of "left_of", "right_of", "near", "center", "none"
  "place_reference": string naming what the relation is relative to (e.g.
    "bin", or another object's description), or null if place_relation is
    "none"

If the instruction asks to stack one object on another, use place_relation
"near" with place_reference set to the object it should go near (stacking
is not supported; treat it as "put it next to that object" instead).

Example for a two-object instruction:
{"steps": [
  {"step": 1, "pick_target": "blue cube", "place_relation": "left_of", "place_reference": "bin"},
  {"step": 2, "pick_target": "red cube", "place_relation": "near", "place_reference": "blue cube"}
]}

Each distinct object mentioned in the instruction gets exactly ONE step, in
the order it should be picked. Never repeat the same pick_target in two
different steps -- if only one object is mentioned, output exactly one step,
even if the instruction also describes where to place it:
{"steps": [
  {"step": 1, "pick_target": "red cube", "place_relation": "left_of", "place_reference": "bin"}
]}
"""


@dataclass
class Step:
    step: int
    pick_target: str
    place_relation: str
    place_reference: 'str | None'


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE per line, '#' comments) -- avoids a
    python-dotenv dependency, copied from
    docs/research/bakeoff_instruction_parser.py for the same reason
    SYSTEM_PROMPT is copied rather than imported."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _parse_and_validate(raw: str) -> list[Step]:
    """Pure JSON-validation logic, no network access -- kept separate from
    parse_instruction() so it's testable without an API key."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f'model response is not valid JSON: {e}') from e
    if not isinstance(obj, dict):
        raise ValueError(f'expected a JSON object with a "steps" key, got '
                         f'{type(obj).__name__}')
    if 'steps' not in obj:
        raise ValueError('model response is missing the "steps" key')
    steps_raw = obj['steps']
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError('"steps" must be a non-empty JSON array')

    required_keys = {'step', 'pick_target', 'place_relation', 'place_reference'}
    steps = []
    for i, s in enumerate(steps_raw):
        if not isinstance(s, dict) or set(s.keys()) != required_keys:
            raise ValueError(f'step {i} has the wrong keys: {s!r} '
                             f'(expected exactly {sorted(required_keys)})')
        if not isinstance(s['step'], int):
            raise ValueError(f'step {i}: "step" must be an int, got '
                             f'{type(s["step"]).__name__}')
        if not isinstance(s['pick_target'], str):
            raise ValueError(f'step {i}: "pick_target" must be a string')
        if s['place_relation'] not in VALID_RELATIONS:
            raise ValueError(f'step {i}: "place_relation" {s["place_relation"]!r} '
                             f'not one of {sorted(VALID_RELATIONS)}')
        if s['place_reference'] is not None and not isinstance(s['place_reference'], str):
            raise ValueError(f'step {i}: "place_reference" must be a string or null')
        steps.append(Step(step=s['step'], pick_target=s['pick_target'],
                          place_relation=s['place_relation'],
                          place_reference=s['place_reference']))
    return steps


def parse_instruction(text: str, api_key: 'str | None' = None) -> list[Step]:
    """Parses a free-text instruction into an ordered Step list via the
    NIM API. Raises ValueError (bad schema) or RuntimeError (network/API
    failure) -- callers should let both propagate and abort the run, per
    this project's 'fail loudly, no silent fallback' decision for this
    call (see docs/superpowers/specs/2026-08-21-reasoning-layer-phase1-design.md)."""
    if api_key is None:
        _load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')
        api_key = os.environ.get('NVIDIA_API_KEY')
    if not api_key:
        raise RuntimeError('NVIDIA_API_KEY not found in the environment or .env -- '
                           'required for --instruction')

    payload = json.dumps({
        'model': MODEL,
        'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': text}],
        'temperature': 0.2,
        'max_tokens': 1024,
        'stream': False,
        'response_format': {'type': 'json_object'},
    }).encode('utf-8')
    req = urllib.request.Request(
        NIM_URL, data=payload, method='POST',
        headers={'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=90.0) as resp:
            body = json.loads(resp.read().decode('utf-8'))
    except urllib.error.URLError as e:
        raise RuntimeError(f'NIM API call failed: {e}') from e
    try:
        content = body['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f'unexpected NIM API response shape: {e}') from e
    return _parse_and_validate(content)
