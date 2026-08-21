"""Bake-off: compare candidate LLMs for the Phase 1 instruction-parsing task
(docs/research/2026-08-20-reasoning-layer-reflectvlm.md, ROADMAP.md P8).

Sends the same set of test instructions to each candidate model via NVIDIA's
OpenAI-compatible NIM endpoint, with the exact system prompt the real
instruction parser would use, and scores each response for:
  - valid JSON (parses cleanly, no chat filler)
  - schema-correct (right keys/types)
  - semantically correct (matches the expected parse for that instruction,
    checked by a human reading the printed output -- this script does not
    attempt automatic semantic scoring)

Standalone, stdlib-only (urllib + json) -- no new dependency in any conda
env. Reads NVIDIA_API_KEY from a local .env file (gitignored) or the
environment.

Usage:
    python docs/research/bakeoff_instruction_parser.py
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# meta/llama-3.1-8b-instruct is the only candidate confirmed (via its NIM
# model card's "Capabilities" section) to natively support "Structured
# Output" (schema-enforced JSON) -- mistral-nemotron, nemotron-3-nano-30b-a3b,
# llama-3.3-70b-instruct, and llama-3.1-nemotron-nano-8b-v1 all confirmed
# "Structured Output: Not supported". Bake-off round 1 confirmed this in
# practice: llama-3.1-8b-instruct was 8/8 valid JSON+schema at ~0.7s avg
# latency; the others had real reliability issues (500/503 errors, 20-124s
# latency, one leaked reasoning trace instead of JSON). Round 2 below
# re-tests only the 2 instructions that exposed a step-duplication quirk in
# round 1, against the refined prompt.
CANDIDATES = [
    "meta/llama-3.1-8b-instruct",
]

# Models confirmed to support response_format={"type": "json_object"}
JSON_MODE_MODELS = {"meta/llama-3.1-8b-instruct"}

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

TEST_INSTRUCTIONS = [
    "Pick up the red cube and place it in the bin.",
    "Pick the blue cube first and put it on the left, then the red one on the right.",
    "Grab the tall cuboid and put it on the far left side.",
    "Pick the blue cube first and put it on the left, then pick the red one and place it near the blue cube.",
    "Stack the small cube on top of the big one.",
    "Put the green block in the center of the bin.",
    "Pick up all three cubes, placing the smallest one first, in the center.",
    "Place the yellow cuboid to the right of the bin.",
]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE per line, '#' comments) -- avoids a
    python-dotenv dependency for this one-off script."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def call_model(model: str, instruction: str, api_key: str, timeout: float = 90.0,
              retries: int = 2) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": False,
    }
    if model in JSON_MODE_MODELS:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        NIM_URL, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    last_err = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload_resp = json.loads(resp.read().decode("utf-8"))
            return payload_resp["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            print(f"    (retry {attempt + 1}/{retries} after {type(e).__name__}: {e})")
    raise last_err


def extract_json(raw: str):
    """Strip markdown code fences if the model added them anyway, then
    parse. Returns (parsed_value_or_None, error_or_None)."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


REQUIRED_KEYS = {"step", "pick_target", "place_relation", "place_reference"}
VALID_RELATIONS = {"left_of", "right_of", "near", "center", "none"}


def schema_errors(parsed) -> list:
    if not isinstance(parsed, dict) or "steps" not in parsed:
        return ['top-level value is not a JSON object with a "steps" key']
    steps = parsed["steps"]
    if not isinstance(steps, list) or not steps:
        return ['"steps" is not a non-empty JSON array']
    errors = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"step {i}: not an object")
            continue
        missing = REQUIRED_KEYS - step.keys()
        if missing:
            errors.append(f"step {i}: missing keys {missing}")
        rel = step.get("place_relation")
        if rel not in VALID_RELATIONS:
            errors.append(f"step {i}: place_relation={rel!r} not in {VALID_RELATIONS}")
    return errors


def main():
    _load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY not found in the environment or .env -- "
            "see .env.example for setup instructions.")

    results = {model: {"valid_json": 0, "valid_schema": 0, "errors": [], "latency_s": []}
               for model in CANDIDATES}

    for model in CANDIDATES:
        print(f"\n{'=' * 70}\n{model}\n{'=' * 70}")
        for instruction in TEST_INSTRUCTIONS:
            t0 = time.time()
            try:
                raw = call_model(model, instruction, api_key)
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TimeoutError) as e:
                print(f"  [API ERROR] {instruction!r}: {e}")
                results[model]["errors"].append(f"{instruction!r}: API error {e}")
                continue
            dt = time.time() - t0
            results[model]["latency_s"].append(dt)

            parsed, err = extract_json(raw)
            if err is not None:
                print(f"  [BAD JSON, {dt:.1f}s] {instruction!r}\n    raw: {raw[:200]!r}\n    error: {err}")
                results[model]["errors"].append(f"{instruction!r}: invalid JSON ({err})")
                continue
            results[model]["valid_json"] += 1

            errs = schema_errors(parsed)
            if errs:
                print(f"  [SCHEMA ERR, {dt:.1f}s] {instruction!r}\n    parsed: {parsed}\n    errors: {errs}")
                results[model]["errors"].append(f"{instruction!r}: schema errors {errs}")
                continue
            results[model]["valid_schema"] += 1
            print(f"  [OK, {dt:.1f}s] {instruction!r}\n    -> {json.dumps(parsed)}")

    n = len(TEST_INSTRUCTIONS)
    print(f"\n{'=' * 70}\nSUMMARY ({n} test instructions each)\n{'=' * 70}")
    for model, r in results.items():
        avg_latency = sum(r["latency_s"]) / len(r["latency_s"]) if r["latency_s"] else float("nan")
        print(f"{model}: valid_json={r['valid_json']}/{n}  valid_schema={r['valid_schema']}/{n}  "
              f"avg_latency={avg_latency:.2f}s  errors={len(r['errors'])}")


if __name__ == "__main__":
    main()
