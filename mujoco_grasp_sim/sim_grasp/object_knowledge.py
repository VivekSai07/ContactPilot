"""[P10 SP2] Object identity + commonsense category from hosted NIM:
identify(crop) names the object, categorize(name) maps it onto the bin
categories (schema-enforced enum, so the model can never invent a bin).
Fails loudly, like instruction_parser (no silent fallback to sim truth)."""
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from sim_grasp.instruction_parser import _load_dotenv

NIM_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
MODEL = 'meta/llama-3.2-11b-vision-instruct'   # 2026-10-03 model test: 10/12 on 3x masked crops
_SLEEP = time.sleep
_BACKOFF = (2.0, 4.0)

IDENTIFY_PROMPT = ('This is a cropped photo of one household product on a table. '
                   'Name the product in a short phrase (e.g. "box of chocolate candy bars"). '
                   'Answer with the phrase only.')


def _nim_transport(body: dict) -> str:
    if not os.environ.get('NVIDIA_API_KEY'):
        _load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')
    key = os.environ.get('NVIDIA_API_KEY')
    if not key:
        raise RuntimeError('NVIDIA_API_KEY not found in the environment or .env -- '
                           'required for --scene-graph')
    req = urllib.request.Request(NIM_URL, data=json.dumps(body).encode(), method='POST',
                                 headers={'Authorization': f'Bearer {key}',
                                          'Content-Type': 'application/json',
                                          'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=90.0) as r:
        return json.loads(r.read())['choices'][0]['message']['content']


def _call(body: dict, transport) -> str:
    transport = transport or _nim_transport
    last = None
    for attempt in range(3):
        try:
            return transport(body)
        except (OSError, urllib.error.URLError, KeyError, ValueError) as e:
            # the hosted endpoint returned transient 404/410s in the model test
            last = e
            if attempt < 2:
                _SLEEP(_BACKOFF[attempt])
    raise RuntimeError(f'NIM call to {MODEL} failed after 3 attempts: {last}')


def identify(crop_rgb: np.ndarray, transport=None) -> str:
    ok, png = cv2.imencode('.png', cv2.cvtColor(np.ascontiguousarray(crop_rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('could not PNG-encode the identification crop')
    b64 = base64.b64encode(png.tobytes()).decode()
    body = {'model': MODEL, 'temperature': 0.0, 'max_tokens': 60, 'messages': [{
        'role': 'user', 'content': [
            {'type': 'text', 'text': IDENTIFY_PROMPT},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}]}]}
    return _call(body, transport).strip().strip('"').strip()


def categorize(name: str, categories: tuple, transport=None) -> str:
    opts = ' or '.join(f'{{"category": "{c}"}}' for c in categories)
    body = {'model': MODEL, 'temperature': 0.0, 'max_tokens': 30,
            'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content':
                          f'Classify the product. Reply JSON {opts}. '
                          'food = anything people eat or drink.'},
                         {'role': 'user', 'content': name}]}
    text = _call(body, transport)
    try:
        cat = json.loads(text[text.index('{'):text.rindex('}') + 1])['category']
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f'categorize: no valid JSON category in {text!r}') from e
    cat = str(cat).strip().lower()
    if cat not in categories:
        raise ValueError(f'categorize: {cat!r} is not one of {categories}')
    return cat


def identification_crop(rgb_hi, segmap_hi, seg_id, pad: int = 36) -> np.ndarray:
    seg = np.asarray(segmap_hi).reshape(rgb_hi.shape[:2])
    mask = seg == seg_id
    vs, us = np.nonzero(mask)
    if len(vs) == 0:
        raise ValueError(f'seg_id {seg_id} not in the hi-res segmap')
    H, W = mask.shape
    y0, y1 = max(0, vs.min() - pad), min(H, vs.max() + 1 + pad)
    x0, x1 = max(0, us.min() - pad), min(W, us.max() + 1 + pad)
    crop = rgb_hi[y0:y1, x0:x1, :3].copy()
    crop[~mask[y0:y1, x0:x1]] = 255      # masked crops beat raw ones in the model test
    return crop


class KnowledgeCache:
    """One identify + one categorize per seg_id per run (seg ids are stable)."""

    def __init__(self, transport=None):
        self.transport = transport
        self._by_seg = {}
        self.calls = 0
        self.seconds = 0.0

    def lookup(self, seg_id, make_crop, categories, oracle_name=None):
        if seg_id in self._by_seg:
            return self._by_seg[seg_id]
        t0 = time.time()
        if oracle_name is None:
            name = identify(make_crop(), self.transport)
            self.calls += 1
        else:
            name = oracle_name
        cat = categorize(name, tuple(categories), self.transport)
        self.calls += 1
        self.seconds += time.time() - t0
        self._by_seg[seg_id] = (name, cat)
        return name, cat

    def items(self):
        return self._by_seg.items()
