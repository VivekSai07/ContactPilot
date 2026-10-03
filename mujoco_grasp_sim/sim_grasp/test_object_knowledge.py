"""Standalone checks for object_knowledge (fake transport, no network)."""
import numpy as np

import sim_grasp.object_knowledge as ok

replies = []
def fake(body):
    replies.append(body)
    return fake.next.pop(0)

# categorize: JSON inside prose, case-normalised, enum-validated
fake.next = ['Sure! {"category": "Food"}']
assert ok.categorize('box of cookies', ('food', 'non_food'), transport=fake) == 'food'
assert replies[-1]['response_format'] == {'type': 'json_object'}
fake.next = ['{"category": "drink"}']
try:
    ok.categorize('juice', ('food', 'non_food'), transport=fake); raise AssertionError('no raise')
except ValueError:
    pass

# identify sends an image and strips quotes/whitespace
fake.next = ['  "Box of crayons."  ']
assert ok.identify(np.zeros((10, 10, 3), np.uint8), transport=fake) == 'Box of crayons.'
content = replies[-1]['messages'][0]['content']
assert content[1]['image_url']['url'].startswith('data:image/png;base64,')
assert replies[-1]['model'] == ok.MODEL == 'meta/llama-3.2-11b-vision-instruct'

# retries then raises RuntimeError
calls = {'n': 0}
def flaky(body):
    calls['n'] += 1
    raise OSError('boom')
_ORIG_SLEEP = ok._SLEEP
ok._SLEEP = lambda s: None
try:
    ok.identify(np.zeros((4, 4, 3), np.uint8), transport=flaky); raise AssertionError('no raise')
except RuntimeError:
    pass
assert calls['n'] == 3

# crop: masked to white, padded, clamped at the border
rgb = np.full((50, 60, 3), 7, np.uint8); seg = np.zeros((50, 60), np.float32)
seg[0:10, 0:8] = 2                                   # touches the top-left border
c = ok.identification_crop(rgb, seg, 2, pad=36)
assert c.shape[0] == 46 and c.shape[1] == 44         # 0..10+36, 0..8+36
assert (c[0, 0] == 7).all() and (c[-1, -1] == 255).all()

# cache: second lookup of the same seg_id makes no calls; oracle skips identify
fake.next = ['box of cookies', '{"category": "food"}']
cache = ok.KnowledgeCache(transport=fake)
r1 = cache.lookup(1, lambda: np.zeros((5, 5, 3), np.uint8), ('food', 'non_food'))
n = len(replies)
r2 = cache.lookup(1, lambda: (_ for _ in ()).throw(AssertionError('crop rebuilt')), ('food', 'non_food'))
assert r1 == r2 == ('box of cookies', 'food') and len(replies) == n and cache.calls == 2
fake.next = ['{"category": "non_food"}']
r3 = cache.lookup(2, lambda: (_ for _ in ()).throw(AssertionError('no crop in oracle')),
                  ('food', 'non_food'), oracle_name='Crayola Bonus 64 Crayons')
assert r3 == ('Crayola Bonus 64 Crayons', 'non_food') and cache.calls == 3
assert dict(cache.items()) == {1: ('box of cookies', 'food'), 2: ('Crayola Bonus 64 Crayons', 'non_food')}
# a rejected key (HTTP 401) must fail fast: one call, no retries
import urllib.error
n401 = []
def _rejected(body):
    n401.append(1)
    raise urllib.error.HTTPError(ok.NIM_URL, 401, 'Unauthorized', {}, None)
try:
    ok._call({}, _rejected)
    raise AssertionError('expected RuntimeError on 401')
except RuntimeError as e:
    assert 'NVIDIA_API_KEY' in str(e)
assert len(n401) == 1
ok._SLEEP = _ORIG_SLEEP  # why: don't leave the module patched for other importers
print('All object_knowledge checks passed.')
