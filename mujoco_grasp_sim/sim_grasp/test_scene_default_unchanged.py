"""Guard: P10 (props scene + second bin) must not change the default boxes
scene. Hashes were captured on main before any P10 code change -- the
generated XML and the settled qpos (which also pins the rng consumption
order) must stay identical. Run directly, no pytest."""
import hashlib

import numpy as np

from sim_grasp.scene_generator import SceneConfig, SceneGenerator

EXPECTED = {
    0: ('c3480a539cb692d8b6a5e35f310f63a4c0dd0cccf6d7eaebb12a7c332c1dbc02',
        'daf8c592bdf39aed75539e174ad0ef326b6ef4c48a06e8eaa7d546330c254dad'),
    3: ('fc7a9e4029993e42bece54962d446a950ba84672a48c8062ade669287373e8d2',
        '3e85dcdf1dc075736620833de05d97f067840addc0e9fe7883bc48229c85a854'),
}

for seed, (xml_h, qpos_h) in EXPECTED.items():
    gen = SceneGenerator(SceneConfig(seed=seed))
    model, data = gen.generate()
    got_xml = hashlib.sha256(gen.scene_xml_path.read_bytes()).hexdigest()
    got_q = hashlib.sha256(np.round(data.qpos, 6).tobytes()).hexdigest()
    assert got_xml == xml_h, f'seed {seed}: default scene XML changed'
    assert got_q == qpos_h, f'seed {seed}: default settled qpos changed (rng order?)'

print('Default boxes scene unchanged (seeds 0, 3).')
