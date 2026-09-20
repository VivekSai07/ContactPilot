"""Builds a merged Panda+Shadow-Hand MJCF asset: grafts Shadow Hand E3M5's
palm-and-fingers subtree onto the Panda's link7, in place of the stock
2-finger parallel gripper, and writes the result into
mujoco_grasp_sim/assets/ (never back into the mujoco_menagerie/ submodule
-- same convention scene_generator.py already uses for panda.xml).

Shadow Hand's own forearm/wrist (rh_forearm, rh_wrist, joints rh_WRJ1/
rh_WRJ2, actuators rh_A_WRJ1/rh_A_WRJ2) are dropped entirely: the Panda's
7 arm joints already provide positioning, and driving a second wrist is
out of scope for this plan (see the design spec's "Decisions").
"""
import re
from pathlib import Path

from sim_grasp.scene_generator import MENAGERIE_PANDA_DIR, GENERATED_DIR, REPO_ROOT

MENAGERIE_SHADOW_HAND_DIR = REPO_ROOT / 'mujoco_menagerie' / 'shadow_hand'

# The exact mount frame the stock Panda gripper's "hand" body used to
# occupy on link7 (panda.xml: <body name="hand" pos="0 0 0.107"
# quat="0.9238795 0 0 -0.3826834">) -- rh_palm is grafted here verbatim,
# becoming a direct child of link7 (rh_wrist/rh_forearm are dropped, so
# rh_palm's own <body> tag is rewritten to carry this mount frame).
HAND_MOUNT_POS = '0 0 0.107'
HAND_MOUNT_QUAT = '0.9238795 0 0 -0.3826834'


def _extract_balanced_tag(text: str, open_tag_literal: str, tag_name: str) -> str:
    """Returns the full `<tag_name ...> ... </tag_name>` block starting at
    the literal `open_tag_literal`, matching its close by depth-counting
    (not a non-greedy regex, which stops at the first nested close instead
    of the real one for a deeply-nested tree like MJCF <default>)."""
    start = text.index(open_tag_literal)
    token_re = re.compile(rf'<{tag_name}\b[^>]*>|</{tag_name}>')
    depth = 0
    for m in token_re.finditer(text, start):
        depth += 1 if not m.group(0).startswith('</') else -1
        if depth == 0:
            return text[start:m.end()]
    raise AssertionError(f'unbalanced <{tag_name}> starting at {open_tag_literal!r}')


def _extract_shadow_hand_pieces(shadow_xml: str) -> dict:
    """Pulls the four pieces needed from shadow_hand/right_hand.xml:
    the <default class="right_hand"> tree, the <mesh>/<material> asset
    entries, the rh_palm body subtree (with its own rh_WRJ1 <joint> line
    removed and its mount frame rewritten), and the parts of
    <tendon>/<actuator>/<contact> that don't reference the dropped wrist."""
    # Shadow Hand's own <default> is wrapped in an outer classless
    # <default> tag (MJCF only allows one classless default per file, and
    # panda.xml already has its own) -- pull just the inner
    # <default class="right_hand"> tree to insert as a sibling of panda's
    # own class="panda" default, inside panda's single classless wrapper.
    # This tree nests many levels deep (finger/thumb sub-defaults), so a
    # non-greedy regex can't find its matching close -- balance-count it.
    default_block = _extract_balanced_tag(
        shadow_xml, '<default class="right_hand">', 'default')
    # panda.xml already defines its own material named "black" (different
    # rgba) -- rename Shadow Hand's to avoid a duplicate-name XML error.
    default_block = default_block.replace('material="black"', 'material="rh_black"')

    mesh_dir = str((MENAGERIE_SHADOW_HAND_DIR / 'assets').resolve())
    asset_block = re.search(r'<asset>(.*?)</asset>', shadow_xml, re.DOTALL).group(1)
    # Individual mesh files get an absolute path, bypassing this file's
    # own <compiler meshdir> (same "absolute path so meshdir is bypassed"
    # trick scene_generator.py already uses for YCB objects).
    asset_block = re.sub(
        r'file="([^"]+)"', lambda m: f'file="{mesh_dir}/{m.group(1)}"', asset_block)
    asset_block = asset_block.replace('name="black"', 'name="rh_black"')

    # rh_palm nests as worldbody > rh_forearm > rh_wrist > rh_palm > (fingers).
    # Its own closing </body> is immediately followed by rh_wrist's and
    # rh_forearm's closes, then </worldbody> -- capture rh_palm's complete,
    # self-balanced subtree (including its own close) and require exactly
    # those 2 ancestor closes + </worldbody> right after it, as a structural
    # sanity check that we grabbed the right node.
    palm_match = re.search(
        r'(<body name="rh_palm".*?</body>)\s*</body>\s*</body>\s*</worldbody>',
        shadow_xml, re.DOTALL)
    assert palm_match is not None, (
        'could not locate rh_palm body subtree followed by exactly 2 '
        'ancestor closes (rh_wrist, rh_forearm) + </worldbody>')
    palm_body = palm_match.group(1)

    # Drop rh_palm's own joint (the dropped rh_WRJ1) -- everything below
    # it (fingers, thumb) is untouched.
    palm_body, n = re.subn(r'\s*<joint[^/]*name="rh_WRJ1"[^/]*/>', '', palm_body)
    assert n == 1, 'expected exactly one rh_WRJ1 <joint> line inside rh_palm'

    # Rewrite rh_palm's own mount frame so it lands exactly where the old
    # Panda gripper's "hand" body used to sit, once grafted directly under
    # link7 (rh_wrist/rh_forearm, which used to carry the frame between
    # them, no longer exist).
    palm_body, n = re.subn(
        r'<body name="rh_palm" pos="[^"]*"',
        f'<body name="rh_palm" pos="{HAND_MOUNT_POS}" quat="{HAND_MOUNT_QUAT}"',
        palm_body, count=1)
    assert n == 1, "could not rewrite rh_palm's mount frame"

    tendon_block = re.search(r'<tendon>(.*?)</tendon>', shadow_xml, re.DOTALL).group(1)

    actuator_block = re.search(
        r'<actuator>(.*?)</actuator>', shadow_xml, re.DOTALL).group(1)
    # Drop the 2 wrist actuators; keep the 18 finger/thumb ones.
    actuator_lines = [l for l in actuator_block.splitlines()
                      if 'rh_A_WRJ' not in l]
    actuator_block = '\n'.join(actuator_lines)

    # Only keep the thumb self-collision exclude -- the other one
    # referenced rh_wrist/rh_forearm, which no longer exist.
    contact_block = '<contact>\n      <exclude body1="rh_thproximal" body2="rh_thmiddle"/>\n    </contact>'

    return {
        'default': default_block,
        'asset': asset_block,
        'palm_body': palm_body,
        'tendon': tendon_block,
        'actuator': actuator_block,
        'contact': contact_block,
    }


def build_panda_shadow_hand_xml() -> str:
    shadow_hand_file = MENAGERIE_SHADOW_HAND_DIR / 'right_hand.xml'
    if not shadow_hand_file.exists():
        raise FileNotFoundError(
            f'{shadow_hand_file} not found -- mujoco_menagerie is sparse-checked-out '
            "to franka_emika_panda only by default. Run "
            "`git -C mujoco_menagerie sparse-checkout set franka_emika_panda shadow_hand` "
            "to add the Shadow Hand assets (see README.md \"Getting the submodules\").")
    panda_xml = (MENAGERIE_PANDA_DIR / 'panda.xml').read_text(encoding='utf-8')
    shadow_xml = shadow_hand_file.read_text(encoding='utf-8')
    pieces = _extract_shadow_hand_pieces(shadow_xml)

    # Rewrite panda.xml's own meshdir to an absolute path (same as
    # scene_generator._patched_panda_xml -- this file is not that
    # function, so it must redo that one rewrite itself).
    abs_panda_meshdir = str((MENAGERIE_PANDA_DIR / 'assets').resolve())
    patched = re.sub(r'meshdir="assets"', f'meshdir="{abs_panda_meshdir}"', panda_xml, count=1)

    # Remove the stock 2-finger gripper: the "hand" body (and everything
    # inside it -- left_finger, right_finger), the "split" fixed tendon,
    # its <equality> coupling, and actuator8. The "hand" body's own
    # complete subtree is self-balanced (hand > left_finger, right_finger),
    # ending 2 closes after right_finger's own close (right_finger, hand);
    # link7's closing tag right after must be left untouched.
    patched, n = re.subn(
        r'\s*<body name="hand".*?</body>\s*</body>',
        '', patched, count=1, flags=re.DOTALL)
    assert n == 1, 'expected exactly one "hand" body subtree in panda.xml'
    patched = re.sub(r'\s*<tendon>.*?</tendon>', '', patched, count=1, flags=re.DOTALL)
    patched = re.sub(
        r'\s*<equality>.*?</equality>', '', patched, count=1, flags=re.DOTALL)
    patched, n = re.subn(
        r'\s*<general[^/]*name="actuator8".*?/>', '', patched, count=1, flags=re.DOTALL)
    assert n == 1, 'expected exactly one actuator8 <general> element in panda.xml'

    # Graft rh_palm directly after link7's collision geom -- it is now a
    # direct child of link7, with no wrapper body (its own mount frame was
    # already rewritten above).
    patched, n = re.subn(r'(<geom mesh="link7_c" class="collision"/>)',
                         r'\1\n' + pieces['palm_body'], patched, count=1)
    assert n == 1, 'could not find the link7 collision geom to graft the hand after'

    # Merge in the Shadow Hand's own <default>, <asset> additions, the
    # remaining <tendon>/<actuator>/<contact> blocks. <asset> is unique in
    # panda.xml so the first (only) </asset> is correct; <default> nests
    # deeply (finger/collision sub-defaults each have their own </default>),
    # so the shadow-hand default tree must go before the LAST </default>
    # (the outermost one), not the first.
    patched = patched.replace('</asset>', pieces['asset'] + '\n  </asset>', 1)
    default_close_idx = patched.rfind('</default>')
    patched = (patched[:default_close_idx] + pieces['default'] + '\n'
               + patched[default_close_idx:])
    # <tendon>/<contact>/<actuator> must land INSIDE <mujoco>...</mujoco>,
    # so build the whole tail and replace the closing tag once.
    tail = (
        f'\n<tendon>{pieces["tendon"]}</tendon>\n'
        f'{pieces["contact"]}\n'
        f'<actuator>{pieces["actuator"]}</actuator>\n</mujoco>'
    )
    patched, n = re.subn(r'</mujoco>\s*$', tail, patched, count=1)
    assert n == 1, 'could not find the closing </mujoco> tag in panda.xml'

    out = GENERATED_DIR / 'panda_shadow_hand.xml'
    out.write_text(patched, encoding='utf-8')
    return out.name
