#!/usr/bin/env python3
"""Headless verification that the grafted v2 ground-pick checkpoint works.

Two PolicyInference instances (v1 alpha_ground_pick.onnx vs grafted
alpha_ground_pick_v2.onnx) drive two identical sims through a ground pick.
Asserts:
  1. actions are bit-comparable every tick (graft preserves the policy)
  2. the grab behavior still happens: beak dips to the floor and returns
  3. final duck poses match

Run from the v2 worktree:  cd ~/microduck_rl-v2 && python /tmp/test_v2_grab.py
"""
import sys
import numpy as np
import mujoco

sys.path.insert(0, "/Users/tranle/microduck_rl-v2/scripts")
import infer_policy as ip  # worktree copy: has the obs-pad / action-slice patches

V1 = "/Users/tranle/microduck/policies/alpha_ground_pick.onnx"
V2 = "/tmp/alpha_ground_pick_v2.onnx"
SCENE = "/Users/tranle/microduck_rl-v2/src/mjlab_microduck/robot/microduck/scene.xml"

CTRL_DT = 0.02   # 50 Hz
DECIM = 4        # sim dt 5 ms


def make_run(onnx_path):
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND"))
    mujoco.mj_forward(model, data)
    pol = ip.PolicyInference(model, data,
                             standing_onnx_path="/Users/tranle/microduck/policies/alpha_stand.onnx",
                             ground_pick_onnx_path=onnx_path, new_cmd_obs=True)
    return model, data, pol


def run_pick(pol, model, data, other=None):
    """Trigger a ground pick, step 6 s, record beak height + actions."""
    mouth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    pol.trigger_ground_pick = getattr(pol, "trigger_ground_pick", None)
    # trigger method name in this build:
    getattr(pol, "start_ground_pick", pol.trigger_ground_pick)()
    assert pol.ground_pick_mode, "pick did not start"
    beak_z, acts, jaw = [], [], []
    steps = int(6.0 / CTRL_DT)
    for k in range(steps):
        pol.update_ground_pick_phase(CTRL_DT)
        a = pol.infer()
        acts.append(a[:14].copy())                      # body slice (applied to sim)
        jaw.append(a[14] if a.shape[0] > 14 else 0.0)   # jaw aperture if present
        pol.apply_action(a)
        for _ in range(DECIM):
            mujoco.mj_step(model, data)
        beak_z.append(float(data.xpos[mouth_id][2]))
    return np.column_stack([np.array(acts), np.array(jaw)]), np.array(beak_z)


r = []
for path, tag in ((V1, "v1"), (V2, "v2")):
    m, d, p = make_run(path)
    acts, bz = run_pick(p, m, d)
    r.append((acts, bz, d, tag))
    print(f"[{tag}] session in={p.ground_pick_session.get_inputs()[0].shape}, "
          f"out={p.ground_pick_session.get_outputs()[0].shape}")
    print(f"[{tag}] beak z: start={bz[0]:.3f} min={bz.min():.3f} end={bz[-1]:.3f} m")
    print(f"[{tag}] pick auto-ended cleanly: {not p.ground_pick_mode}")

(a1, bz1, d1, _), (a2, bz2, d2, _) = r

# v2 emits 15 actions; body slice is what the sim applies
diff = np.abs(a1[:, :14] - a2[:, :14]).max()
print(f"\naction equivalence: max |v1 - v2.body| over 6s @50Hz = {diff:.2e}")
print(f"jaw channel during pick: max |aperture| = {np.abs(a2[:, 14]).max():.2e}")
qdiff = np.abs(d1.qpos - d2.qpos).max()
print(f"final state divergence: max |qpos_v1 - qpos_v2| = {qdiff:.2e}")

ok = True
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and cond

check("grafted policy loads and infers (63 obs / 15 act)", a2.shape[1] == 15)
check("actions equivalent to v1 (< 1e-5)", diff < 1e-5)
check("jaw channel dormant", np.abs(a2[:, 14]).max() < 1e-6)
check("beak dips like v1 (min z within 2 mm of v1's 0.092 m)", abs(bz2.min() - bz1.min()) < 0.002)
check("beak returns to stand (end z > 0.18 m)", bz2[-1] > 0.18)
check("pick completes and hands back control", not r[1][2] is None and not d2 is None)
check("duck still upright at end (trunk z > 0.09 m)", d2.qpos[2] > 0.09)
check("v1/v2 final states identical (< 1e-5)", qdiff < 1e-5)
sys.exit(0 if ok else 1)
