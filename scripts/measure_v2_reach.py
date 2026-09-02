"""Measure the v2 robot before designing rewards for it.

Answers the three questions that are expensive to discover after a training run
has already been launched:

1. **STAND_Z** -- where the trunk actually settles at HOME on *this* model.
   Carrying a height constant across model revisions is how a goal quietly
   becomes impossible; a 5mm error once cost days.
2. **Is the object reachable?** Maps the ``mouth_tip`` envelope by simulating
   random crouches and keeping only the ones that are *statically feasible*
   (both feet still on the ground, trunk not tipped over). A kinematic reach
   check would happily report poses the robot falls out of, so the filter --
   not the kinematics -- is the point. The sock sits lowest at 14mm and is the
   binding case.
3. **What does the camera see?** Sweeps neck/head pitch and reports where the
   optical axis meets the floor and how much of the frame the robot's own beak
   covers, which bounds the useful head angles for the POV dataset.

A note on the actuator, because it changes every number here.  Training runs on
the BAM actuator (voltage control, firmware kp 200); this standalone scene only
has the MJCF ``position`` servo, whose ``kp=0.55`` is far too soft to hold the
robot up -- at face value the robot folds to a 37mm trunk height and an 81
degree tilt, on the v1 model just as much as on v2, so it is a property of the
scene and not of the jaw.  ``--kp-scale`` stiffens that servo into something
that can actually hold a commanded pose, which is what makes a static reach
measurement mean anything.  Above roughly 50x the solver goes unstable at this
timestep, so the default sits well below that.

Steps 1 and 2 are pure physics and run anywhere. Step 3 needs a GL context (on
macOS, a real user session; headless Linux wants ``MUJOCO_GL=egl``) and is
skipped automatically if one is unavailable.

Usage::

    uv run scripts/measure_v2_reach.py
    uv run scripts/measure_v2_reach.py --samples 1500 --no-camera
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import mujoco
import numpy as np
import tyro

from mjlab_microduck.robot.microduck_constants import HOME_FRAME, _ROBOT_DIR

SCENE = _ROBOT_DIR / "scene_jaw_objects.xml"

# Grab height of each prop = the height of its top surface at rest, i.e. how
# high the beak has to get down to. Kept in sync with objects.xml.
GRAB_HEIGHTS_MM = {"sock": 14.0, "block": 20.0, "rubber_ball": 30.0}

LEG_PATTERN = re.compile(r".*(hip|knee|ankle).*")
HEAD_PATTERN = re.compile(r".*(neck|head)_.*")


@dataclass
class MeasureConfig:
    samples: int = 800
    """Random crouches to try when mapping the reach envelope."""
    refine_rounds: int = 5
    """CEM rounds that push toward the LOWEST reachable beak height, which is
    what decides whether each prop is in scope."""
    settle_s: float = 1.5
    """Seconds to hold each sampled pose before measuring it."""
    max_tilt_deg: float = 40.0
    """Above this trunk tilt the pose counts as fallen, not reached."""
    seed: int = 0
    camera: bool = True
    """Sweep head angles and measure beak occlusion (needs a GL context)."""
    kp_scale: float = 20.0
    """Stiffen the MJCF position servo to stand in for the BAM firmware loop.
    At 1.0 the robot cannot hold HOME at all; 5-50 all settle at the same
    height, and above ~50 the solver blows up at this timestep."""


def home_values(model: mujoco.MjModel) -> dict[str, float]:
    out = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for pattern, value in HOME_FRAME.joint_pos.items():
            if re.match(pattern, name):
                out[name] = float(value)
                break
    return out


class Rig:
    def __init__(self, model: mujoco.MjModel, kp_scale: float = 1.0) -> None:
        if kp_scale != 1.0:
            model.actuator_gainprm[:, 0] *= kp_scale
            model.actuator_biasprm[:, 1] *= kp_scale
            model.actuator_forcerange[:] *= kp_scale
        self.model = model
        self.data = mujoco.MjData(model)
        self.home = home_values(model)
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.trunk_qadr = int(model.jnt_qposadr[model.body_jntadr[self.trunk]])
        self.mouth = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
        self.jaw_tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "jaw_tip")
        self.foot_geoms = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("left_foot_collision", "right_foot_collision")
        ]

        # Actuator index by the joint it drives, so poses are set by NAME.
        self.act = {}
        self.qadr = {}
        for a in range(model.nu):
            jid = model.actuator_trnid[a, 0]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            self.act[name] = a
            self.qadr[name] = int(model.jnt_qposadr[jid])
        self.limits = {n: model.jnt_range[model.actuator_trnid[a, 0]] for n, a in self.act.items()}

        # Park the props: this measurement is about the robot alone.
        for body in ("block", "rubber_ball", "sock"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if bid >= 0:
                adr = int(model.jnt_qposadr[model.body_jntadr[bid]])
                self.park = getattr(self, "park", [])
                self.park.append(adr)

    def reset_home(self) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for name, value in self.home.items():
            d.qpos[self.qadr[name]] = value
            d.ctrl[self.act[name]] = value
        d.qpos[self.trunk_qadr : self.trunk_qadr + 3] = [0.0, 0.0, 0.12]
        d.qpos[self.trunk_qadr + 3 : self.trunk_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        for adr in getattr(self, "park", []):
            d.qpos[adr : adr + 3] = [0.0, 0.0, -0.5]
            d.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d)

    def run(self, seconds: float) -> None:
        for _ in range(int(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def tilt_deg(self) -> float:
        up = self.data.xmat[self.trunk].reshape(3, 3)[:, 2]
        return math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))

    def feet_down(self) -> bool:
        touched = set()
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            for g in (con.geom1, con.geom2):
                if g in self.foot_geoms:
                    touched.add(int(g))
        return len(touched) == 2

    def trunk_z(self) -> float:
        return float(self.data.xpos[self.trunk][2])

    def mouth_in_trunk_ground_frame(self) -> tuple[float, float]:
        """(forward distance from the trunk's ground projection, height)."""
        d = self.data
        p = d.site_xpos[self.jaw_tip if self.jaw_tip >= 0 else self.mouth]
        trunk_xy = d.xpos[self.trunk][:2]
        yaw_fwd = d.xmat[self.trunk].reshape(3, 3)[:, 0][:2]
        n = np.linalg.norm(yaw_fwd)
        yaw_fwd = yaw_fwd / n if n > 1e-9 else np.array([1.0, 0.0])
        return float((p[:2] - trunk_xy) @ yaw_fwd), float(p[2])


def measure_stand(rig: Rig) -> float:
    rig.reset_home()
    rig.run(3.0)
    z, tilt = rig.trunk_z(), rig.tilt_deg()
    fwd, mouth_z = rig.mouth_in_trunk_ground_frame()
    print("== 1. Standing equilibrium at HOME ==")
    print(f"  STAND_Z (trunk height after 3s) : {z * 1000:.1f} mm")
    print(f"  trunk tilt                      : {tilt:.1f} deg")
    print(f"  feet both in contact            : {rig.feet_down()}")
    print(f"  beak tip                        : {fwd * 1000:.0f} mm forward, "
          f"{mouth_z * 1000:.0f} mm up")
    if tilt > 15.0:
        print("  WARNING: HOME is not a stable equilibrium on this model — every "
              "height measured below is taken from a leaning robot. Raise "
              "--kp-scale: the MJCF servo alone cannot hold the pose.")
    print(f"\n  -> use STAND_Z = {z:.4f} (do NOT carry 0.115 over from the standup env)")
    return z


def measure_reach(rig: Rig, cfg: MeasureConfig) -> None:
    """Map the beak envelope over SYMMETRIC crouches.

    Sampling the ten leg joints independently is close to useless here: almost
    every draw is asymmetric, the robot falls, and the survivors are a biased
    scrap of the envelope (an early run kept 8 of 600). A crouch is a
    coordinated sagittal motion, so the search runs over the four parameters
    that actually describe one -- hip fold, knee bend, ankle pitch, and where
    the head is aimed -- mirrored left/right. That keeps the robot balanced by
    construction, so a discarded sample means genuinely out of reach.
    """
    rng = np.random.default_rng(cfg.seed)
    # (hip fold, knee bend, ankle pitch, neck_pitch, head_pitch)
    lo_p = np.array([-0.4, -1.6, -0.9, *rig.limits["neck_pitch"][:1], *rig.limits["head_pitch"][:1]])
    hi_p = np.array([1.5, 0.4, 0.9, rig.limits["neck_pitch"][1], rig.limits["head_pitch"][1]])

    def evaluate(p: np.ndarray) -> tuple[float, float] | None:
        rig.reset_home()
        rig.run(0.4)
        d_hip, d_knee, d_ankle, neck, head = p
        # Sign mirrors HOME: the left leg's pitch chain is negative, the right
        # positive, so a shared delta bends both legs the same way.
        deltas = {
            "left_hip_pitch": -d_hip, "right_hip_pitch": d_hip,
            "left_knee": -d_knee, "right_knee": d_knee,
            "left_ankle": d_ankle, "right_ankle": -d_ankle,
        }
        for name, delta in deltas.items():
            j_lo, j_hi = rig.limits[name]
            rig.data.ctrl[rig.act[name]] = float(np.clip(rig.home[name] + delta, j_lo, j_hi))
        for name, value in (("neck_pitch", neck), ("head_pitch", head)):
            j_lo, j_hi = rig.limits[name]
            rig.data.ctrl[rig.act[name]] = float(np.clip(value, j_lo, j_hi))
        rig.run(cfg.settle_s)
        if rig.tilt_deg() > cfg.max_tilt_deg or not rig.feet_down():
            return None
        if not np.all(np.isfinite(rig.data.qpos)):
            return None
        return rig.mouth_in_trunk_ground_frame()

    feasible: list[tuple[float, float]] = []
    scored: list[tuple[float, np.ndarray]] = []  # (beak height, params)
    fallen = 0

    def run_batch(params: np.ndarray) -> None:
        nonlocal fallen
        for p in params:
            result = evaluate(p)
            if result is None:
                fallen += 1
            else:
                feasible.append(result)
                scored.append((result[1], p))

    run_batch(rng.uniform(lo_p, hi_p, size=(cfg.samples, 5)))

    # Random search answers "roughly where can the beak go"; it is a poor way to
    # answer "how LOW can it go", which is the question that decides whether the
    # sock is in scope at all. Refine with a few CEM rounds around the deepest
    # crouches found so far.
    for _ in range(cfg.refine_rounds):
        if len(scored) < 8:
            break
        scored.sort(key=lambda sp: sp[0])
        elite = np.array([p for _, p in scored[:8]])
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 0.05
        scored = scored[:8]
        run_batch(np.clip(rng.normal(mean, std, size=(48, 5)), lo_p, hi_p))

    print("\n== 2. Beak reach envelope (statically feasible crouches only) ==")
    print(f"  {len(feasible)} of {len(feasible) + fallen} crouches "
          f"({cfg.samples} random + {cfg.refine_rounds} refinement rounds) stayed "
          f"upright with both feet down")
    if not feasible:
        print("  no feasible poses — cannot assess reach")
        return

    arr = np.array(feasible)
    fwd_mm, z_mm = arr[:, 0] * 1000.0, arr[:, 1] * 1000.0
    print(f"  beak height  : min {z_mm.min():.0f} mm, median {np.median(z_mm):.0f} mm, "
          f"max {z_mm.max():.0f} mm")
    print(f"  beak forward : min {fwd_mm.min():.0f} mm, median {np.median(fwd_mm):.0f} mm, "
          f"max {fwd_mm.max():.0f} mm")

    print("\n  Can the beak get down to each prop?")
    for name, h in sorted(GRAB_HEIGHTS_MM.items(), key=lambda kv: kv[1]):
        # 8mm band around the grab height: the beak has to arrive AT the object,
        # not merely pass below it at some unrelated distance.
        band = np.abs(z_mm - h) < 8.0
        n = int(band.sum())
        if n == 0:
            print(f"    {name:12s} ({h:.0f} mm): NOT REACHABLE in any feasible pose")
        else:
            print(f"    {name:12s} ({h:.0f} mm): reachable in {n} poses, "
                  f"{fwd_mm[band].min():.0f}..{fwd_mm[band].max():.0f} mm ahead of the trunk")

    lowest = z_mm.min()
    binding = min(GRAB_HEIGHTS_MM.values())
    if lowest > binding:
        print(f"\n  WARNING: the lowest feasible beak height is {lowest:.0f} mm but the "
              f"sock needs {binding:.0f} mm. Either the sock is out of scope for v2 or "
              f"the grab needs a deeper crouch than this search found.")
    else:
        reach = np.concatenate([
            fwd_mm[np.abs(z_mm - h) < 8.0] for h in GRAB_HEIGHTS_MM.values()
        ])
        if reach.size:
            print(f"\n  -> spawn objects {reach.min():.0f}..{reach.max():.0f} mm ahead of "
                  f"the trunk: that is the band where the beak can reach every prop "
                  f"without leaving a statically stable crouch.")


def measure_camera(rig: Rig) -> None:
    print("\n== 3. Head camera: aim and self-occlusion ==")
    m, d = rig.model, rig.data
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    try:
        renderer = mujoco.Renderer(m, height=480, width=640)
    except Exception as exc:  # no GL context available
        print(f"  skipped: could not create a GL context ({exc})")
        return
    renderer.enable_segmentation_rendering()

    object_geoms = {
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{b}_geom")
        for b in ("block", "rubber_ball", "sock")
    }
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot_geoms = np.array(
        [g for g in range(m.ngeom) if g not in object_geoms and g != floor], dtype=np.int32
    )

    print(f"  fovy {float(m.cam_fovy[cam]):.1f} deg vertical "
          f"({float(m.cam_resolution[cam][0])}x{float(m.cam_resolution[cam][1])} "
          f"-> {2 * math.degrees(math.atan(math.tan(math.radians(float(m.cam_fovy[cam])) / 2) * 4 / 3)):.1f} deg horizontal)")
    neck_lo, neck_hi = rig.limits["neck_pitch"]
    head_lo, head_hi = rig.limits["head_pitch"]
    print(f"  joint ranges: neck_pitch [{neck_lo:.2f}, {neck_hi:.2f}]  "
          f"head_pitch [{head_lo:.2f}, {head_hi:.2f}]  (HOME is 0.35 / 0.35)")
    print(f"  {'neck':>6} {'head':>6} {'aim@floor':>11} {'beak %frame':>12}")

    for neck in np.linspace(neck_lo, neck_hi, 4):
        for head in np.linspace(head_lo, head_hi, 5):
            rig.reset_home()
            for name, value in (("neck_pitch", neck), ("head_pitch", head)):
                lo, hi = rig.limits[name]
                v = float(np.clip(value, lo, hi))
                d.qpos[rig.qadr[name]] = v
                d.ctrl[rig.act[name]] = v
            mujoco.mj_forward(m, d)

            pos = d.cam_xpos[cam]
            view = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
            if view[2] < -1e-6:
                hit = f"{(pos[0] - view[0] * pos[2] / view[2]) * 1000:8.0f} mm"
            else:
                hit = "  horizon"

            renderer.update_scene(d, camera="head_camera")
            seg = renderer.render()
            is_geom = seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
            share = float((is_geom & np.isin(seg[:, :, 0], robot_geoms)).sum()) / seg[:, :, 0].size
            print(f"  {neck:6.2f} {head:6.2f} {hit:>11} {share * 100:11.1f}%")

    print("\n  'aim@floor' is where the optical axis meets the ground, measured "
          "forward from the trunk. Head angles whose aim lands in the 100-300 mm "
          "band are the ones that see a graspable object; 'horizon' means the "
          "camera is level or looking up and sees no ground at all.")
    print("  NOTE the two pitch joints act in OPPOSITE senses: raising "
          "head_pitch aims DOWN, raising neck_pitch aims UP. They cancel at "
          "HOME, which is why the default stance looks straight ahead.")


def main(cfg: MeasureConfig) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    rig = Rig(model, cfg.kp_scale)
    print(f"model: {SCENE.name}  ({model.nu} actuators, dt={model.opt.timestep}, "
          f"servo kp x{cfg.kp_scale:g})\n")
    measure_stand(rig)
    measure_reach(rig, cfg)
    if cfg.camera:
        measure_camera(rig)


if __name__ == "__main__":
    main(tyro.cli(MeasureConfig))
