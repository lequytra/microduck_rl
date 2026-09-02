"""Render a labelled first-person dataset from the microduck's head camera.

Produces, per frame, the RGB image the real Pi Cam v2 / IMX219 would see plus
two labels:

* a **2D bounding box** per visible object, and
* the object's **(dx, dy, dz) in the robot's trunk frame**.

The second label is the one that matters downstream: the runtime pipeline is
bbox -> (dx, dy) -> go-to-point command, and having the ground-truth offset in
the same record means the bbox-to-command regression can be fit directly
instead of being derived from an assumed ground plane.

Boxes come from MuJoCo's **segmentation** render, not from projecting geometry.
That distinction is not cosmetic: the camera sits inside the head and the beak
occludes part of every frame, so a projected box would happily label an object
that is entirely hidden behind the bill. Segmentation counts real visible
pixels, which also gives a free occlusion measure (``visible_px`` vs the box
area) to filter marginal frames.

Intrinsics: ``head_camera`` carries ``fovy=48.8`` and ``resolution="640 480"``,
matching the IMX219's published 62.2 x 48.8 degree field of view. Rendering at
a different aspect ratio changes the effective horizontal FOV and teaches the
detector a wrong scale, so ``--width/--height`` should keep 4:3.

Rendering needs a GL context. On macOS run it from a normal user session (an
offscreen CGL context still needs a CoreGraphics connection); on a headless
Linux box set ``MUJOCO_GL=egl`` or ``osmesa``.

Usage::

    uv run scripts/render_pov.py --out data/pov --frames 2000
    uv run scripts/render_pov.py --out /tmp/peek --frames 20 --debug-overlay
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw

from mjlab_microduck.robot.microduck_constants import HOME_FRAME, _ROBOT_DIR

SCENE = _ROBOT_DIR / "scene_jaw_objects.xml"
# Bodies from objects.xml. Each has exactly one geom, named <body>_geom.
OBJECT_BODIES = ("block", "rubber_ball", "sock")
# Parking spot for the objects that are not in the current frame. Below the
# floor plane, so they are hidden no matter where the camera looks.
PARK = (0.0, 0.0, -0.5)


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lo, self.hi))


@dataclass
class RenderConfig:
    out: Path
    """Output directory. Images go to <out>/images, labels to <out>/labels.jsonl."""
    frames: int = 1000
    width: int = 640
    height: int = 480
    seed: int = 0
    min_visible_px: int = 12
    """Objects with fewer visible pixels than this are dropped from the labels."""
    max_objects: int = 3
    """Upper bound on props placed per frame; the actual count is sampled."""
    debug_overlay: bool = False
    """Also write <out>/overlay/*.png with the boxes drawn, for eyeballing."""

    # --- domain randomization ---
    obj_radius: Range = field(default_factory=lambda: Range(0.08, 0.90))
    """Distance from the trunk to each prop, in metres."""
    obj_bearing_deg: Range = field(default_factory=lambda: Range(-45.0, 45.0))
    look_down: Range = field(default_factory=lambda: Range(0.30, 1.70))
    """How far below the horizon the camera is aimed, as ``head_pitch -
    neck_pitch``.

    Sampling the two pitch joints independently wastes most frames on the sky:
    they act in OPPOSITE senses (head_pitch aims down, neck_pitch aims up) and
    cancel at HOME, so it is their DIFFERENCE that sets the aim, and only a
    narrow slice of the joint box points at the ground. Measured with
    scripts/measure_v2_reach.py: 0.5 puts the optical axis on the floor ~390mm
    ahead, 1.6 brings it in to ~100mm. The split between the two joints is then
    free, which keeps the neck/head posture varied."""
    head_yaw: Range = field(default_factory=lambda: Range(-0.5, 0.5))
    head_roll: Range = field(default_factory=lambda: Range(-0.2, 0.2))
    jaw: Range = field(default_factory=lambda: Range(0.0, 0.7))
    leg_noise: float = 0.06
    """Uniform +/- noise on every leg joint, in rad (stance variation)."""
    trunk_z: Range = field(default_factory=lambda: Range(0.105, 0.125))
    trunk_tilt: Range = field(default_factory=lambda: Range(-0.10, 0.10))
    light_diffuse: Range = field(default_factory=lambda: Range(0.35, 0.95))
    light_ambient: Range = field(default_factory=lambda: Range(0.10, 0.45))


def home_joint_values(model: mujoco.MjModel) -> dict[str, float]:
    """Resolve HOME_FRAME's regex patterns against the real joint names.

    HOME_FRAME is the single source of truth for the standing pose; re-listing
    the angles here is how a pose silently drifts out of sync with training.
    Matching is first-match-wins, exactly as mjlab resolves it.
    """
    import re

    out: dict[str, float] = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for pattern, value in HOME_FRAME.joint_pos.items():
            if re.match(pattern, name):
                out[name] = float(value)
                break
    return out


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "xyz")
    return q


class PovScene:
    """The scene plus the id bookkeeping the labeller needs."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.home = home_joint_values(model)

        self.trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.trunk_qadr = model.jnt_qposadr[model.body_jntadr[self.trunk_id]]
        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera"
        )
        if self.camera_id < 0:
            raise RuntimeError("scene has no head_camera")

        self.joint_qadr = {
            name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in self.home
        }
        self.limits = {
            name: model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.home
        }

        self.objects: dict[str, dict] = {}
        for body in OBJECT_BODIES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if bid < 0:
                raise RuntimeError(f"scene has no body {body!r}")
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{body}_geom")
            size = model.geom_size[gid]
            # Resting half-height: box uses its z half-extent, sphere its radius.
            half_h = float(size[2] if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX else size[0])
            self.objects[body] = {
                "body_id": bid,
                "geom_id": gid,
                "qadr": int(model.jnt_qposadr[model.body_jntadr[bid]]),
                "half_height": half_h,
            }

        object_geoms = {o["geom_id"] for o in self.objects.values()}
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.robot_geom_ids = np.array(
            [g for g in range(model.ngeom) if g not in object_geoms and g != floor_id],
            dtype=np.int32,
        )

    def randomize(self, cfg: RenderConfig, rng: np.random.Generator) -> list[str]:
        """Draw one scene: robot pose, prop placement, lighting. Returns the
        names of the props that were placed (the rest are parked)."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)

        for name, value in self.home.items():
            adr = self.joint_qadr[name]
            if "hip" in name or "knee" in name or "ankle" in name:
                value += rng.uniform(-cfg.leg_noise, cfg.leg_noise)
            d.qpos[adr] = value

        look_down = cfg.look_down.sample(rng)
        neck_lo, neck_hi = self.limits["neck_pitch"]
        head_lo, head_hi = self.limits["head_pitch"]
        neck = rng.uniform(
            max(neck_lo, head_lo - look_down), min(neck_hi, head_hi - look_down)
        )
        d.qpos[self.joint_qadr["neck_pitch"]] = neck
        d.qpos[self.joint_qadr["head_pitch"]] = neck + look_down

        for name, rng_cfg in (
            ("head_yaw", cfg.head_yaw),
            ("head_roll", cfg.head_roll),
            ("jaw", cfg.jaw),
        ):
            if name in self.joint_qadr:
                d.qpos[self.joint_qadr[name]] = rng_cfg.sample(rng)

        base_yaw = float(rng.uniform(-math.pi, math.pi))
        base_xy = rng.uniform(-0.5, 0.5, size=2)
        adr = self.trunk_qadr
        d.qpos[adr : adr + 3] = [base_xy[0], base_xy[1], cfg.trunk_z.sample(rng)]
        d.qpos[adr + 3 : adr + 7] = quat_from_rpy(
            cfg.trunk_tilt.sample(rng), cfg.trunk_tilt.sample(rng), base_yaw
        )

        n_objects = int(rng.integers(1, cfg.max_objects + 1))
        placed = list(rng.permutation(list(self.objects))[:n_objects])
        cos_y, sin_y = math.cos(base_yaw), math.sin(base_yaw)
        for name, obj in self.objects.items():
            oadr = obj["qadr"]
            if name in placed:
                r = cfg.obj_radius.sample(rng)
                bearing = math.radians(cfg.obj_bearing_deg.sample(rng))
                # Sample in the robot frame, then rotate into the world, so the
                # props stay in front of the robot whatever its yaw is.
                fx, fy = r * math.cos(bearing), r * math.sin(bearing)
                pos = (
                    base_xy[0] + cos_y * fx - sin_y * fy,
                    base_xy[1] + sin_y * fx + cos_y * fy,
                    obj["half_height"],
                )
                quat = quat_from_rpy(0.0, 0.0, float(rng.uniform(-math.pi, math.pi)))
            else:
                pos, quat = PARK, np.array([1.0, 0.0, 0.0, 0.0])
            d.qpos[oadr : oadr + 3] = pos
            d.qpos[oadr + 3 : oadr + 7] = quat
            # Appearance DR: the detector must not key on the exact prop colour.
            m.geom_rgba[obj["geom_id"], :3] = rng.uniform(0.08, 0.95, size=3)

        for light in range(m.nlight):
            m.light_diffuse[light, :] = cfg.light_diffuse.sample(rng)
            m.light_ambient[light, :] = cfg.light_ambient.sample(rng)
            m.light_pos[light, :2] = rng.uniform(-3.0, 3.0, size=2)

        mujoco.mj_forward(m, d)
        return placed

    def object_in_trunk_frame(self, name: str) -> np.ndarray:
        """(dx, dy, dz) of a prop in the trunk frame — the regression target."""
        d = self.data
        rot = d.xmat[self.trunk_id].reshape(3, 3)
        return rot.T @ (d.xpos[self.objects[name]["body_id"]] - d.xpos[self.trunk_id])


def boxes_from_segmentation(
    seg: np.ndarray, scene: PovScene, placed: list[str], min_px: int
) -> tuple[list[dict], float]:
    """Exact boxes for the visible props, plus the robot's self-occlusion share."""
    obj_ids = seg[:, :, 0]
    obj_types = seg[:, :, 1]
    is_geom = obj_types == mujoco.mjtObj.mjOBJ_GEOM

    labels = []
    for name in placed:
        mask = is_geom & (obj_ids == scene.objects[name]["geom_id"])
        visible_px = int(mask.sum())
        if visible_px < min_px:
            continue
        ys, xs = np.nonzero(mask)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        dx, dy, dz = scene.object_in_trunk_frame(name)
        labels.append(
            {
                "name": name,
                "bbox_xyxy": [x0, y0, x1, y1],
                "visible_px": visible_px,
                # < 1.0 means something (usually the beak) cuts into the box.
                "fill_ratio": round(visible_px / max((x1 - x0) * (y1 - y0), 1), 4),
                "offset_trunk_m": [round(float(dx), 5), round(float(dy), 5), round(float(dz), 5)],
                "range_m": round(float(math.hypot(dx, dy)), 5),
            }
        )

    robot_px = int((is_geom & np.isin(obj_ids, scene.robot_geom_ids)).sum())
    return labels, robot_px / obj_ids.size


def main(cfg: RenderConfig) -> None:
    if abs(cfg.width / cfg.height - 4 / 3) > 1e-6:
        print(
            f"[render_pov] WARNING: {cfg.width}x{cfg.height} is not 4:3. The camera's "
            f"fovy=48.8 only gives the IMX219's 62.2 deg horizontal FOV at 4:3; "
            f"another aspect silently retrains the detector's scale."
        )

    images_dir = cfg.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = cfg.out / "overlay"
    if cfg.debug_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    scene = PovScene(model)
    rng = np.random.default_rng(cfg.seed)

    renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
    seg_renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
    seg_renderer.enable_segmentation_rendering()

    labels_path = cfg.out / "labels.jsonl"
    n_labelled = 0
    occlusions = []

    with labels_path.open("w") as labels_file:
        for frame in range(cfg.frames):
            placed = scene.randomize(cfg, rng)

            renderer.update_scene(scene.data, camera="head_camera")
            rgb = renderer.render()
            seg_renderer.update_scene(scene.data, camera="head_camera")
            seg = seg_renderer.render()

            objects, robot_share = boxes_from_segmentation(
                seg, scene, placed, cfg.min_visible_px
            )
            occlusions.append(robot_share)
            n_labelled += len(objects)

            name = f"{frame:06d}.png"
            Image.fromarray(rgb).save(images_dir / name)
            labels_file.write(
                json.dumps(
                    {
                        "image": f"images/{name}",
                        "width": cfg.width,
                        "height": cfg.height,
                        "fovy_deg": float(model.cam_fovy[scene.camera_id]),
                        "robot_pixel_share": round(robot_share, 4),
                        "objects": objects,
                    }
                )
                + "\n"
            )

            if cfg.debug_overlay:
                overlay = Image.fromarray(rgb.copy())
                draw = ImageDraw.Draw(overlay)
                for obj in objects:
                    draw.rectangle(obj["bbox_xyxy"], outline=(0, 255, 0), width=2)
                    draw.text(
                        (obj["bbox_xyxy"][0] + 2, obj["bbox_xyxy"][1] + 2),
                        f"{obj['name']} {obj['range_m']:.2f}m",
                        fill=(0, 255, 0),
                    )
                overlay.save(overlay_dir / name)

            if (frame + 1) % 100 == 0:
                print(f"[render_pov] {frame + 1}/{cfg.frames} frames")

    share = float(np.mean(occlusions))
    print(f"[render_pov] wrote {cfg.frames} frames and {n_labelled} boxes to {cfg.out}")
    print(
        f"[render_pov] the robot's own geometry (mostly the beak) covers "
        f"{share * 100:.1f}% of the frame on average "
        f"(min {min(occlusions) * 100:.1f}%, max {max(occlusions) * 100:.1f}%)"
    )
    if share > 0.5:
        print(
            "[render_pov] WARNING: over half the frame is the robot itself. Check the "
            "camera pose before generating a large dataset."
        )


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))
