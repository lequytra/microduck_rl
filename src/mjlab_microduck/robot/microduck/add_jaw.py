#!/usr/bin/env python3
"""Generate robot_allcollisions_jaw.xml: the allcollisions model + an ACTUATED
jaw hinge, mirroring add_backlash.py's generator pattern.

Surgery:
- The `jaw` (visual + collision) and `soft_mouth_top` geoms move from the
  rigid `jaw_soft` (head) body into a new child body `jaw` with a hinge joint.
  Hinge 0 rad = closed, +0.52 rad = open (hardware: -5 deg..+30 deg on DXL #34).
- The jaw ACTUATOR is appended last in the actuator list, so the 14 body
  actuators keep indices 0-13 and every actuator-order-driven obs/action
  mapping is preserved. (qpos order shifts mid-tree — MuJoCo numbers joints in
  document order — so all joint reads MUST go through actuator-trnid lookup or
  joint names, never raw qpos offsets. infer_policy.py already does this.)

Usage: python add_jaw.py   (writes robot_allcollisions_jaw.xml next to the source)
"""
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "robot_allcollisions.xml"
DST = HERE / "robot_allcollisions_jaw.xml"

JAW_MESHES = {"jaw", "soft_mouth_top"}
JAW_RANGE = (-0.52, 0.09)   # rad; 0 = closed, NEGATIVE = open (verified visually:
                            # +rotation tucks the beak under the head)
JAW_MASS = 0.010            # 10 g lower beak

tree = ET.parse(SRC)
root = tree.getroot()

jaw_soft = root.find(".//body[@name='jaw_soft']")
assert jaw_soft is not None, "jaw_soft body not found"

# Detach the moving geoms from the head body
moved = [g for g in jaw_soft.findall("geom")
         if g.get("mesh") in JAW_MESHES]
assert len(moved) == 3, f"expected 3 jaw geoms, found {len(moved)}"
for g in moved:
    jaw_soft.remove(g)

# Pivot: the jaw visual geom's own frame (its pos/quat become the body's;
# geoms go to identity so the closed pose is pixel-identical to the rigid model)
jaw_visual = next(g for g in moved if g.get("mesh") == "jaw" and g.get("class") == "visual")
pivot_pos = jaw_visual.get("pos")
pivot_quat = jaw_visual.get("quat")

jaw_body = ET.SubElement(jaw_soft, "body", {"name": "jaw", "pos": pivot_pos, "quat": pivot_quat})
ET.SubElement(jaw_body, "joint", {
    "name": "jaw", "type": "hinge", "axis": "0 0 1",
    "range": f"{JAW_RANGE[0]} {JAW_RANGE[1]}",
})
ET.SubElement(jaw_body, "inertial", {
    "pos": "0 0 -0.01", "mass": str(JAW_MASS),
    "diaginertia": "1e-6 1e-6 1e-6",
})
for g in moved:
    g.set("pos", "0 0 0" if g is jaw_visual else g.get("pos"))
    g.set("quat", "1 0 0 0" if g is jaw_visual else g.get("quat"))
    if g is not jaw_visual:
        # keep each secondary geom's small offset relative to the pivot
        g.set("pos", g.get("pos"))  # offset computed below
    jaw_body.append(g)

# Fix the secondary geom offsets: they were expressed in jaw_soft's frame, so
# re-express relative to the pivot (same orientation -> simple subtraction)
import numpy as np
pv = np.array([float(v) for v in pivot_pos.split()])
for g in moved:
    if g is jaw_visual:
        continue
    old = np.array([float(v) for v in g.get("pos").split()])
    g.set("pos", " ".join(f"{v:.8g}" for v in (old - pv)))
    # quats differ slightly per geom; offsets are ~0.1 mm, rotation error negligible

# Append the jaw actuator LAST (body actuators keep 0-13)
actuator = root.find("actuator")
ET.SubElement(actuator, "position",
              {"class": "chosen_actuator", "name": "jaw", "joint": "jaw"})

ET.indent(tree, space="  ")
tree.write(DST, xml_declaration=True, encoding="unicode")
print(f"wrote {DST}")
