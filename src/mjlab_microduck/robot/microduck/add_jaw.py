#!/usr/bin/env python3
"""Articulate the beak of an onshape-to-robot MJCF export into a driven jaw.

The Onshape export puts the whole head assembly on a single rigid body
(``jaw_soft``, whose only DOF is ``head_roll``), with the yellow bill as a pair
of static ``jaw`` mesh geoms.  This script splits those geoms onto a new child
body hinged about the head's pitch axis and adds the matching XL330 actuator,
turning the 14-servo v1 robot into the 15-servo v2 robot::

    jaw_soft (head_roll)
      └── jaw_lower (jaw)          <- new hinge, actuated
            geom jaw (visual + collision)
            site jaw_tip

Frame notes (measured on the STAND keyframe, not guessed):  inside ``jaw_soft``
local +Y is the world pitch axis, local -Z points forward out of the beak, and
local +X points up.  The bill mesh spans z in [-0.078, -0.009] and x in
[-0.017, +0.012], and the upper mouth surface (``soft_mouth_top``) sits at
x ~= -0.007.  The hinge therefore goes at the rear of the mouth line and a
POSITIVE jaw angle swings the tip downward (open); 0 is closed, which keeps
the v1 HOME pose valid for every other joint.

Joint ordering matters.  MuJoCo orders joints by body-tree position, so the jaw
lands at index 9, between ``head_roll`` (8) and ``right_hip_yaw`` (now 10).
The v1 task configs hardcode ``_LEG_JOINTS = [0..4, 9..13]``, which is why this
is emitted as a SEPARATE model file rather than modifying the v1 robot: the v1
tasks keep using the v1 XML and are unaffected.  The actuator is inserted in
the same position so ctrl index still equals joint index.

Mass bookkeeping: the jaw body takes ``--jaw-mass`` kg and the same amount is
subtracted from ``jaw_soft`` so total robot mass is unchanged (the sim mass is
matched to the real robot).  ``jaw_soft``'s inertia tensor is left as-is, which
slightly over-states the head's rotational inertia -- refine alongside a real
CAD split if the head dynamics need it.

Usage::

    python3 add_jaw.py robot_allcollisions_jaw.xml
"""

import argparse
import re
import sys

JAW_MARKER = "<!-- Part jaw -->"
GEOM_JAW_RE = re.compile(r'^\s*<geom\b[^>]*mesh="jaw"[^>]*/>\s*$')
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
INERTIAL_RE = re.compile(r'^(\s*)<inertial\b(.*)/>\s*$')

# Hinge pivot in the jaw_soft local frame (m): rear of the mouth line.
PIVOT = (-0.007, 0.0, -0.020)
# Beak tip in the jaw_soft local frame, from the mouth_tip site.
TIP = (-0.00809334, 0.0, -0.0777383)


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def _vec(v) -> str:
    return " ".join(_fmt(x) for x in v)


def build_jaw_body(geom_lines: list[str], indent: str, mass: float,
                   lo: float, hi: float) -> list[str]:
    """Emit the ``jaw_lower`` body, re-basing the bill geoms onto its origin."""
    inner = indent + "  "
    out = [
        f"{indent}<!-- Part jaw -> articulated lower jaw (add_jaw.py). Hinge at the\n",
        f"{indent}     rear of the mouth line about the head pitch axis; +angle = open. -->\n",
        f'{indent}<body name="jaw_lower" pos="{_vec(PIVOT)}">\n',
        f'{inner}<joint axis="0 1 0" name="jaw" type="hinge"'
        f' range="{_fmt(lo)} {_fmt(hi)}" class="chosen_actuator"/>\n',
    ]

    # Bill inertia, approximated as a uniform box over the mesh bounding box
    # (0.030 x 0.091 x 0.069 m) about its centroid.
    ixx = mass * (0.091 ** 2 + 0.069 ** 2) / 12.0
    iyy = mass * (0.030 ** 2 + 0.069 ** 2) / 12.0
    izz = mass * (0.030 ** 2 + 0.091 ** 2) / 12.0
    com = (-0.0027 - PIVOT[0], 0.0, -0.0434 - PIVOT[2])
    out.append(
        f'{inner}<inertial pos="{_vec(com)}" mass="{_fmt(mass)}"'
        f' diaginertia="{_fmt(ixx)} {_fmt(iyy)} {_fmt(izz)}"/>\n'
    )

    for line in geom_lines:
        attrs = dict(ATTR_RE.findall(line))
        pos = [float(x) for x in attrs.get("pos", "0 0 0").split()]
        rebased = [pos[i] - PIVOT[i] for i in range(3)]
        line = re.sub(r'pos="[^"]*"', f'pos="{_vec(rebased)}"', line)
        # Name the contact surface so friction/condim can be targeted by regex
        # (mjlab's CollisionCfg matches on geom NAME; the Onshape export leaves
        # every non-foot collision geom unnamed).
        if attrs.get("class") == "collision":
            line = line.replace("<geom ", '<geom name="jaw_collision" ', 1)
        out.append(line.replace(indent, inner, 1))

    tip = [TIP[i] - PIVOT[i] for i in range(3)]
    out.append(f'{inner}<site group="3" name="jaw_tip" pos="{_vec(tip)}"/>\n')
    out.append(f"{indent}</body>\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", help="MJCF file to modify in place")
    parser.add_argument("--jaw-mass", type=float, default=0.010,
                        help="lower-jaw mass in kg, subtracted from jaw_soft "
                             "so total mass is preserved (default: 0.010)")
    parser.add_argument("--open-rad", type=float, default=0.70,
                        help="max jaw opening in rad (default: 0.70 ~ 40 deg)")
    parser.add_argument("--close-rad", type=float, default=-0.05,
                        help="negative travel past closed, so the servo can "
                             "squeeze against a held object (default: -0.05)")
    args = parser.parse_args()

    with open(args.xml) as f:
        lines = f.readlines()

    if any('name="jaw"' in line and "<joint" in line for line in lines):
        print(f"[add_jaw] {args.xml} already has a jaw joint — aborting.")
        return 1

    out: list[str] = []
    i = 0
    n = len(lines)
    jaw_done = False
    actuator_done = False
    mass_done = False
    upper_done = False

    while i < n:
        line = lines[i]

        # 1. Replace the static bill geoms with the hinged jaw body.
        if not jaw_done and JAW_MARKER in line:
            indent = line[: len(line) - len(line.lstrip())]
            i += 1
            geom_lines = []
            while i < n and GEOM_JAW_RE.match(lines[i]):
                geom_lines.append(lines[i])
                i += 1
            if not geom_lines:
                print("[add_jaw] ERROR: found the jaw marker but no jaw geoms.")
                return 1
            out.extend(build_jaw_body(geom_lines, indent, args.jaw_mass,
                                      args.close_rad, args.open_rad))
            jaw_done = True
            continue

        # 2. Take the jaw's mass out of jaw_soft (total mass preserved). The
        #    inertial we want is the first one after the jaw_soft body opens.
        if not mass_done and '<body name="jaw_soft"' in line:
            out.append(line)
            i += 1
            while i < n and not INERTIAL_RE.match(lines[i]):
                out.append(lines[i])
                i += 1
            if i < n:
                m = re.search(r'mass="([^"]*)"', lines[i])
                if m is None:
                    print("[add_jaw] ERROR: jaw_soft inertial has no mass.")
                    return 1
                new_mass = float(m.group(1)) - args.jaw_mass
                if new_mass <= 0:
                    print("[add_jaw] ERROR: --jaw-mass exceeds jaw_soft mass.")
                    return 1
                out.append(
                    re.sub(r'mass="[^"]*"', f'mass="{new_mass:.6g}"', lines[i])
                )
                i += 1
            mass_done = True
            continue

        # 3. Name the upper-mouth contact surface. The bill closes against the
        #    underside of the head shell, so that geom needs targetable friction
        #    too -- without a name, CollisionCfg can't reach it.
        if (not upper_done and '<geom' in line and 'class="collision"' in line
                and 'mesh="bottom_head_shell"' in line):
            out.append(line.replace("<geom ", '<geom name="upper_mouth_collision" ', 1))
            upper_done = True
            i += 1
            continue

        out.append(line)

        # 4. Add the actuator right after head_roll, so ctrl index == joint index.
        if not actuator_done and '<position' in line and 'joint="head_roll"' in line:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(
                f'{indent}<position class="chosen_actuator" name="jaw" joint="jaw"/>\n'
            )
            actuator_done = True

        i += 1

    if not (jaw_done and actuator_done and mass_done and upper_done):
        print(f"[add_jaw] ERROR: incomplete edit (body={jaw_done} "
              f"actuator={actuator_done} mass={mass_done} upper_mouth={upper_done}).")
        return 1

    with open(args.xml, "w") as f:
        f.writelines(out)

    print(f"[add_jaw] articulated the bill of {args.xml}: jaw joint at index 9 "
          f"(range {args.close_rad:g}..{args.open_rad:g} rad), "
          f"{args.jaw_mass * 1000:g} g moved off jaw_soft.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
