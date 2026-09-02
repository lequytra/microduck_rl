"""The v2 policy family contract: 64-D observation, 15-D action.

v1 policies are 61-D / 14 actions.  v2 adds an actuated lower jaw, which adds
one joint to ``joint_pos``, ``joint_vel`` and ``last_action`` -- three slots,
not one -- so the observation grows by 3, from 61 to **64**::

    [0:3]    base_ang_vel       (roll, pitch, yaw -- body-frame IMU)
    [3:6]    projected_gravity  (body-frame)
    [6:21]   joint_pos_rel      (15 joints, relative to HOME)
    [21:36]  joint_vel_rel      (15 joints)
    [36:51]  last_action        (15 joints)
    [51:54]  twist command
    [54:58]  head_pose command
    [58:64]  body_pose command

The 13-D command block is byte-identical to v1 and the 14 original joints keep
their relative order, so transfer from a v1 checkpoint is a matter of copying
the shared rows and initialising the three new input rows and one new output
row near zero.

Joint ordering
--------------
MuJoCo orders joints by body-tree position, and the jaw hangs off ``jaw_soft``
(whose joint is ``head_roll``, index 8).  The jaw therefore lands at index 9 and
pushes the right leg to 10-14::

    0-4    left leg   (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8    neck/head  (neck_pitch, head_pitch, head_yaw, head_roll)
    9      jaw
    10-14  right leg

This is exactly why v2 is a separate robot model: six v1 configs hardcode
``_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]``, and on a v2 model those
literals would silently address the jaw and the wrong leg joints instead of
crashing.  v2 configs must therefore resolve indices through the helpers below
rather than writing literals, and ``tests/test_v2_jaw_model.py`` asserts
``V2_JOINT_NAMES`` still matches the compiled model.
"""

from __future__ import annotations

V2_OBS_DIM = 64
V2_ACTION_DIM = 15

# Canonical v2 joint order == ctrl order (add_jaw.py inserts the jaw actuator
# in the matching position, and the test asserts both against the real model).
V2_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "jaw",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

JAW_JOINT_NAME = "jaw"
# Jaw travel, in sync with add_jaw.py's defaults. Negative is "squeeze past
# closed" so the position servo can keep loading a held object.
JAW_CLOSED = 0.0
JAW_OPEN = 0.70
JAW_MIN = -0.05


def joint_indices(*names: str) -> list[int]:
    """Indices of ``names`` in the v2 joint layout.

    Use this instead of index literals so that a future joint insertion breaks
    loudly (KeyError) rather than silently re-pointing a reward at the wrong
    joints -- the failure mode that adding the jaw would have caused in v1.
    """
    lookup = {name: i for i, name in enumerate(V2_JOINT_NAMES)}
    missing = [n for n in names if n not in lookup]
    if missing:
        raise KeyError(f"not v2 joints: {missing}; known: {V2_JOINT_NAMES}")
    return [lookup[n] for n in names]


def joints_matching(*substrings: str) -> list[int]:
    """Indices of every v2 joint whose name contains any of ``substrings``."""
    return [
        i
        for i, name in enumerate(V2_JOINT_NAMES)
        if any(s in name for s in substrings)
    ]


LEG_JOINTS: list[int] = joints_matching("hip", "knee", "ankle")
NECK_JOINTS: list[int] = joints_matching("neck", "head")
JAW_JOINTS: list[int] = joint_indices(JAW_JOINT_NAME)
# Everything the jaw shouldn't disturb -- used by the posture/pose rewards so a
# grab never pays by contorting the body.
BODY_JOINTS: list[int] = LEG_JOINTS + NECK_JOINTS
