"""Microduck v2 jaw-pick training environment.

Builds on ``make_microduck_ground_pick_env_cfg`` (crouch / mouth-down-to-ground
pick + return) and extends it to the v2 jaw contract (see
``scripts/graft_v2_jaw.py`` — the authoritative source):

    v1 obs  (61-D, shared family layout)                          slots
      base_ang_vel  (3) + projected_gravity (3) + joint_pos (14)
      + joint_vel (14) + last_action (14) + twist (3) + head (4) + body (6)  0..60
    v2 jaw add slots 61/62:
      obs[61] = jaw angle        (1 = open,   = -qpos_jaw / 0.52)
      obs[62] = jaw current mag  (unsigned actuator current, "contact force")
    v1 actions (14) → v2 (15):  act[0..13] = body joint positions,
                                act[14]     = jaw aperture (0 = closed default)

The physics model ``robot_allcollisions_jaw.xml`` carries the jaw hinge
``<joint name="jaw" ... range="-0.52 0.09"/>`` as the 15th joint AND its
``<position name="jaw" joint="jaw"/>`` actuator LAST — so the concatenated
policy action is [14 body, 1 jaw] with jaw at index 14, and the body obs
blocks stay exactly 14-wide on this model.

The constants agent's ``MICRODUCK_GROUND_PICK_JAW_ROBOT_CFG`` applies the
canonical BAM actuator to all 15 joints (the real mouth servo is a DXL XL330
M6), so the jaw gets the same xl330 m6 physics as the body; HOME_FRAME's
patterns don't match ``jaw``, so it falls through to its XML default 0 rad =
closed.

── v2 policy skeleton ────────────────────────────────────────────────────────
A v2 training checkpoint can be initialized by grafting the trained v1
ground-pick actor with ``scripts/graft_v2_actor.py`` (widen obs 61→63 with zero
columns, dormant jaw action row; ``graft_v2_jaw.py`` does the same for ONNX)
and passing the resulting actor state_dict as ``--agent.graft_from <path.pt>``.
The ``GraftOnPolicyRunner`` loads it before the first ``learn()`` step so PPO
starts from the v1 behavior with the jaw channel dormant.
"""

from copy import deepcopy
from dataclasses import dataclass, field

# Symmetry: inherited as OFF from the ground-pick module — the v2 jaw obs is
# asymmetric (jaw current is a contact-force read-out), and the 61D
# SYMMETRY_CFG ``_OBS_PERM`` does not cover the two new slots anyway.
ENABLE_SYMMETRY = False

# Phase constants stay in sync with the ground-pick factory — the jaw reward
# is gated on the SAME segmented phase profile (descent → hold → rise).
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_GROUND_PICK_JAW_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    DESCENT_END,
    HOLD_END,
    RISE_END,
    make_microduck_ground_pick_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# Jaw joint selector used by the two jaw obs terms and the jaw reward.
_JAW_ASSET_CFG = SceneEntityCfg("robot", joint_names=("jaw",))

# NOTE: existing v1 .pt checkpoints / the v1 ONNX export use the 61-D obs
# layout. The v2 obs contract below is 63-D; policies are NOT hot-swappable
# with v1 policies in the runtime — only the ONNX exported from a v2 run is.


def make_microduck_jaw_pick_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Microduck v2 jaw-pick environment configuration.

    Exactly mirrors ``make_microduck_ground_pick_env_cfg(play, rough)`` with
    the v2 jaw contract applied on top (see module docstring).
    """
    cfg = make_microduck_ground_pick_env_cfg(play=play, rough=rough)

    # ── 1. Robot — jaw model (15th actuator `jaw`, appended last) ────────────
    cfg.scene.entities["robot"] = MICRODUCK_GROUND_PICK_JAW_ROBOT_CFG

    # ── 2. Actions — split into TWO terms (body THEN jaw) ────────────────────
    # IMPORTANT INVARIANT: policy action must be [body(14) | jaw(1)] with jaw at
    # index 14. mjlab concatenates action terms in dict insertion order, so the
    # body term is declared FIRST and the jaw term LAST. Any reordering here
    # silently remaps act[14] in the runtime (graft_v2_jaw.py expects jaw last).
    # The inherited single ``joint_pos`` term (actuator_names=(r".*",)) would
    # match the jaw actuator too (giving 15 already) — replaced by the two
    # explicit terms below.
    cfg.actions = {
        # 14 body dims — every servo but the jaw (regex excludes `jaw$`).
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(r"^(?!jaw$).*",),
            scale=1.0,
            use_default_offset=True,  # target = default joint pos + action (as v1)
        ),
        # 1 jaw dim — plain XML position actuator appended last in the model.
        # scale 0.52 = jaw's full negative-travel from HOME; use_default_offset
        # means target 0 = the jaw's default (HOME) position = CLOSED, so an
        # all-zero jaw aperture (policy init / grafted-dormant) keeps the mouth
        # shut and PPO must discover the opening direction via reward.
        "jaw": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("jaw",),
            scale=0.52,
            use_default_offset=True,
        ),
    }

    # ── 3. Observations — v1 blocks stay 14-wide on the jaw model ────────────
    # The jaw joint is named `jaw` (NOT passive_*), so every v1 selector that
    # uses `^(?!passive_).*` would silently grow to 15 and shift the whole obs.
    # Re-pin the joint_pos/joint_vel selectors (both groups) to exclude jaw.
    body_joints = SceneEntityCfg("robot", joint_names=(r"^(?!passive_|jaw$).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            # Terms were deepcopied per group inside the ground-pick factory.
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(
                body_joints
            )

    # last_action obs must stay 14-D. mjlab's ``mdp.last_action`` concats ALL
    # action terms when action_name is None (would be 15 now); pin it to the
    # body term so the actions obs block keeps v1's 14-wide body slice.
    # (actor and critic share the base term object — one write covers both.)
    cfg.observations["actor"].terms["actions"].params["action_name"] = "joint_pos"

    # ── v2 obs contract: actor obs is exactly 63-D, jaw at slots 61/62 ───────
    # Insertion order = concatenation order. The ground-pick factory appends
    # head_command then body_command LAST, so the two jaw terms added below
    # land after body_command → slots 61 and 62:
    #   [0:3] base_ang_vel [3:6] projected_gravity [6:20] joint_pos(14)
    #   [20:34] joint_vel [34:48] last_action [48:51] twist [51:55] head
    #   [55:61] body [61] jaw_angle [62] jaw_current      = 63-D total.
    # jaw_angle is read from HOME (a clean hinge on the robot body) → no noise;
    # jaw_current is a Dynamixel current-sense readout → small uniform noise to
    # model sensor noise on the magnitude channel.
    for grp in ("actor", "critic"):
        cfg.observations[grp].terms["jaw_angle"] = ObservationTermCfg(
            func=microduck_mdp.jaw_angle_obs,
            params={"asset_cfg": deepcopy(_JAW_ASSET_CFG)},
        )
        cfg.observations[grp].terms["jaw_current"] = ObservationTermCfg(
            func=microduck_mdp.jaw_current_obs,
            params={"asset_cfg": deepcopy(_JAW_ASSET_CFG)},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

    # ── 4. Reward — jaw aperture tracking (gated on the shared phase profile)
    # Positive Gaussian on the jaw angle derived from the v1 phase command
    # (target obs 1 = open during the hold segment). Weight 2.0 in the same
    # league as the other task terms; phase constants imported from the
    # ground-pick module so the profile stays in sync.
    cfg.rewards["jaw_aperture"] = RewardTermCfg(
        func=microduck_mdp.jaw_aperture_phased_reward,
        weight=2.0,
        params={
            "command_name": "twist",
            "std": 0.15,
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
            "asset_cfg": deepcopy(_JAW_ASSET_CFG),
        },
    )

    # ── 5. Fix inherited reward indexing on the 15-servo model ───────────────
    # ground_pick_return_pose_legs slices the servo joint view with hardcoded
    # v1 indices [0..4, 9..13]. On the jaw model the jaw joint is mid-tree
    # (servo-view position 9), so those indices would reward driving the JAW to
    # the right-hip home pose during the rise — directly fighting jaw_aperture —
    # and drop the right ankle. Exclude the jaw by NAME so the legacy indices
    # keep their v1 meaning over the 14 body servos. (Neck terms are unaffected:
    # neck joints sit before the jaw at servo-view 5..8.)
    legs = cfg.rewards["ground_pick_return_pose_legs"]
    legs.params["servo_exclude_names"] = ("jaw",)

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

@dataclass
class MicroduckJawPickRlCfg(RslRlOnPolicyRunnerCfg):
    """RL runner config for the v2 jaw-pick task.

    Fields mirror ``MicroduckGroundPickRlCfg`` (an RslRlOnPolicyRunnerCfg
    *instance* in the ground-pick module — not subclassable), plus a custom
    ``graft_from`` field for a v2 actor state_dict (see GraftOnPolicyRunner).

    ``graft_from`` is a plain str so it survives ``dataclasses.asdict`` in
    mjlab/scripts/train.py and is configurable via ``--agent.graft_from``.
    """

    # actor/critic/algorithm copied from MicroduckGroundPickRlCfg.
    actor: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,  # matches velocity; normalizer baked into ONNX by export.py
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        )
    )
    critic: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        )
    )
    algorithm: RslRlPpoAlgorithmCfg = field(
        default_factory=lambda: PpoWithSymmetryCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
        )
    )
    wandb_project: str = "mjlab_microduck"
    experiment_name: str = "jaw_pick_v2"
    run_name: str = "jaw_pick_v2"
    save_interval: int = 250
    num_steps_per_env: int = 24
    max_iterations: int = 2000

    graft_from: str = ""
    """Path to a v2 ACTOR state_dict .pt (63-obs / 15-act). Empty = cold start.
    Produced by scripts/graft_v2_actor.py (widen v1 actor + zero-initialized jaw
    channel). Loaded by GraftOnPolicyRunner before the first learn() step."""


# NOTE: MicroduckGroundPickRlCfg is built as an RslRlOnPolicyRunnerCfg INSTANCE
# in the ground-pick module (not subclassable), so the actor/critic/algorithm
# fields above were copied field-for-field from that instance.
