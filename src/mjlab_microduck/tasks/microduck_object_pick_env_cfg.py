"""Microduck ObjectPick task — crouch, close the beak on an object, lift, hold.

This is the "targeted grab" half of the v2 pipeline, and the first task in the
repo that runs on the **v2 robot**: 15 servos (the 14 v1 joints plus an
actuated lower jaw at index 9) and therefore a **64D observation / 15 action**
contract. See tasks/v2.py. It is NOT hot-swappable with the 61D v1 policies;
the runtime has to load it against the v2 model.

What changed versus the earlier payload-force sketch
----------------------------------------------------
With a real jaw and the beak collision geoms, holding is no longer something
the reward has to model — the physics either holds the object or it does not.
So there is no payload force, no weld, and no stillness proxy standing in for
a grip. The reward's whole job is to say what counts as a pick.

The reward never mentions the jaw
---------------------------------
Nothing pays for closing the beak. A closure reward is farmed by chomping on
air within a few hundred iterations, because chomping is free and gripping is
hard. Closure only ever pays through ``object_lift_progress`` (the object
leaves the ground) and ``object_hold`` (it stays up, in the beak, with the
robot standing). Those three gates on the hold term are each blocking a
specific cheat: pushing the object along the floor, balancing it on the head,
and falling on top of it.

Physics risk, stated up front
-----------------------------
Rigid grasping in MuJoCo is finicky, and no reward fixes a beak that
physically cannot hold anything. The props carry high friction, condim 4 and
softened contacts, and the beak geoms are given matching friction
(JAW_COLLISION in microduck_constants), but the real gate is Joe's manual
teleop harness: if a human driving the jaw cannot pick the object up, do not
start a training run. Expect the sock to be the worst of the three — it is a
rigid box standing in for a deformable object.
"""


from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_JAW_ROBOT_CFG,
    MICRODUCK_OBJECT_CFGS,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Mirror-loss is off and stays off: the symmetry table in symmetry.py is
# hardcoded for the 61D v1 layout and would mis-permute a 64D observation.
# The task is asymmetric anyway.
ENABLE_SYMMETRY = False

OBJECT_NAMES = ("block", "rubber_ball", "sock")
BEAK_SENSOR_PREFIX = "beak_contact"

# MEASURED on this model with scripts/measure_v2_reach.py, not carried over
# from another env. The trunk settles at 116.6mm at HOME (the standup env's
# 0.115 is a different model revision), and the beak can reach every prop's
# grab height from a statically stable crouch somewhere in the 60-110mm band
# ahead of the trunk.
STAND_Z = 0.1166
OBJECT_OFFSET = (0.085, 0.0)
OBJECT_NOISE_XY = 0.02

# Held = off the ground by this much. Larger than the placement noise so that
# nudging the prop cannot be mistaken for lifting it.
HOLD_CLEARANCE = 0.02
# Posture gate for a hold: near standing height and roughly level.
HOLD_MIN_HEIGHT = 0.09
HOLD_TILT_LIMIT = 0.35
# The actual deliverable: hold it this long, continuously.
HOLD_REQUIRED_S = 3.0

# Long enough to crouch, grab, stand back up and still hold for 3s.
EPISODE_LENGTH_S = 12.0

# Noise on the target point handed to the policy. The real target comes from a
# detector, so training on a perfect one produces a policy that breaks on the
# first few-millimetre bounding-box error.
GRAB_TARGET_NOISE = 0.01


def make_microduck_object_pick_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    # Built on the velocity recipe for its DR / observation-noise / delay /
    # NaN-guard stack, not for its gait. Rebuilding those by hand is the
    # documented way to get a policy that only works in the viewer.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # === SCENE: v2 robot + all three props ===
    cfg.scene.entities = {
        "robot": MICRODUCK_JAW_ROBOT_CFG,
        **{name: MICRODUCK_OBJECT_CFGS[name] for name in OBJECT_NAMES},
    }
    cfg.grab_object_names = OBJECT_NAMES
    cfg.grab_target_noise = GRAB_TARGET_NOISE
    cfg.episode_length_s = EPISODE_LENGTH_S

    # Contact between either beak surface and each prop. `add_jaw.py` names the
    # two beak geoms specifically so this sensor can address them — every other
    # collision geom in the Onshape export is unnamed.
    #
    # One sensor per prop rather than one sensor against all three: a secondary
    # pattern is only treated as a regex when it is scoped to a single entity,
    # so "any prop" is not expressible in one sensor. mdp._beak_contact gathers
    # these by the active-prop index.
    beak_sensors = tuple(
        ContactSensorCfg(
            name=f"{BEAK_SENSOR_PREFIX}_{name}",
            primary=ContactMatch(
                mode="geom",
                pattern=r"^(jaw|upper_mouth)_collision$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="geom", pattern=r".*", entity=name),
            fields=("found",),
            reduce="none",
            num_slots=1,
        )
        for name in OBJECT_NAMES
    )
    # Head-into-the-floor impact, same sensor the ground-pick env uses. The beak
    # has to come all the way down to the floor here, and without a price on the
    # arrival force the fastest approach is a dive.
    head_impact_cfg = ContactSensorCfg(
        name="head_impact_contact",
        primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + beak_sensors + (head_impact_cfg,)
    # Three extra free bodies (prop-terrain and prop-robot contacts) on top of
    # the full-collision robot's budget. The ball-kick env raises this to 50 for
    # one extra body; three want more headroom.
    cfg.sim.nconmax = 80

    # === COMMANDS ===
    # This is a stationary manipulation task: no walking command. The twist slot
    # keeps a small non-zero range rather than being deleted, so its three input
    # neurons stay alive and the 64D layout is preserved for hot-swapping.
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-0.02, 0.02)
    twist.ranges.lin_vel_y = (-0.02, 0.02)
    twist.ranges.ang_vel_z = (-0.05, 0.05)
    twist.rel_standing_envs = 0.8
    twist.rel_turn_in_place_envs = 0.0

    # body_pose slots [0:3] are repurposed to carry the object's target point in
    # the robot frame — the same (dx, dy, dz) the detector produces. Slots [3:6]
    # keep their small sampled values so those neurons stay alive.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.grab_target_command,
            params={"command_name": "body_pose"},
        )

    # Asymmetric actor-critic: the actor only ever sees the (noisy) target point,
    # exactly as it will at deployment; the critic sees the truth so the value
    # function is not guessing at a state the actor cannot observe.
    cfg.observations["critic"].terms["object_position"] = ObservationTermCfg(
        func=microduck_mdp.object_pos_in_base,
    )
    cfg.observations["critic"].terms["object_velocity"] = ObservationTermCfg(
        func=microduck_mdp.object_vel_in_base,
    )
    cfg.observations["critic"].terms["object_lift"] = ObservationTermCfg(
        func=microduck_mdp.object_lift_obs,
    )

    # === REWARDS ===
    # Strip the gait rewards: they are all conditioned on a velocity command
    # that is now ~zero, and air_time in particular would pay for stepping
    # around while the task is to stand still and crouch.
    for name in (
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "track_linear_velocity",
        "track_angular_velocity",
    ):
        cfg.rewards.pop(name, None)

    # Leg pose reward pulls toward HOME, which fights the crouch the task
    # requires. Keep it weak rather than deleting it — it is what stops the
    # legs from splaying into unrecoverable poses.
    cfg.rewards["pose"].weight = 0.2
    # The velocity recipe's selector is "everything that isn't passive, neck or
    # head", which on v2 also picks up the jaw — an 11th joint against a 10-long
    # per-joint std tuple. Excluding it is right on the merits anyway: pulling
    # the jaw toward HOME (closed) is a standing penalty on opening it.
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|jaw$|.*neck.*|.*head.*).*",)
    )
    # upright is a task gate here, not just a style preference: standing back up
    # with the object is half the deliverable.
    cfg.rewards["upright"].weight = 1.0

    # --- the task ---
    cfg.rewards["object_lift"] = RewardTermCfg(
        func=microduck_mdp.object_lift_progress,
        weight=8.0,
        params={"max_paid_rate": 0.12, "target_lift": 0.06},
    )
    cfg.rewards["object_hold"] = RewardTermCfg(
        func=microduck_mdp.object_hold,
        weight=4.0,
        params={
            "sensor_prefix": BEAK_SENSOR_PREFIX,
            "clearance": HOLD_CLEARANCE,
            "min_height": HOLD_MIN_HEIGHT,
            "tilt_limit": HOLD_TILT_LIMIT,
        },
    )
    # Weight 0: logged as a real success criterion, never optimized. Total
    # reward can climb a long way on shaping alone while the pick never once
    # happens, and this is the term that says so.
    cfg.rewards["hold_success"] = RewardTermCfg(
        func=microduck_mdp.object_hold_success,
        weight=0.0,
        params={
            "sensor_prefix": BEAK_SENSOR_PREFIX,
            "clearance": HOLD_CLEARANCE,
            "min_height": HOLD_MIN_HEIGHT,
            "tilt_limit": HOLD_TILT_LIMIT,
            "required_s": HOLD_REQUIRED_S,
        },
    )
    # Approach shaping. Random exploration essentially never puts a beak on a
    # 26mm block, so without this the lift term is never once triggered and
    # there is nothing to learn from. Switches off as soon as the object is up.
    cfg.rewards["mouth_to_object"] = RewardTermCfg(
        func=microduck_mdp.mouth_to_object,
        weight=2.0,
        params={
            # Measured: the beak starts ~0.21m from the prop, so the coarse
            # scale is what makes the descent visible at all.
            "std": 0.04,
            "coarse_std": 0.13,
            "max_lift": 0.01,
            # Passed explicitly, not left to the default: the managers only
            # resolve site_names on a SceneEntityCfg that appears in params.
            "asset_cfg": SceneEntityCfg("robot", site_names=["jaw_tip"]),
        },
    )

    # --- penalties ---
    # SELF-NEGATING (returns <= 0) -> POSITIVE weight. A negative weight here
    # double-negates into a reward for knocking the object away, which is the
    # exact failure it exists to price. Every Episode_Reward/<penalty> in wandb
    # must read <= 0; if this one reads positive, the sign is wrong.
    cfg.rewards["object_disturb"] = RewardTermCfg(
        func=microduck_mdp.object_disturb_penalty,
        weight=0.0,  # ramped in by curriculum once the approach exists
        params={"max_lift": 0.01, "tolerance": 0.02, "scale": 10.0},
    )
    # mjlab-base COST style: returns >= 0, so this one takes a NEGATIVE weight.
    # Both conventions live in mdp.py and the only reliable check is the wandb
    # one — every Episode_Reward/<penalty> must read <= 0.
    #
    # Kept deliberately mild. The ground-pick env runs this at -2.0 because it
    # wants NO contact at all; here the beak must reach the floor to get under
    # the object, and the roller-standup env is the cautionary tale — a head
    # impact penalty at -1.0 there became the largest negative term on the board
    # and the policy converged to lying still rather than risk it.
    cfg.rewards["head_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-0.2,
        params={"sensor_name": "head_impact_contact", "threshold": 2.0},
    )

    # === EVENTS ===
    # Registered AFTER reset_base: it reads the robot root from qpos, so the
    # base pose has to be final first (events run in dict insertion order).
    cfg.events["reset_grab_object"] = EventTermCfg(
        func=microduck_mdp.reset_grab_object,
        mode="reset",
        params={
            "offset": OBJECT_OFFSET,
            "noise_xy": OBJECT_NOISE_XY,
        },
    )
    # The robot starts standing, in front of the object — this task begins where
    # GoToPoint ends.
    cfg.events["reset_base"].params["pose_range"]["z"] = (STAND_Z - 0.005, STAND_Z + 0.005)
    # Pushes fight a delicate manipulation; keep them, but gentle.
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
        }

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === CURRICULUM ===
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)

    # Both of these are taxes, and an attempt-tax that is active while a hard
    # skill is still being discovered makes "do nothing" the winning policy.
    # Neither is introduced until the approach exists.
    cfg.curriculum["object_disturb_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "object_disturb",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 300 * NUM_STEPS_PER_ENV, "weight": 0.5},
                {"step": 800 * NUM_STEPS_PER_ENV, "weight": 1.0},
            ],
        },
    )
    # Approach shaping fades once the lift term can carry the policy on its own;
    # left at full weight it competes with lifting (the beak is closest to the
    # object when the object is still on the ground).
    cfg.curriculum["mouth_to_object_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "mouth_to_object",
            "weight_stages": [
                {"step": 0, "weight": 2.0},
                {"step": 600 * NUM_STEPS_PER_ENV, "weight": 1.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 0.5},
            ],
        },
    )

    return cfg


MicroduckObjectPickRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
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
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="object_pick",
    run_name="object_pick",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    # An episodic trick on a stationary base; budget like the other tricks, not
    # like a gait.
    max_iterations=2_000,
)
