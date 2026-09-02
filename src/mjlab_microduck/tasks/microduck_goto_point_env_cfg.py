"""Microduck GoToPoint task — walk to a commanded (dx, dy, yaw) and stop there.

This is the "targeted walk" half of the v2 pipeline: the object detector turns
a bounding box into an offset, and this policy drives the robot to it and holds
still, facing the object, so the grab policy can take over.

Built on ``make_microduck_velocity_env_cfg()`` rather than from scratch, which
is not a shortcut so much as a requirement: that recipe carries the whole
domain-randomization stack, the observation noise and delay models, the
BAM friction expansion, the NaN guards and the encoder bias. Rebuilding a
locomotion env without them produces a policy that walks in the viewer and
falls over on hardware.

The one structural change from the velocity recipe
--------------------------------------------------
``RelativeGoalPositionCommand`` splits what the policy SEES from what the gait
rewards TRACK:

* the twist observation slot carries ``[dx_body, dy_body, yaw_error]`` — the
  goal, in metres, which is what the perception stack can actually supply;
* ``vel_command_b`` carries a velocity setpoint derived from that goal and
  clamped to the ranges the gait was tuned on, so ``track_linear_velocity``,
  ``track_angular_velocity``, ``air_time``, ``foot_clearance`` and ``foot_slip``
  all keep working untouched.

Deleting the velocity tracking rewards and replacing them with a pure
position objective was the obvious alternative and the wrong one: gait quality
in this repo lives almost entirely in those terms. Slowing down near the goal
then costs nothing extra — the setpoint tapers with distance on its own.

Obs is the shared 61D layout, so this policy stays hot-swappable with the
existing walk/stand policies at runtime.
"""


from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

# Mirror-loss is meaningless here: reflecting the robot without also reflecting
# the goal turns "walk to the point on my left" into a wrong label.
ENABLE_SYMMETRY = False

# How far the goal can be. The upper bound grows by curriculum; a policy that
# cannot yet walk 30cm learns nothing useful from a 1.2m goal.
GOAL_RADIUS_START = (0.20, 0.50)
GOAL_RADIUS_FINAL = (0.20, 1.20)

# Inside ARRIVAL_RADIUS the steering switches from "head toward the goal" to
# "adopt the commanded heading", and the arrival reward starts paying.
# Inside STOP_RADIUS the velocity setpoint is exactly zero.
ARRIVAL_RADIUS = 0.20
STOP_RADIUS = 0.05

# Goals resample a few times per 20s episode, so each episode contains several
# approach-and-stop cycles instead of one approach and 16s of camping.
GOAL_RESAMPLE_S = (6.0, 10.0)


def make_microduck_goto_point_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # === COMMAND ===
    # Rebuild the twist term as a goal command, keeping the velocity ranges: they
    # are the envelope the internal setpoint is clamped into, so the gait is
    # never asked for a speed it was not trained at.
    twist = cfg.commands["twist"]
    cfg.commands["twist"] = microduck_mdp.RelativeGoalPositionCommandCfg(
        **{
            **vars(twist),
            "resampling_time_range": GOAL_RESAMPLE_S,
            "goal_radius_range": GOAL_RADIUS_START,
            "arrival_radius": ARRIVAL_RADIUS,
            "stop_radius": STOP_RADIUS,
            "approach_gain": 1.2,
            "max_speed": 0.35,
            "turn_gain": 1.5,
            "rel_arrived_envs": 0.10,
            "rel_behind_envs": 0.15,
        }
    )
    # Standing envs are the arrived bucket's job here; leaving the base
    # mechanism on would zero the setpoint for envs that are NOT at the goal.
    cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.commands["twist"].rel_turn_in_place_envs = 0.0

    # === OBSERVATION ===
    # Show the policy the GOAL, not the setpoint. Handing it the setpoint would
    # be handing it the answer — it would learn to follow a velocity it is
    # already given and never learn to close the position loop itself.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["command"] = ObservationTermCfg(
            func=microduck_mdp.goal_position_command,
            params={"command_name": "twist"},
        )

    # === REWARDS ===
    # Potential-based: approaching pays, holding station pays exactly zero.
    # Rate-capped so charging the goal cannot collect the whole potential in
    # two steps and buy itself the violence to do it.
    cfg.rewards["goal_progress"] = RewardTermCfg(
        func=microduck_mdp.goal_progress,
        weight=5.0,
        params={"command_name": "twist", "max_paid_rate": 0.4},
    )
    # Product, not sum: a sum lets the policy bank ~80% of every factor by
    # drifting through the goal at an angle, still moving. The stds are loose
    # on purpose — four tight Gaussians multiply to zero everywhere and give
    # no gradient at all.
    cfg.rewards["goal_arrival"] = RewardTermCfg(
        func=microduck_mdp.goal_arrival_composite,
        weight=1.0,  # ramped to 3.0 by the curriculum below
        params={
            "command_name": "twist",
            "dist_std": 0.08,
            "yaw_std": 0.35,
            "still_std": 0.15,
            "upright_std": 0.25,
        },
    )
    # SELF-NEGATING penalty (returns <= 0) -> POSITIVE weight. A negative weight
    # would double-negate into a reward for barrelling through the goal, which
    # is the exact behaviour this exists to stop. Ramped from 0: an attempt-tax
    # applied while the policy is still learning to reach goals at all makes
    # "never approach" the winning strategy.
    cfg.rewards["goal_overshoot"] = RewardTermCfg(
        func=microduck_mdp.goal_overshoot_penalty,
        weight=0.0,
        params={"command_name": "twist", "speed_ref": 0.15},
    )

    # === CURRICULUM ===
    # command_vel widens the velocity ranges over training; here those ranges
    # are a fixed clamp on an internally computed setpoint, so widening them
    # would silently let the goal command outrun the trained gait.
    cfg.curriculum.pop("command_vel", None)

    cfg.curriculum["goal_command"] = CurriculumTermCfg(
        func=microduck_mdp.goal_command_curriculum,
        params={
            "command_name": "twist",
            # Phase-aligned with what the policy can actually do: near goals and
            # almost no rear-cone spawns first, distance and turning added only
            # once approaching works at all.
            "stages": [
                {
                    "step": 0,
                    "goal_radius_range": GOAL_RADIUS_START,
                    "rel_behind_envs": 0.05,
                    "rel_arrived_envs": 0.10,
                },
                {
                    "step": 800 * NUM_STEPS_PER_ENV,
                    "goal_radius_range": (0.20, 0.80),
                    "rel_behind_envs": 0.12,
                    "rel_arrived_envs": 0.12,
                },
                {
                    "step": 2000 * NUM_STEPS_PER_ENV,
                    "goal_radius_range": GOAL_RADIUS_FINAL,
                    "rel_behind_envs": 0.20,
                    "rel_arrived_envs": 0.15,
                },
            ],
        },
    )
    cfg.curriculum["goal_arrival_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "goal_arrival",
            # Early on, getting there at all is the skill; precision at the goal
            # only becomes the objective once the approach exists.
            "weight_stages": [
                {"step": 0, "weight": 1.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 2.0},
                {"step": 2500 * NUM_STEPS_PER_ENV, "weight": 3.0},
            ],
        },
    )
    cfg.curriculum["goal_overshoot_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "goal_overshoot",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 0.3},
                {"step": 3000 * NUM_STEPS_PER_ENV, "weight": 0.5},
            ],
        },
    )

    return cfg


MicroduckGoToPointRlCfg = RslRlOnPolicyRunnerCfg(
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
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="goto_point",
    run_name="goto_point",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    # A gait is the expensive kind of task; budget like the velocity recipe.
    max_iterations=6_000,
)
