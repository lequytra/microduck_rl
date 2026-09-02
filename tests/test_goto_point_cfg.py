"""GoToPoint: config invariants plus the command's setpoint behaviour.

The interesting claim in this env is that the policy SEES a goal offset while
the gait rewards TRACK an internally derived velocity. If those two ever get
crossed — the observation showing the setpoint, or the gait tracking the raw
goal — the env still builds and still trains, it just trains the wrong thing.
The command tests below pin the split down.
"""

import math
import types

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_goto_point_env_cfg import (
    ARRIVAL_RADIUS,
    ENABLE_SYMMETRY,
    GOAL_RADIUS_FINAL,
    GOAL_RADIUS_START,
    STOP_RADIUS,
    MicroduckGoToPointRlCfg,
    make_microduck_goto_point_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)

# These are where gait quality lives. The temptation when switching to a
# position objective is to delete them and reward distance directly; that
# throws away all of the velocity recipe's tuning.
GAIT_REWARDS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "air_time",
    "foot_clearance",
    "foot_slip",
)


@pytest.fixture(scope="module")
def cfg():
    return make_microduck_goto_point_env_cfg()


# ── config ───────────────────────────────────────────────────────────────────
def test_env_builds_train_play_and_rough():
    assert make_microduck_goto_point_env_cfg() is not None
    assert make_microduck_goto_point_env_cfg(play=True) is not None
    assert make_microduck_goto_point_env_cfg(rough=True) is not None


def test_gait_rewards_survive(cfg):
    for name in GAIT_REWARDS:
        assert name in cfg.rewards, f"gait reward deleted: {name}"


def test_command_is_the_goal_term(cfg):
    assert isinstance(cfg.commands["twist"], mdp.RelativeGoalPositionCommandCfg)


def test_policy_sees_the_goal_not_the_setpoint(cfg):
    # Showing it vel_command_b would hand it the answer: it would learn to
    # follow a velocity it is already given and never close the position loop.
    for group in ("actor", "critic"):
        assert cfg.observations[group].terms["command"].func is mdp.goal_position_command


def test_obs_layout_is_unchanged_from_the_velocity_recipe(cfg):
    # Same 61D layout term-for-term, so the ONNX still loads in the runtime's
    # walk slot. The goal command replaces the twist slot's CONTENTS, not its
    # width: [dx, dy, yaw_error] is 3 wide, exactly like a twist.
    velocity = make_microduck_velocity_env_cfg()
    for group in ("actor", "critic"):
        assert list(cfg.observations[group].terms.keys()) == list(
            velocity.observations[group].terms.keys()
        ), f"observation layout diverged on {group}"


def test_stop_radius_is_inside_the_arrival_radius(cfg):
    # Arrival is where steering switches from bearing to goal heading; stop is
    # where the setpoint hits exactly zero. Inverted, the robot would be told
    # to stop before it had turned to face the right way.
    assert 0.0 < STOP_RADIUS < ARRIVAL_RADIUS


def test_goal_radius_only_grows(cfg):
    # A policy that cannot walk 30cm learns nothing from a 1.2m goal.
    assert GOAL_RADIUS_START[1] < GOAL_RADIUS_FINAL[1]
    stages = cfg.curriculum["goal_command"].params["stages"]
    radii = [s["goal_radius_range"][1] for s in stages]
    assert radii == sorted(radii)
    assert stages[0]["step"] == 0
    assert stages[0]["goal_radius_range"] == GOAL_RADIUS_START
    assert stages[-1]["goal_radius_range"] == GOAL_RADIUS_FINAL


def test_both_rare_spawn_buckets_exist_and_grow(cfg):
    # Uniform sampling essentially never produces "already at the goal", and
    # rear-cone goals are a distinct skill (turn around first) that gets a
    # sliver of experience otherwise.
    stages = cfg.curriculum["goal_command"].params["stages"]
    for key in ("rel_arrived_envs", "rel_behind_envs"):
        shares = [s[key] for s in stages]
        assert all(share > 0.0 for share in shares), f"{key} bucket is empty"
        assert shares == sorted(shares)


def test_base_standing_mechanism_is_off(cfg):
    # Standing is a CONSEQUENCE of being at the goal here. The base class's
    # standing mask would zero the setpoint for envs that are NOT at the goal.
    assert cfg.commands["twist"].rel_standing_envs == 0.0


def test_velocity_curriculum_is_removed(cfg):
    # Those ranges are now a clamp on an internally computed setpoint; widening
    # them would let the goal command outrun the gait it was tuned for.
    assert "command_vel" not in cfg.curriculum


def test_overshoot_penalty_takes_a_positive_weight_and_ramps_from_zero(cfg):
    # Self-negating (returns <= 0). A negative weight double-negates into a
    # reward for barrelling through the goal.
    stages = cfg.curriculum["goal_overshoot_weight"].params["weight_stages"]
    weights = [s["weight"] for s in stages]
    assert all(w >= 0.0 for w in weights)
    assert weights[0] == 0.0 == cfg.rewards["goal_overshoot"].weight
    assert weights == sorted(weights)


def test_arrival_weight_ramps_up_after_the_approach_exists(cfg):
    stages = cfg.curriculum["goal_arrival_weight"].params["weight_stages"]
    weights = [s["weight"] for s in stages]
    assert weights[0] == cfg.rewards["goal_arrival"].weight
    assert weights == sorted(weights)
    assert weights[-1] > weights[0]


def test_symmetry_is_off():
    # Mirroring the robot without mirroring the goal turns "walk to the point
    # on my left" into a wrong label.
    assert ENABLE_SYMMETRY is False
    assert MicroduckGoToPointRlCfg.algorithm.symmetry_cfg is None


def test_budgeted_like_a_gait():
    assert MicroduckGoToPointRlCfg.experiment_name == "goto_point"
    assert MicroduckGoToPointRlCfg.max_iterations >= 4_000


def test_tasks_are_registered():
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    assert "Mjlab-GoToPoint-Flat-MicroDuck" in tasks
    assert "Mjlab-GoToPoint-Rough-MicroDuck" in tasks


# ── the command's setpoint, driven directly ──────────────────────────────────
class _Command:
    """RelativeGoalPositionCommand with __init__ and the base class bypassed.

    Only `_update_command` is under test, and it reads nothing but the robot
    root pose, the goal, and cfg — so a hand-built instance exercises the real
    code path without a MuJoCo scene.
    """

    def __new__(cls, *, goal_xy, goal_yaw, pos_xy, heading, cfg, qpos_xy=None):
        self = object.__new__(mdp.RelativeGoalPositionCommand)
        n = 1
        self.cfg = cfg
        # `device` and `num_envs` are read-only properties that defer to _env.
        # qpos carries the AUTHORITATIVE root pose during a reset; the tests
        # below deliberately let it disagree with root_link_pos_w.
        qx, qy = qpos_xy if qpos_xy is not None else pos_xy
        qpos = torch.zeros(n, 13)
        qpos[:, 0], qpos[:, 1], qpos[:, 2] = qx, qy, 0.115
        qpos[:, 3] = math.cos(heading / 2.0)
        qpos[:, 6] = math.sin(heading / 2.0)
        self._env = types.SimpleNamespace(
            device="cpu",
            num_envs=n,
            sim=types.SimpleNamespace(data=types.SimpleNamespace(qpos=qpos)),
        )
        self._goal_pos_w = torch.tensor([goal_xy], dtype=torch.float32)
        self._goal_yaw_w = torch.tensor([goal_yaw], dtype=torch.float32)
        self.goal_error_b = torch.zeros(n, 3)
        self.distance = torch.zeros(n)
        self.vel_command_b = torch.zeros(n, 3)
        self.vel_command_w = torch.zeros(n, 3)
        data = types.SimpleNamespace(
            root_link_pos_w=torch.tensor([[*pos_xy, 0.115]], dtype=torch.float32),
            heading_w=torch.tensor([heading], dtype=torch.float32),
        )
        self.robot = types.SimpleNamespace(
            data=data,
            indexing=types.SimpleNamespace(free_joint_q_adr=list(range(7))),
        )
        return self


def _cmd(goal_xy=(1.0, 0.0), goal_yaw=0.0, pos_xy=(0.0, 0.0), heading=0.0):
    cfg = make_microduck_goto_point_env_cfg().commands["twist"]
    command = _Command(
        goal_xy=goal_xy, goal_yaw=goal_yaw, pos_xy=pos_xy, heading=heading, cfg=cfg
    )
    command._update_command()
    return command


def test_goal_error_is_in_the_body_frame():
    # Goal is due north in world; robot faces east. In its own frame the goal
    # is to the LEFT (+y), not ahead.
    command = _cmd(goal_xy=(0.0, 1.0), heading=0.0)
    assert command.goal_error_b[0, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert command.goal_error_b[0, 1].item() == pytest.approx(1.0, abs=1e-5)


def test_observed_offset_is_clipped_but_the_distance_is_not():
    # A far goal must not saturate the observation normalizer; the setpoint and
    # the progress reward still need the true distance.
    command = _cmd(goal_xy=(50.0, 0.0))
    assert command.goal_error_b[0, 0].item() == pytest.approx(command.cfg.obs_clip_m)
    assert command.distance.item() == pytest.approx(50.0, abs=1e-3)


def test_setpoint_tapers_to_zero_at_the_goal():
    far = _cmd(goal_xy=(1.0, 0.0)).vel_command_b[0, 0].item()
    near = _cmd(goal_xy=(0.12, 0.0)).vel_command_b[0, 0].item()
    at = _cmd(goal_xy=(0.01, 0.0)).vel_command_b[0, 0].item()
    assert far > near > at
    # Inside the stop radius the setpoint is exactly zero: arriving early buys
    # nothing, so there is no jackpot for sprinting in and stopping dead.
    assert at == 0.0


def test_setpoint_never_exceeds_the_ranges_the_gait_was_tuned_on():
    cfg = make_microduck_goto_point_env_cfg().commands["twist"]
    for goal in ((50.0, 0.0), (-50.0, 0.0), (0.0, 50.0), (0.0, -50.0)):
        command = _cmd(goal_xy=goal, goal_yaw=math.pi)
        vx, vy, wz = command.vel_command_b[0].tolist()
        eps = 1e-5  # float32 rounding on the clamp boundary
        assert cfg.ranges.lin_vel_x[0] - eps <= vx <= cfg.ranges.lin_vel_x[1] + eps
        assert cfg.ranges.lin_vel_y[0] - eps <= vy <= cfg.ranges.lin_vel_y[1] + eps
        assert cfg.ranges.ang_vel_z[0] - eps <= wz <= cfg.ranges.ang_vel_z[1] + eps
        assert abs(vx) <= cfg.max_speed + eps
        assert abs(vy) <= cfg.max_speed + eps


def test_far_away_the_robot_steers_toward_the_goal():
    # Goal ahead and to the left -> turn left, regardless of the goal heading.
    command = _cmd(goal_xy=(1.0, 1.0), goal_yaw=-math.pi / 2)
    assert command.vel_command_b[0, 2].item() > 0.0


def test_at_the_goal_the_robot_turns_to_the_commanded_heading():
    # Bearing is meaningless at zero distance (a millimetre of jitter swings it
    # 180 degrees), which is why a bearing-trained policy spins once it
    # arrives. Here the yaw error takes over.
    command = _cmd(goal_xy=(0.005, 0.0), goal_yaw=1.0)
    assert command.vel_command_b[0, 2].item() > 0.0
    opposite = _cmd(goal_xy=(0.005, 0.0), goal_yaw=-1.0)
    assert opposite.vel_command_b[0, 2].item() < 0.0


def test_a_goal_behind_produces_a_turn_not_a_backward_walk():
    command = _cmd(goal_xy=(-1.0, 0.0), heading=0.0, goal_yaw=math.pi)
    assert abs(command.vel_command_b[0, 2].item()) > 0.0


def test_goal_is_sampled_around_the_robot_s_qpos_not_its_cached_pose():
    # The bug this pins down: `root_link_pos_w` is derived state that only
    # refreshes on the next sim.forward(), which runs AFTER the command manager
    # resets. Sampling against it placed every fresh goal around the previous
    # episode's pose — and on the first reset around the world origin, which on
    # a terrain tiled at env_spacing put goals metres away in another env's
    # cell. It showed up as goal_distance ~6m against a 0.5m sampling radius.
    #
    # So: stale cache says the robot is at the origin, qpos says it is at
    # (3, -3). Every sampled goal must land near (3, -3).
    cfg = make_microduck_goto_point_env_cfg().commands["twist"]
    true_xy = (3.0, -3.0)
    for _ in range(20):
        command = _Command(
            goal_xy=(0.0, 0.0),
            goal_yaw=0.0,
            pos_xy=(0.0, 0.0),  # the stale cache
            qpos_xy=true_xy,  # the truth
            heading=0.0,
            cfg=cfg,
        )
        command.vel_command_b = torch.zeros(1, 3)
        for flag in (
            "is_standing_env",
            "is_heading_env",
            "is_world_env",
            "is_forward_env",
            "is_turn_in_place_env",
        ):
            setattr(command, flag, torch.zeros(1, dtype=torch.bool))
        command.heading_target = torch.zeros(1)
        command._resample_command(torch.tensor([0]))
        offset = (command._goal_pos_w[0] - torch.tensor(true_xy)).norm().item()
        assert offset <= cfg.goal_radius_range[1] + 1e-5, (
            f"goal landed {offset:.2f}m from the robot, outside the sampled radius"
        )


# ── rewards ──────────────────────────────────────────────────────────────────
class _RewardEnv(types.SimpleNamespace):
    pass


def _reward_env(distance, *, speed=0.0, yaw_error=0.0, tilt=0.0, first_step=False):
    term = types.SimpleNamespace(
        distance=torch.tensor([distance]),
        goal_error_b=torch.tensor([[distance, 0.0, yaw_error]]),
        cfg=types.SimpleNamespace(stop_radius=STOP_RADIUS),
    )
    data = types.SimpleNamespace(
        root_link_lin_vel_b=torch.tensor([[speed, 0.0, 0.0]]),
        root_link_ang_vel_b=torch.zeros(1, 3),
        projected_gravity_b=torch.tensor([[tilt, 0.0, -1.0]]),
    )
    return _RewardEnv(
        command_manager=types.SimpleNamespace(get_term=lambda _: term),
        scene={"robot": types.SimpleNamespace(data=data)},
        step_dt=0.02,
        device="cpu",
        episode_length_buf=torch.tensor([0 if first_step else 100]),
    )


def test_progress_pays_for_closing_and_charges_for_backing_off():
    env = _reward_env(1.0)
    mdp.goal_progress(env)
    env.command_manager.get_term(None).distance = torch.tensor([0.99])
    assert mdp.goal_progress(env).item() > 0.0
    # Holding station pays exactly zero, so orbiting the goal earns nothing.
    assert mdp.goal_progress(env).item() == 0.0
    env.command_manager.get_term(None).distance = torch.tensor([1.05])
    assert mdp.goal_progress(env).item() < 0.0


def test_progress_ignores_the_step_a_goal_is_resampled_on():
    # The goal teleports on resample; that jump is not progress.
    env = _reward_env(1.0)
    mdp.goal_progress(env)
    env.command_manager.get_term(None).distance = torch.tensor([0.1])
    env.episode_length_buf = torch.tensor([0])
    assert mdp.goal_progress(env).item() == 0.0


def test_progress_is_rate_capped():
    env = _reward_env(5.0)
    mdp.goal_progress(env)
    env.command_manager.get_term(None).distance = torch.tensor([0.0])
    assert mdp.goal_progress(env).item() <= 0.4 + 1e-6


def test_arrival_composite_collapses_on_any_single_failure():
    good = mdp.goal_arrival_composite(_reward_env(0.0)).item()
    assert good > 0.5
    # Each of these alone should cost most of the reward — that is the point of
    # a product: a sum would still pay ~75% for three-out-of-four.
    for bad in (
        _reward_env(0.5),  # not there
        _reward_env(0.0, yaw_error=1.5),  # wrong way round
        _reward_env(0.0, speed=0.6),  # still moving
        _reward_env(0.0, tilt=0.9),  # leaning
    ):
        assert mdp.goal_arrival_composite(bad).item() < good * 0.25


def test_overshoot_penalty_is_negative_only_inside_the_stop_radius():
    assert mdp.goal_overshoot_penalty(_reward_env(0.0, speed=0.5)).item() < 0.0
    assert mdp.goal_overshoot_penalty(_reward_env(0.0, speed=0.0)).item() == 0.0
    # Outside the stop radius, moving fast IS the job.
    assert mdp.goal_overshoot_penalty(_reward_env(1.0, speed=0.5)).item() == 0.0
