"""ObjectPick: config invariants, and the gates that stop the reward being farmed.

Two halves. The first checks the cfg wires up what it claims (v2 robot, props,
beak sensor, penalty signs). The second drives the reward functions directly
against a fake env and asserts each documented exploit actually scores zero —
"the gate is in the code" and "the gate fires" are different claims, and only
the second one survives contact with PPO.
"""

import types

import pytest
import torch

from mjlab_microduck.robot.microduck_constants import (
    GRAB_HEIGHTS,
    MICRODUCK_JAW_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_object_pick_env_cfg import (
    BEAK_SENSOR_PREFIX,
    ENABLE_SYMMETRY,
    EPISODE_LENGTH_S,
    HOLD_CLEARANCE,
    HOLD_MIN_HEIGHT,
    HOLD_REQUIRED_S,
    HOLD_TILT_LIMIT,
    OBJECT_NAMES,
    OBJECT_NOISE_XY,
    MicroduckObjectPickRlCfg,
    make_microduck_object_pick_env_cfg,
)

# Reward terms that only make sense against a walking velocity command. The
# twist command here is ~zero, so air_time in particular would pay the policy
# for stepping around instead of crouching.
GAIT_REWARDS = (
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "track_linear_velocity",
    "track_angular_velocity",
)


@pytest.fixture(scope="module")
def cfg():
    return make_microduck_object_pick_env_cfg()


# ── config ───────────────────────────────────────────────────────────────────
def test_env_builds_train_and_play():
    assert make_microduck_object_pick_env_cfg() is not None
    assert make_microduck_object_pick_env_cfg(play=True) is not None


def test_runs_on_the_v2_robot(cfg):
    # The whole point of v2: an actuated jaw. A v1 robot here would build fine
    # and simply never be able to grip anything.
    assert cfg.scene.entities["robot"] is MICRODUCK_JAW_ROBOT_CFG


def test_all_three_props_are_in_the_scene(cfg):
    assert OBJECT_NAMES == ("block", "rubber_ball", "sock")
    for name in OBJECT_NAMES:
        assert name in cfg.scene.entities
    assert tuple(cfg.grab_object_names) == OBJECT_NAMES
    # Every prop needs a resting height, or the lift potential is measured
    # against a KeyError.
    for name in OBJECT_NAMES:
        assert name in GRAB_HEIGHTS


def test_every_prop_gets_its_own_beak_contact_sensor(cfg):
    # A secondary pattern is only a regex when scoped to one entity, so "beak
    # against any prop" needs one sensor per prop; a missing one would silently
    # read as "never in the beak" for that prop and it would never be picked up.
    sensors = {s.name: s for s in cfg.scene.sensors}
    for name in OBJECT_NAMES:
        sensor = sensors[f"{BEAK_SENSOR_PREFIX}_{name}"]
        # Must address BOTH beak surfaces: an object held against the upper
        # mouth alone would read as "not in the beak".
        assert sensor.primary.pattern == r"^(jaw|upper_mouth)_collision$"
        assert sensor.secondary.entity == name
    assert cfg.rewards["object_hold"].params["sensor_prefix"] == BEAK_SENSOR_PREFIX


def test_contact_budget_was_raised_for_the_extra_bodies(cfg):
    # Three free props on top of a full-collision robot. Too small an nconmax
    # silently drops contacts, which looks like a beak that cannot grip.
    assert cfg.sim.nconmax >= 80


def test_gait_rewards_are_gone(cfg):
    for name in GAIT_REWARDS:
        assert name not in cfg.rewards, f"gait reward survived: {name}"


def test_object_placed_after_the_base_is_reset(cfg):
    # reset_grab_object reads the robot root from qpos, so reset_base has to
    # have run first. Event order is dict insertion order.
    names = list(cfg.events.keys())
    assert names.index("reset_grab_object") > names.index("reset_base")


def test_placement_noise_stays_inside_the_reach_envelope(cfg):
    # 85mm +/- noise has to stay inside the 60-110mm band the beak was measured
    # to reach from a stable crouch (scripts/measure_v2_reach.py).
    offset_x = cfg.events["reset_grab_object"].params["offset"][0]
    assert 0.06 <= offset_x - OBJECT_NOISE_XY
    assert offset_x + OBJECT_NOISE_XY <= 0.11


def test_hold_clearance_is_a_real_margin(cfg):
    # A clearance of ~0 would read placement jitter as a lift.
    assert cfg.rewards["object_hold"].params["clearance"] == HOLD_CLEARANCE
    assert HOLD_CLEARANCE >= 0.01


def test_hold_success_is_logged_but_never_optimized(cfg):
    # It is the deliverable ("3 seconds"), not a training signal: as a reward it
    # would be a sparse jackpot. At weight 0 it shows up in wandb as the real
    # success criterion while total reward climbs on shaping.
    term = cfg.rewards["hold_success"]
    assert term.weight == 0.0
    assert term.params["required_s"] == HOLD_REQUIRED_S == 3.0


def test_no_reward_pays_for_closing_the_jaw(cfg):
    # The single most important invariant in this env. A closure reward is
    # farmed by chomping air, immediately, because chomping is free.
    for name in cfg.rewards:
        assert "jaw" not in name, f"reward term mentions the jaw: {name}"


def test_penalty_signs_match_each_function_s_convention(cfg):
    # mdp.py carries BOTH conventions and mixing them up is the bug that has
    # bitten four envs: a self-negating `*_penalty` (returns <= 0) with a
    # negative weight double-negates into a reward for the violation.
    assert cfg.rewards["object_disturb"].func is mdp.object_disturb_penalty
    assert cfg.rewards["object_disturb"].weight >= 0.0
    # mjlab-base cost style (returns >= 0) -> negative weight.
    assert cfg.rewards["head_impact"].func is mdp.body_impact_cost
    assert cfg.rewards["head_impact"].weight < 0.0


def test_task_terms_take_positive_weights(cfg):
    for name in ("object_lift", "object_hold", "mouth_to_object"):
        assert cfg.rewards[name].weight > 0.0


def test_taxes_start_at_zero_and_ramp(cfg):
    # An attempt-tax active during skill discovery makes "do nothing" win.
    stages = cfg.curriculum["object_disturb_weight"].params["weight_stages"]
    assert stages[0]["step"] == 0
    assert stages[0]["weight"] == 0.0
    assert cfg.rewards["object_disturb"].weight == 0.0
    weights = [s["weight"] for s in stages]
    assert weights == sorted(weights)


def test_approach_shaping_fades(cfg):
    # Left at full weight it competes with lifting: the beak is CLOSEST to the
    # object while the object is still on the ground.
    stages = cfg.curriculum["mouth_to_object_weight"].params["weight_stages"]
    weights = [s["weight"] for s in stages]
    assert weights[0] == cfg.rewards["mouth_to_object"].weight
    assert weights == sorted(weights, reverse=True)
    assert weights[-1] < weights[0]


def test_episode_is_long_enough_to_crouch_grab_stand_and_hold(cfg):
    assert cfg.episode_length_s == EPISODE_LENGTH_S
    # Crouch + grab + stand back up, and still 3s of holding left over.
    assert EPISODE_LENGTH_S > HOLD_REQUIRED_S * 2


def test_twist_slot_is_kept_alive_but_neutral(cfg):
    # Deleting the slot would break the 64D layout; leaving it wide would ask a
    # stationary manipulation task to walk.
    twist = cfg.commands["twist"]
    assert abs(twist.ranges.lin_vel_x[1]) <= 0.05
    assert abs(twist.ranges.ang_vel_z[1]) <= 0.05
    # Non-zero, so those three input neurons keep receiving signal.
    assert twist.ranges.lin_vel_x[1] > 0.0


def test_object_target_reaches_the_actor_and_truth_reaches_only_the_critic(cfg):
    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms
    assert actor["body_command"].func is mdp.grab_target_command
    for truth in ("object_position", "object_velocity", "object_lift"):
        assert truth in critic
        assert truth not in actor, (
            f"{truth} is privileged information the real robot cannot observe"
        )


def test_target_point_is_noisy(cfg):
    # The target comes from a detector at deployment. Training on a perfect one
    # produces a policy that breaks on the first few-millimetre bbox error.
    assert cfg.grab_target_noise > 0.0


def test_symmetry_is_off():
    # symmetry.py's mirror table is hardcoded for the 61D v1 layout; applied to
    # a 64D observation it would permute the wrong entries.
    assert ENABLE_SYMMETRY is False
    assert MicroduckObjectPickRlCfg.algorithm.symmetry_cfg is None


def test_experiment_name_is_distinct():
    assert MicroduckObjectPickRlCfg.experiment_name == "object_pick"


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    assert "Mjlab-ObjectPick-Flat-MicroDuck" in list_tasks()


# ── fake env: exercise the gates themselves ──────────────────────────────────
class _FakeData:
    def __init__(self, pos=None, quat=None, lin_vel=None, gravity=None, site=None):
        self.root_link_pos_w = pos
        self.root_link_quat_w = quat
        self.root_link_lin_vel_w = lin_vel
        self.projected_gravity_b = gravity
        self.site_pos_w = site


class _FakeEntity:
    def __init__(self, **kwargs):
        self.data = _FakeData(**kwargs)


# The approach reward resolves its site through a SceneEntityCfg; the fake env
# has no scene to resolve against, so hand it the already-resolved id.
_JAW_CFG = types.SimpleNamespace(name="robot", site_ids=[0])


class _FakeSensor:
    def __init__(self, found):
        self.data = types.SimpleNamespace(found=found)


class _FakeScene(dict):
    def __init__(self, entities, terrain_z, sensors=None):
        super().__init__(entities)
        n = len(terrain_z)
        origins = torch.zeros(n, 3)
        origins[:, 2] = torch.as_tensor(terrain_z, dtype=torch.float32)
        self.terrain = types.SimpleNamespace(env_origins=origins)
        self.sensors = sensors or {}


def _make_env(
    *,
    object_z,
    robot_z=0.115,
    tilt=0.0,
    contact=True,
    n=1,
    terrain_z=0.0,
    beak_xyz=(0.0, 0.0, 0.0),
    object_xy=(0.085, 0.0),
):
    """One env per element, with the block as the active prop."""
    object_pos = torch.zeros(n, 3)
    object_pos[:, 0] = object_xy[0]
    object_pos[:, 1] = object_xy[1]
    object_pos[:, 2] = torch.as_tensor(object_z, dtype=torch.float32)

    robot_pos = torch.zeros(n, 3)
    robot_pos[:, 2] = robot_z
    gravity = torch.zeros(n, 3)
    gravity[:, 0] = tilt
    gravity[:, 2] = -1.0

    site = torch.zeros(n, 1, 3)
    site[:, 0, :] = torch.tensor(beak_xyz)

    entities = {
        "robot": _FakeEntity(
            pos=robot_pos,
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
            gravity=gravity,
            lin_vel=torch.zeros(n, 3),
            site=site,
        ),
        "block": _FakeEntity(pos=object_pos, lin_vel=torch.zeros(n, 3)),
        "rubber_ball": _FakeEntity(pos=torch.zeros(n, 3), lin_vel=torch.zeros(n, 3)),
        "sock": _FakeEntity(pos=torch.zeros(n, 3), lin_vel=torch.zeros(n, 3)),
    }
    # One sensor per prop; only the active one is ever consulted.
    sensors = {
        f"{BEAK_SENSOR_PREFIX}_{name}": _FakeSensor(
            torch.full((n, 1), 1.0 if contact else 0.0)
        )
        for name in OBJECT_NAMES
    }
    env = types.SimpleNamespace(
        scene=_FakeScene(entities, [terrain_z] * n, sensors),
        cfg=types.SimpleNamespace(grab_object_names=OBJECT_NAMES),
        num_envs=n,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.full((n,), 100, dtype=torch.long),
    )
    env._grab_active_idx = torch.zeros(n, dtype=torch.long)
    return env


def _rest_z(name="block", terrain_z=0.0):
    return terrain_z + GRAB_HEIGHTS[name] / 2.0


_HELD_Z = _rest_z() + HOLD_CLEARANCE + 0.01


def test_hold_pays_when_the_object_is_genuinely_held():
    env = _make_env(object_z=_HELD_Z)
    assert mdp.object_hold(env).item() == pytest.approx(1.0)


def test_pushing_the_object_along_the_floor_is_not_a_hold():
    # Object at rest height, beak touching it, robot standing: every gate but
    # clearance is satisfied. This is the shove-it-along exploit.
    env = _make_env(object_z=_rest_z())
    assert mdp.object_hold(env).item() == 0.0


def test_balancing_the_object_without_the_beak_is_not_a_hold():
    env = _make_env(object_z=_HELD_Z, contact=False)
    assert mdp.object_hold(env).item() == 0.0


def test_falling_on_the_object_is_not_a_hold():
    # Object airborne and in contact, but the robot is collapsed — the pinning
    # exploit that a height-plus-contact reward alone would pay in full.
    collapsed = _make_env(object_z=_HELD_Z, robot_z=HOLD_MIN_HEIGHT - 0.02)
    assert mdp.object_hold(collapsed).item() == 0.0
    toppled = _make_env(object_z=_HELD_Z, tilt=HOLD_TILT_LIMIT + 0.1)
    assert mdp.object_hold(toppled).item() == 0.0


def test_hold_is_measured_against_the_local_terrain_height():
    # env_origins carry a per-env z. Measuring lift against absolute world z
    # would read every prop on raised terrain as permanently lifted.
    raised = 0.5
    env = _make_env(
        object_z=_rest_z(terrain_z=raised), terrain_z=raised, robot_z=raised + 0.115
    )
    assert mdp.object_hold(env).item() == 0.0


def test_lift_progress_pays_only_for_rising():
    env = _make_env(object_z=_rest_z())
    mdp.object_lift_progress(env)
    env.scene["block"].data.root_link_pos_w[:, 2] = _rest_z() + 0.01
    rising = mdp.object_lift_progress(env).item()
    assert rising > 0.0
    # Holding steady pays exactly zero: the hold term covers that, and paying
    # twice would make hovering more attractive than lifting further.
    assert mdp.object_lift_progress(env).item() == 0.0
    # Dropping does not refund, so bouncing the object cannot be farmed.
    env.scene["block"].data.root_link_pos_w[:, 2] = _rest_z()
    assert mdp.object_lift_progress(env).item() == 0.0
    env.scene["block"].data.root_link_pos_w[:, 2] = _rest_z() + 0.01
    assert mdp.object_lift_progress(env).item() == 0.0


def test_lift_progress_is_rate_capped():
    # The anti-jackpot guard: without it, flicking the object into the air
    # collects the whole potential in two steps, and the cheapest way to do
    # that is a violent head-whip rather than a grip.
    env = _make_env(object_z=_rest_z())
    mdp.object_lift_progress(env)
    env.scene["block"].data.root_link_pos_w[:, 2] = _rest_z() + 5.0
    flick = mdp.object_lift_progress(env).item()

    slow = _make_env(object_z=_rest_z())
    mdp.object_lift_progress(slow)
    slow.scene["block"].data.root_link_pos_w[:, 2] = _rest_z() + 0.002
    assert flick <= mdp.object_lift_progress(slow).item() * 2.0 + 1e-6


def test_lift_progress_ignores_the_first_step_of_an_episode():
    # The prop teleports at reset; that jump is placement, not progress.
    env = _make_env(object_z=_rest_z() + 0.05)
    env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    assert mdp.object_lift_progress(env).item() == 0.0


def test_hold_success_needs_three_continuous_seconds():
    env = _make_env(object_z=_HELD_Z)
    steps = int(HOLD_REQUIRED_S / env.step_dt)
    for _ in range(steps - 1):
        assert mdp.object_hold_success(env).item() == 0.0
    assert mdp.object_hold_success(env).item() == 1.0


def test_hold_success_streak_resets_when_the_object_is_dropped():
    env = _make_env(object_z=_HELD_Z)
    for _ in range(int(HOLD_REQUIRED_S / env.step_dt) - 5):
        mdp.object_hold_success(env)
    env.scene["block"].data.root_link_pos_w[:, 2] = _rest_z()  # dropped
    mdp.object_hold_success(env)
    env.scene["block"].data.root_link_pos_w[:, 2] = _HELD_Z  # picked back up
    # Must serve the full 3s again, not resume near the finish line.
    for _ in range(int(HOLD_REQUIRED_S / env.step_dt) - 1):
        assert mdp.object_hold_success(env).item() == 0.0
    assert mdp.object_hold_success(env).item() == 1.0


def test_approach_shaping_switches_off_once_the_object_is_up():
    beak = (0.085, 0.0, _rest_z())
    on_ground = _make_env(object_z=_rest_z(), beak_xyz=beak)
    assert mdp.mouth_to_object(on_ground, asset_cfg=_JAW_CFG).item() > 0.5
    lifted = _make_env(object_z=_HELD_Z, beak_xyz=beak)
    # Otherwise the policy hovers next to an object it has already picked up
    # and collects the approach reward forever.
    assert mdp.mouth_to_object(lifted, asset_cfg=_JAW_CFG).item() == 0.0


def test_approach_shaping_is_visible_from_where_the_robot_actually_starts(cfg):
    # MEASURED on the v2 model: a standing robot's jaw_tip is ~0.21m from a prop
    # at the nominal offset. A single tight std reads exp(-27) there — the
    # reward is correct and completely invisible, which is the failure this
    # two-scale form exists to avoid. It must also rise monotonically.
    params = dict(cfg.rewards["mouth_to_object"].params)
    params.pop("asset_cfg", None)
    values = []
    for distance in (0.21, 0.15, 0.10, 0.05, 0.0):
        env = _make_env(
            object_z=_rest_z(), beak_xyz=(0.085, 0.0, _rest_z() + distance)
        )
        values.append(mdp.mouth_to_object(env, asset_cfg=_JAW_CFG, **params).item())
    assert values[0] > 0.01, "no gradient at the start of an episode"
    assert values == sorted(values)
    assert values[-1] == pytest.approx(1.0)


def test_disturb_penalty_is_negative_and_only_charged_before_the_lift():
    env = _make_env(object_z=_rest_z(), object_xy=(0.085, 0.0))
    env._grab_spawn_xy = torch.tensor([[0.085, 0.0]])
    assert mdp.object_disturb_penalty(env).item() == 0.0

    knocked = _make_env(object_z=_rest_z(), object_xy=(0.20, 0.0))
    knocked._grab_spawn_xy = torch.tensor([[0.085, 0.0]])
    assert mdp.object_disturb_penalty(knocked).item() < 0.0

    # Carried: same displacement, but the object is off the ground. Moving it
    # horizontally is now the job.
    carried = _make_env(object_z=_HELD_Z, object_xy=(0.20, 0.0))
    carried._grab_spawn_xy = torch.tensor([[0.085, 0.0]])
    assert mdp.object_disturb_penalty(carried).item() == 0.0


def test_parked_props_stay_on_the_floor_and_inside_their_own_env_cell(cfg):
    # Two failure modes, both found in sim rather than on paper. Parking props
    # below the floor does not work: the terrain plane is infinite and
    # one-sided, so a prop underneath is penetrating it and gets shoved back
    # up into the episode. And envs are TILED into one world at env_spacing,
    # not simulated as independent worlds, so parking far to the side drops
    # props into the neighbouring env.
    assert mdp._PARKED_OFFSET > 0.0, "parking must be lateral, not below ground"
    n_objects = len(OBJECT_NAMES)
    furthest = mdp._PARKED_OFFSET + 0.1 * (n_objects - 1)
    assert furthest < cfg.scene.env_spacing / 2.0, (
        f"a prop parked {furthest}m out lands in the next env's cell "
        f"(spacing {cfg.scene.env_spacing}m)"
    )
    # And far outside the reach of a robot that is never told to walk here.
    assert mdp._PARKED_OFFSET > 0.3


def test_reads_follow_the_active_prop():
    # All three props live in the scene at once and two are parked below the
    # floor. Reading a fixed entity would measure a parked prop in two thirds
    # of the envs.
    env = _make_env(object_z=_rest_z(), n=1)
    env.scene["sock"].data.root_link_pos_w[:, 2] = _rest_z("sock") + 0.10
    env._grab_active_idx = torch.tensor([OBJECT_NAMES.index("sock")])
    assert mdp._object_lift(env).item() == pytest.approx(0.10, abs=1e-6)


def test_grab_target_command_overwrites_only_the_position_slots():
    env = _make_env(object_z=_rest_z())
    command = torch.tensor([[9.0, 9.0, 9.0, 0.4, 0.5, 0.6]])
    env.command_manager = types.SimpleNamespace(get_command=lambda _: command)
    env.cfg.grab_target_noise = 0.0
    out = mdp.grab_target_command(env)
    assert out.shape == (1, 6)
    # Slots 3:6 must survive: a command input that is never non-zero has dead
    # weights forever, and a later curriculum needs them alive.
    assert torch.allclose(out[:, 3:], command[:, 3:])
    assert torch.allclose(out[:, :3], mdp.object_pos_in_base(env))
