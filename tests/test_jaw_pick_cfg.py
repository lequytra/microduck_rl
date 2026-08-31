"""Cfg invariants for the v2 jaw-pick env (63 obs / 15 act contract).

Locks in: the v1 obs blocks stay 14-wide on the jaw model (the `jaw` joint
is NOT passive_*, so the shared regexes would silently grow them), the two
jaw obs terms land at slots 61/62 (insertion order = layout), the action
splits body(14) + jaw(1) with jaw at act[14], and the graft warm-start
conversion preserves body weights exactly.
"""
import sys
from pathlib import Path

import torch

from mjlab_microduck.tasks.microduck_jaw_pick_env_cfg import (
    MicroduckJawPickRlCfg,
    make_microduck_jaw_pick_env_cfg,
)
from mjlab_microduck.tasks import mdp as microduck_mdp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import graft_v2_actor  # noqa: E402


def test_jaw_pick_robot_is_jaw_model():
    from mjlab_microduck.robot.microduck_constants import (
        MICRODUCK_GROUND_PICK_JAW_ROBOT_CFG,
    )
    cfg = make_microduck_jaw_pick_env_cfg()
    assert cfg.scene.entities["robot"] is MICRODUCK_GROUND_PICK_JAW_ROBOT_CFG


def test_jaw_pick_actions_split_body_then_jaw():
    """Two action terms, body (jaw excluded) THEN jaw → jaw lands at act[14]."""
    cfg = make_microduck_jaw_pick_env_cfg()
    keys = list(cfg.actions.keys())
    assert keys == ["joint_pos", "jaw"], keys
    body, jaw = cfg.actions["joint_pos"], cfg.actions["jaw"]
    assert body.actuator_names == (r"^(?!jaw$).*",)
    assert body.scale == 1.0
    assert jaw.actuator_names == ("jaw",)
    assert jaw.scale == 0.52


def test_jaw_pick_obs_contract_63():
    """v1 blocks stay 14-wide; jaw obs appended last in BOTH groups."""
    cfg = make_microduck_jaw_pick_env_cfg()
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        # jaw terms are the LAST two (insertion order = obs layout → 61, 62)
        assert list(terms.keys())[-2:] == ["jaw_angle", "jaw_current"]
        assert terms["jaw_angle"].func is microduck_mdp.jaw_angle_obs
        assert terms["jaw_current"].func is microduck_mdp.jaw_current_obs
        # the joint blocks must exclude the jaw explicitly (silent-64 trap)
        for name in ("joint_pos", "joint_vel"):
            assert terms[name].params["asset_cfg"].joint_names == (
                r"^(?!passive_|jaw$).*",
            ), name
        # last_action stays 14-D by pointing at the body term only
        assert terms["actions"].params.get("action_name") == "joint_pos"


def test_jaw_pick_reward_wired():
    cfg = make_microduck_jaw_pick_env_cfg()
    r = cfg.rewards
    assert r["jaw_aperture"].func is microduck_mdp.jaw_aperture_phased_reward
    assert r["jaw_aperture"].weight == 2.0  # positive weight, positive reward
    for k in ("descent_end", "hold_end", "rise_end"):
        assert k in r["jaw_aperture"].params
    # inherited ground-pick penalties keep their signs (the ≤0 invariant)
    for name in ("head_impact_penalty", "feet_flat", "action_rate_l2",
                 "joint_torques_l2", "self_collisions"):
        assert r[name].weight < 0, name


def test_jaw_pick_variants_build():
    assert "jaw_aperture" in make_microduck_jaw_pick_env_cfg(rough=True).rewards
    assert "jaw_aperture" in make_microduck_jaw_pick_env_cfg(play=True).rewards


def test_jaw_pick_runner_cfg():
    agent = MicroduckJawPickRlCfg()
    assert agent.experiment_name == "jaw_pick_v2"
    assert agent.graft_from == ""
    assert agent.max_iterations == 2000


def test_graft_v2_actor_roundtrip():
    """v1-shaped synthetic sd → v2 sd: body bit-exact, jaw dormant, std fresh."""
    sd_v1 = graft_v2_actor.synth_v1_state_dict()
    sd_v2 = graft_v2_actor.graft(sd_v1, jaw_std=1.0)
    assert sd_v2["mlp.0.weight"].shape == (512, 63)
    assert sd_v2["mlp.6.weight"].shape == (15, 128)
    assert sd_v2["mlp.6.bias"].shape == (15,)
    assert sd_v2["distribution.std_param"].shape == (15,)
    # new input columns and output row are exactly zero
    assert sd_v2["mlp.0.weight"][:, 61:].abs().max() == 0
    assert sd_v2["mlp.6.weight"][14].abs().max() == 0
    assert sd_v2["mlp.6.bias"][14] == 0
    # normalizer: new dims passthrough (mean 0, var/std 1), count copied
    assert sd_v2["obs_normalizer._mean"][0, 61:].abs().max() == 0
    assert (sd_v2["obs_normalizer._std"][0, 61:] == 1).all()
    assert (sd_v2["obs_normalizer._var"][0, 61:] == 1).all()
    assert sd_v2["obs_normalizer.count"] == sd_v1["obs_normalizer.count"]
    # jaw exploration std fresh
    assert sd_v2["distribution.std_param"][14] == 1.0
    # body forward pass identical (verify() asserts < 1e-5 and jaw == 0)
    graft_v2_actor.verify(sd_v1, sd_v2)
