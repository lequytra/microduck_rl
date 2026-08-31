#!/usr/bin/env python3
"""Graft a v1 microduck actor checkpoint (61 obs / 14 act) into a v2 jaw-policy
actor state_dict (63 obs / 15 act) as an rsl_rl training warm-start.

v2 adds:  obs[61] = jaw angle, obs[62] = jaw contact force
          act[14] = jaw aperture

Widen-and-zero-init, same rationale as scripts/graft_v2_jaw.py: the first layer
gains two all-zero columns (jaw obs contribute nothing), the last layer gains an
all-zero row (jaw action stays dormant at 0), and the obs normalizer widens to
mean=0 / var = std =1 passthrough. `distribution.std_param` gains the runner
cfg's init_std (`--jaw-std`, default 1.0) so the jaw channel starts with full
exploration noise while the 14 body weights stay bit-identical. The grafted
state_dict loads with runner.alg.actor.load_state_dict(..., strict=True).

Caveat vs the ONNX graft: export.py bakes EmpiricalNormalization's 1e-2 eps into
the ONNX Div constants (`(x - _mean) / (_std + 1e-2)`), so the ONNX Div holds the
*composed* std. The checkpoint buffers hold the clean moments (_mean, _var,
_std), so this script re-applies eps at verify time exactly as
EmpiricalNormalization.forward does (see rsl_rl/modules/normalization.py).

    uv run python scripts/graft_v2_actor.py V1_CKPT.pt OUT.pt [--jaw-std 1.0]
    uv run python scripts/graft_v2_actor.py --self-test X OUT.pt
        # self-test: synthesize a random v1-shaped actor state_dict, graft,
        # and verify the round-trip numerically.
"""
import argparse

import torch
import torch.nn.functional as F

V1_OBS, V2_OBS = 61, 63
V1_ACT, V2_ACT = 14, 15
HIDDEN = (512, 256, 128)
EPS = 1e-2  # EmpiricalNormalization eps, added to _std OUTSIDE the square root

# Expected v1 actor keys and shapes (rsl_rl 5.0.1 MLPModel layout
# 61 -> 512 -> 256 -> 128 -> 14, ELU, GaussianDistribution scalar std).
EXPECTED = {
    "obs_normalizer._mean": (1, V1_OBS),
    "obs_normalizer._var": (1, V1_OBS),
    "obs_normalizer._std": (1, V1_OBS),
    "obs_normalizer.count": (),
    "mlp.0.weight": (HIDDEN[0], V1_OBS),
    "mlp.0.bias": (HIDDEN[0],),
    "mlp.2.weight": (HIDDEN[1], HIDDEN[0]),
    "mlp.2.bias": (HIDDEN[1],),
    "mlp.4.weight": (HIDDEN[2], HIDDEN[1]),
    "mlp.4.bias": (HIDDEN[2],),
    "mlp.6.weight": (V1_ACT, HIDDEN[2]),
    "mlp.6.bias": (V1_ACT,),
    "distribution.std_param": (V1_ACT,),
}


def sanity_check(sd: dict) -> None:
    """Assert every expected key/shape is present; warn on unexpected extras."""
    for key, shape in EXPECTED.items():
        if key not in sd:
            raise KeyError(f"v1 actor_state_dict is missing key {key!r}")
        t = sd[key]
        if key == "obs_normalizer.count":
            if t.ndim != 0 and t.numel() != 1:
                raise ValueError(f"obs_normalizer.count shape {tuple(t.shape)} != scalar")
            continue
        if tuple(t.shape) != shape:
            raise ValueError(f"{key} shape {tuple(t.shape)} != expected {shape}")
    for key in sorted(set(sd) - set(EXPECTED)):
        print(f"warning: unexpected key {key!r} copied through unchanged")


def graft(sd_v1: dict, jaw_std: float) -> dict:
    """Widen a v1 actor state_dict to the v2 63-obs / 15-act skeleton."""
    sd = {k: v.clone() for k, v in sd_v1.items()}  # copies unexpected keys through

    # Obs normalizer: mean 0 / var & std 1 passthrough for the two jaw obs.
    sd["obs_normalizer._mean"] = torch.cat([sd["obs_normalizer._mean"], torch.zeros((1, 2), dtype=sd["obs_normalizer._mean"].dtype)], dim=1)
    sd["obs_normalizer._var"] = torch.cat([sd["obs_normalizer._var"], torch.ones((1, 2), dtype=sd["obs_normalizer._var"].dtype)], dim=1)
    sd["obs_normalizer._std"] = torch.cat([sd["obs_normalizer._std"], torch.ones((1, 2), dtype=sd["obs_normalizer._std"].dtype)], dim=1)

    # First layer: two all-zero columns -> jaw obs contribute nothing.
    sd["mlp.0.weight"] = torch.cat([sd["mlp.0.weight"], torch.zeros((HIDDEN[0], 2), dtype=sd["mlp.0.weight"].dtype)], dim=1)

    # Last layer: one all-zero row -> jaw action dormant at 0.
    sd["mlp.6.weight"] = torch.cat([sd["mlp.6.weight"], torch.zeros((1, HIDDEN[2]), dtype=sd["mlp.6.weight"].dtype)], dim=0)
    sd["mlp.6.bias"] = torch.cat([sd["mlp.6.bias"], torch.zeros(1, dtype=sd["mlp.6.bias"].dtype)])

    # Jaw channel starts at full exploration noise (runner cfg init_std).
    sd["distribution.std_param"] = torch.cat(
        [sd["distribution.std_param"], torch.tensor([jaw_std], dtype=sd["distribution.std_param"].dtype)])
    return sd


def mlp_forward(obs: torch.Tensor, sd: dict) -> torch.Tensor:
    """Manual MLPModel forward mirroring EmpiricalNormalization.forward exactly."""
    x = (obs - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + EPS)
    for i in (0, 2, 4):
        x = F.elu(F.linear(x, sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]))
    return F.linear(x, sd["mlp.6.weight"], sd["mlp.6.bias"])


def verify(sd_v1: dict, sd_v2: dict) -> None:
    """Body actions must match < 1e-5 under noisy jaw obs; jaw action ~0."""
    rng = torch.Generator()
    rng.manual_seed(0)
    obs61 = torch.randn(64, V1_OBS, generator=rng)
    obs63 = torch.cat([obs61, torch.randn(64, V2_OBS - V1_OBS, generator=rng) * 5], dim=1)
    a1 = mlp_forward(obs61, sd_v1)
    a2 = mlp_forward(obs63, sd_v2)
    body_err = (a2[:, :V1_ACT] - a1).abs().max().item()
    jaw_max = a2[:, V1_ACT:].abs().max().item()
    print(f"graft check: body |v2-v1| = {body_err:.2e}, jaw |aperture| = {jaw_max:.2e}")
    assert body_err < 1e-5, f"body actions diverged: {body_err:.2e}"
    assert jaw_max < 1e-6, f"jaw action not dormant: {jaw_max:.2e}"


def synth_v1_state_dict() -> dict:
    """Random v1-shaped actor state_dict (mean 0, positive var/std, stds > 0)."""
    g = torch.Generator()
    g.manual_seed(7)
    sd = {}
    for key, shape in EXPECTED.items():
        if key == "obs_normalizer.count":
            sd[key] = torch.tensor(4321, dtype=torch.long)
        elif key == "obs_normalizer._var":
            sd[key] = torch.rand(shape, generator=g) + 0.5
        elif key == "obs_normalizer._std":
            sd[key] = sd["obs_normalizer._var"].sqrt()
        elif key == "distribution.std_param":
            sd[key] = torch.rand(shape, generator=g) * 0.8 + 0.05
        else:
            sd[key] = torch.randn(shape, generator=g)
    return sd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("v1_ckpt", nargs="?", help="v1 training checkpoint .pt (ignored with --self-test)")
    p.add_argument("out", help="output path for the grafted v2 actor state_dict .pt")
    p.add_argument("--jaw-std", type=float, default=1.0,
                   help="std_param for the new jaw action channel (runner cfg init_std, default 1.0)")
    p.add_argument("--self-test", action="store_true",
                   help="synthesize a random v1 actor state_dict and verify the graft round-trip")
    args = p.parse_args()

    if args.self_test:
        sd_v1 = synth_v1_state_dict()
        src = "synthetic v1 state_dict"
    else:
        if not args.v1_ckpt:
            p.error("V1_CKPT.pt is required unless --self-test")
        ckpt = torch.load(args.v1_ckpt, map_location="cpu", weights_only=False)
        if "actor_state_dict" not in ckpt:
            raise KeyError("v1 checkpoint has no 'actor_state_dict' top-level key "
                           f"(got {sorted(ckpt)})")
        sd_v1 = ckpt["actor_state_dict"]
        src = args.v1_ckpt

    sanity_check(sd_v1)
    sd_v2 = graft(sd_v1, args.jaw_std)
    verify(sd_v1, sd_v2)

    torch.save(sd_v2, args.out)
    print(f"grafted {src} -> {args.out}")


if __name__ == "__main__":
    main()
