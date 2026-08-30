#!/usr/bin/env python3
"""Graft a v1 microduck policy (61 obs / 14 act) into a v2 jaw-policy skeleton
(63 obs / 15 act) — the standing start for training a learned grasp.

v2 adds:  obs[61] = jaw angle, obs[62] = jaw contact force
          act[14] = jaw aperture

Widen-and-zero-init: the first layer gains two all-zero columns (jaw obs
contribute nothing), the last layer gains an all-zero row (jaw action starts
dormant at mid-travel). The grafted network computes the v1 policy EXACTLY on
the 14 body actions, so the trained crouch/stand behavior is preserved while
PPO later unfreezes the jaw channel.

Verifier built in: grafted-vs-v1 body actions must match to < 1e-5 and the
jaw action must be ~0, cross-checked through onnxruntime.

    uv run --with onnx python scripts/graft_v2_jaw.py SRC.onnx DST.onnx

Note: initializer names (obs_normalizer._mean, mlp.N.weight) follow
scripts/export.py's rsl_rl MLP layout — 61->512->256->128->14, ELU, with the
obs normalizer baked in as Sub+Div. If export.py's architecture changes, this
script needs the same change.
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def elu(x):
    return np.where(x > 0, x, np.expm1(np.clip(x, -20, None)))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    init = {i.name: numpy_helper.to_array(i) for i in onnx.load(src).graph.initializer}

    mean, std = init["obs_normalizer._mean"], init["onnx::Div_24"]  # [1, 61]
    W0, b0 = init["mlp.0.weight"], init["mlp.0.bias"]               # [512, 61]
    W2, b2 = init["mlp.2.weight"], init["mlp.2.bias"]               # [256, 512]
    W4, b4 = init["mlp.4.weight"], init["mlp.4.bias"]               # [128, 256]
    W6, b6 = init["mlp.6.weight"], init["mlp.6.bias"]               # [14, 128]

    def mlp(obs, mean_, std_, W0_, W6_, b6_):
        x = (obs - mean_) / std_
        for W, b in ((W0_, b0), (W2, b2), (W4, b4), (W6_, b6_)):
            x = x @ W.T + b
            if W is not W6_:
                x = elu(x)
        return x

    # Graft: widen input 61->63 (new dims: mean 0, std 1 passthrough) and
    # output 14->15 (dormant jaw row).
    mean63 = np.concatenate([mean, np.zeros((1, 2), np.float32)], axis=1)
    std63 = np.concatenate([std, np.ones((1, 2), np.float32)], axis=1)
    W0_63 = np.concatenate([W0, np.zeros((512, 2), np.float32)], axis=1)
    W6_15 = np.concatenate([W6, np.zeros((1, 128), np.float32)], axis=0)
    b6_15 = np.concatenate([b6, np.zeros(1, np.float32)])

    # Verify: body identical under noisy jaw obs, jaw dormant.
    rng = np.random.default_rng(0)
    obs61 = rng.normal(size=(64, 61)).astype(np.float32)
    obs63 = np.concatenate([obs61, rng.normal(size=(64, 2)).astype(np.float32) * 5], axis=1)
    a1 = mlp(obs61, mean, std, W0, W6, b6)
    a2 = mlp(obs63, mean63, std63, W0_63, W6_15, b6_15)
    body_err = np.abs(a1 - a2[:, :14]).max()
    jaw_max = np.abs(a2[:, 14]).max()
    print(f"graft check: body |v2-v1| = {body_err:.2e}, jaw |aperture| = {jaw_max:.2e}")
    assert body_err < 1e-5 and jaw_max < 1e-6

    # Export the v2 skeleton (same graph shape: Sub -> Div -> Gemm x4, ELU x3).
    import torch

    class V2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.from_numpy(mean63.copy()))
            self.register_buffer("std", torch.from_numpy(std63.copy()))
            layers = [(63, 512, W0_63, b0), (512, 256, W2, b2),
                      (256, 128, W4, b4), (128, 15, W6_15, b6_15)]
            self.layers = torch.nn.ModuleList()
            for i, (din, dout, W, b) in enumerate(layers):
                lin = torch.nn.Linear(din, dout)
                lin.weight.data = torch.from_numpy(W.copy())
                lin.bias.data = torch.from_numpy(b.copy())
                self.layers.append(lin)

        def forward(self, obs):
            x = (obs - self.mean) / self.std
            for lin in self.layers[:-1]:
                x = torch.nn.functional.elu(lin(x))
            return self.layers[-1](x)

    torch.onnx.export(V2().eval(), torch.zeros(1, 63), dst,
                      input_names=["obs"], output_names=["actions"],
                      opset_version=18)

    import onnxruntime as ort
    ref = np.concatenate([ort.InferenceSession(src).run(None, {"obs": obs61[i:i+1]})[0]
                          for i in range(8)])
    out2 = np.concatenate([ort.InferenceSession(dst).run(None, {"obs": obs63[i:i+1]})[0]
                           for i in range(8)])
    print(f"ort cross-check: body |v2-v1| = {np.abs(out2[:, :14] - ref).max():.2e}, "
          f"jaw = {np.abs(out2[:, 14]).max():.2e}")
    print(f"exported {dst}  obs[1,63] -> actions[1,15]")


if __name__ == "__main__":
    main()
