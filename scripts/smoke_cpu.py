"""Build and step a task on CPU — the smoke test you can run without a GPU.

`uv run train ... --agent.max_iterations 5` is still the real gate, but it needs
CUDA: MuJoCo Warp's CPU build has no `cuda` device, so mjlab's `select_gpus()`
indexes an empty list and dies before iteration 0. On a laptop that leaves the
config errors it would have caught — a reward term reading a field that does
not exist, an observation that came out the wrong width, an event ordered
before the state it reads — undiscovered until a remote run starts.

This does the part that does not need a GPU: compile the scene, build every
manager, and step the env with random actions, asserting the observation width
and that nothing goes non-finite. It is slow (Warp on CPU) so it runs a handful
of envs for a handful of steps, which is enough — these failures all happen on
step one.

    uv run scripts/smoke_cpu.py Mjlab-ObjectPick-Flat-MicroDuck
    uv run scripts/smoke_cpu.py --num-envs 2 --steps 20 <TASK_ID>
"""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="task id, e.g. Mjlab-ObjectPick-Flat-MicroDuck")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--expect-obs", type=int, default=None,
                        help="assert the actor observation is exactly this wide")
    args = parser.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    obs, _ = env.reset()

    actor = obs["actor"] if isinstance(obs, dict) else obs
    print(f"actor obs: {tuple(actor.shape)}")
    for group, tensor in (obs.items() if isinstance(obs, dict) else []):
        print(f"  {group}: {tuple(tensor.shape)}")
    print(f"actions:   {env.action_manager.total_action_dim}")
    print(f"rewards:   {sorted(env.reward_manager.active_terms)}")

    if args.expect_obs is not None and actor.shape[-1] != args.expect_obs:
        print(f"FAIL: actor obs is {actor.shape[-1]}D, expected {args.expect_obs}D")
        return 1

    for step in range(args.steps):
        action = torch.zeros(
            env.num_envs, env.action_manager.total_action_dim, device=env.device
        ).uniform_(-0.3, 0.3)
        obs, reward, _, _, _ = env.step(action)
        actor = obs["actor"] if isinstance(obs, dict) else obs
        for name, tensor in (("obs", actor), ("reward", reward)):
            if not torch.isfinite(tensor).all():
                print(f"FAIL: non-finite {name} at step {step}")
                return 1

    print(f"OK: {args.steps} steps, finite obs and rewards")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
