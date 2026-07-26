# Policy Adapters

Policies implement:

```python
class Policy:
    def reset(self, task=None, seed=None): ...
    def act(self, observation): ...
    def close(self): ...
```

Built-ins include `random`, `scripted_oracle`, `bc_policy`, `lerobot`, `robomimic`, `diffusion`, and `openvla`.

The scripted oracle uses the repo-local ManiSkill scripted heuristic by default and can be overridden with `NYSSA_SCRIPTED_ORACLE_POLICY`. The BC policy loads `NYSSA_BC_CHECKPOINT` or `checkpoints/bc_policy.json` by default and can be overridden with `NYSSA_BC_POLICY`. RoboMimic loads `NYSSA_ROBOMIMIC_CHECKPOINT` or `checkpoints/robomimic_policy.pth` by default. LeRobot loads `NYSSA_LEROBOT_POLICY_PATH` or `checkpoints/lerobot_policy` by default. Diffusion and OpenVLA require a factory from the environment when invoked by name from the CLI.

```bash
NYSSA_SCRIPTED_ORACLE_POLICY=my_project.policies:create_scripted_oracle uv run nyssa run --policy scripted_oracle ...
NYSSA_BC_CHECKPOINT=checkpoints/bc_policy.json uv run nyssa run --policy bc_policy ...
NYSSA_BC_POLICY=my_project.policies:create_bc_policy uv run nyssa run --policy bc_policy ...
NYSSA_ROBOMIMIC_CHECKPOINT=checkpoints/robomimic_policy.pth uv run nyssa run --policy robomimic ...
NYSSA_LEROBOT_POLICY_PATH=checkpoints/lerobot_policy uv run nyssa run --policy lerobot ...
NYSSA_LEROBOT_POLICY=my_project.policies:create_lerobot_policy uv run nyssa run --policy lerobot ...
NYSSA_ROBOMIMIC_POLICY=my_project.policies:create_robomimic_policy uv run nyssa run --policy robomimic ...
NYSSA_DIFFUSION_POLICY=my_project.policies:create_diffusion_policy uv run nyssa run --policy diffusion ...
NYSSA_OPENVLA_POLICY=my_project.policies:create_openvla_policy uv run nyssa run --policy openvla ...
```

Install all supported policy stacks as part of the canonical environment:

```bash
uv sync --extra all --extra dev
```

For an existing lightweight environment, add only the policy stacks while
preserving its simulator dependencies:

```bash
uv sync --inexact --extra lerobot --extra robomimic --extra vla --extra diffusion
```

The `vla` extra covers common PyTorch/Transformers dependencies. Install OpenVLA model code and checkpoints from the upstream project when running a real OpenVLA policy.

Train the repo-local linear BC baseline:

```bash
uv run nyssa train-bc runs/scripted_oracle/episodes.json --out checkpoints/bc_policy.json
```

Export demos for robomimic and start upstream robomimic training:

```bash
uv run nyssa export --run runs/scripted_oracle --format robomimic --out runs/scripted_oracle/robomimic.hdf5
uv run nyssa train-robomimic --config configs/policies/robomimic_bc_flat.json
```
