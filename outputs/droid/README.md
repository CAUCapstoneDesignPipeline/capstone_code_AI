---
license: other
license_name: flux-kommunity-license
license_link: LICENSE.md
library_name: lerobot
pipeline_tag: robotics
base_model: black-forest-labs/flux-3-action-base
tags: [robotics, lerobot, droid, franka, flux, world-action-model, world-model, robot-learning, black-forest-labs, flux-3]
---

# FLUX 3 Action DROID

<video controls autoplay loop muted playsinline src="https://huggingface.co/black-forest-labs/flux-3-action-droid/resolve/main/assets/flux-3-action-demo.mp4"></video>

FLUX 3 Action is an open weights 7B world action model. It takes camera frames, the robot's state and a text instruction, and returns the next chunk of actions, denoised together with the next video frames. This repository is part of the [FLUX 3 Action collection](https://huggingface.co/collections/black-forest-labs/flux-3-action-6ab25aef555dd30ab86567f8).

For more information, read the [documentation](https://docs.bfl.ai/flux_3/flux3_action_overview).
The training and inference code is in [black-forest-labs/flux-action](https://github.com/black-forest-labs/flux-action) on GitHub.

Fine-tuned on [DROID](https://droid-dataset.github.io/), FLUX 3 Action places first on the RoboLab-120 benchmark at 42.92% task success.

## Evaluation

RoboLab-120 is 120 tabletop tasks in Isaac Sim, 10 trials each, on a DROID-style Franka setup; a trial succeeds only if the task is completed as instructed. The full board is on the [RoboLab leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html).

| Model | Type | Success | Parameters |
| --- | --- | --- | --- |
| FLUX 3 Action | WAM | 42.92% | 7B |
| Cosmos3-Nano-Policy | WAM | 36.8% | 16B |
| π0.5 | VLA | 28.0% | 3.3B |

WAM: predicts future frames and actions together. VLA: a vision-language model that outputs actions directly.

## Usage

Validated BF16 DROID policy. See [License](#license) for usage terms.
The root package contains the LeRobot config, model and saved processors.

```python
from lerobot.policies.flux3 import Flux3Policy
from lerobot.policies.factory import make_pre_post_processors

repo_id = "black-forest-labs/flux-3-action-droid"
policy = Flux3Policy.from_pretrained(repo_id)
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=repo_id)
```

Use the FLUX3 integration with shared-encoder Hub subfolder/revision support.
The VAE and text encoder load automatically from the pinned
[base repository](https://huggingface.co/black-forest-labs/flux-3-action-base).
LeRobot uses `config.json`; standalone FLUX Action uses `config.native.json`.
The standalone export loader must support that native config filename.

Optimization checkpoints are opt-in:

| Recipe | BF16 subfolder | FP8r subfolder |
|---|---|---|
| Base: 4 steps with guidance | root (omit subfolder) | `variants/fp8r` |
| Guidance-distilled: 4 steps | `variants/gd` | `variants/gd-fp8r` |
| Step-distilled: 1 step | `variants/sd` | `variants/sd-fp8r` |

[Standalone FLUX Action](https://github.com/black-forest-labs/flux-action/pull/3) supports all six packages.
BF16 packages also include LeRobot configs; native FP8r packages require FLUX Action.
LeRobot reads `config.json`; FLUX Action reads `config.native.json` when present, otherwise `config.json`.
The default LeRobot load fetches none of the variant weights.

## Sampling

Use four Cosmos UniPC steps, shift 5, video guidance 4 and action guidance 1.

## Precision

The DiT checkpoint contains BF16 tensors.

`lora.json` is an optional LeRobot task-LoRA training recipe; it contains no adapter weights.

## Observation and action contract

The native policy takes `task`, `state`, and three RGB cameras named `images.wrist`, `images.left`, and `images.right`. Camera order maps to the original `observation.images.wrist_image_left`, `observation.images.exterior_image_1_left`, and `observation.images.exterior_image_2_left`. Images are CHW 3x360x640 floats in [0,1], composed into a 544x736 canvas.

State contains seven arm joint positions in radians and gripper closed fraction. Outputs are absolute joint commands followed by gripper closed fraction (0 open, 1 closed), with 32 actions at 15 Hz. Normalization is identity. Gripper inversion and action scale 2 are applied inside the model.

The saved processor JSON files preserve the original LeRobot observation contract for adapters. The native policy implements its own preprocessing and does not automatically consume those files.
## Hardware

FLUX 3 Action runs in about 32 GB of GPU memory in bfloat16 on an NVIDIA H200; with `--quantize fp8 --offload-text-encoder` it fits 24 GB cards. Tested with Python 3.12, CUDA 12.8, torch 2.10.0 and transformers 5.16.1.

## Documentation

- [Run the model](https://docs.bfl.ai/flux_3/flux3_action_inference)
- [Fine-tune](https://docs.bfl.ai/flux_3/flux3_action_finetuning)
- [flux-action on GitHub](https://github.com/black-forest-labs/flux-action)

## Out-of-scope use

FLUX 3 Action outputs joint targets. Nothing in the model bounds joint velocity, force or workspace; the application must enforce those limits and keep a hardware stop within reach. Validate on a simulator or with the arm's safety limits engaged before running near people.

The model and its derivatives may not be used:

- In any way that violates applicable law.
- To control a machine in a way that endangers people, without human oversight and a means to stop it.
- For fully automated decision making or high risk applications that adversely impact an individual's legal rights or otherwise create or modify a binding, enforceable obligation.
- To harass, abuse, threaten, stalk, or bully individuals or groups of individuals.
- For the purpose of exploiting, harming or attempting to exploit or harm minors in any way.

Nothing contained in this model card should be interpreted as or deemed a restriction or modification to the license the model is released under.

## Responsible AI development

Black Forest Labs is committed to responsible model development and deployment. FLUX 3 Action outputs motor commands and, on request, predicted camera frames of the scene it is acting in. For information about our mitigations, evaluation processes and policies, see [Capable, Open, and Safe: Combating AI Misuse](https://bfl.ai/blog/capable-open-and-safe-combating-ai-misuse). To report safety concerns, contact [safety@blackforestlabs.ai](mailto:safety@blackforestlabs.ai).

## License

This model falls under the [FLUX Kommunity License v.1.0](LICENSE.md). The text encoder in [flux-3-action-base](https://huggingface.co/black-forest-labs/flux-3-action-base) is an unmodified copy of Qwen3-VL-4B-Instruct under Apache-2.0. The code in [flux-action](https://github.com/black-forest-labs/flux-action) has its own license.

## Trademarks and IP

This project may contain trademarks or logos for projects, products, or services. Use of Black Forest Labs and FLUX trademarks or logos in modified versions of this project must not cause confusion or imply sponsorship or endorsement. Any use of third-party trademarks, intellectual property or logos are subject to those third-party's policies.

## Citation

If you use FLUX 3 Action in your work, please cite:

```bibtex
@misc{bfl2026flux3action,
  author       = {{Black Forest Labs}},
  title        = {{FLUX} 3 Action},
  year         = {2026},
  url          = {https://bfl.ai/models/flux-3-action},
}