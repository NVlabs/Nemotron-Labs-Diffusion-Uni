# Minimal SFT and online OPD

## Student environment

Requires CUDA 12.6 and a compatible NVIDIA driver.

```bash
conda env create -f environment.yml
conda activate nlduni-minimal-training
pip install --no-build-isolation flash-attn==2.8.0.post2
```

## Teacher environments

Use separate environments for the two teachers. Install a CUDA-enabled PyTorch build before the remaining packages.

```bash
python3.12 -m venv /envs/vlm-teacher
/envs/vlm-teacher/bin/pip install -r requirements-vlm-teacher.txt
python3.13 -m venv /envs/edit-teacher
/envs/edit-teacher/bin/pip install -r requirements-edit-teacher.txt
```

The VLM teacher downloads its fused Mamba and causal-convolution kernels on first use. For offline execution, set `LOCAL_KERNELS` to the downloaded package directories: `kernels-community/mamba-ssm=/cache/mamba:kernels-community/causal-conv1d=/cache/conv`.

## Launch

On a teacher node with eight GPUs:

```bash
export VLM_MODEL=/models/vlm-teacher
export EDIT_MODEL=Qwen/Qwen-Image-Edit-2511
export VLM_PYTHON=/envs/vlm-teacher/bin/python
export EDIT_PYTHON=/envs/edit-teacher/bin/python
bash run_teachers.sh
```

`VLM_MODEL` must be a compatible Nemotron Nano Omni checkpoint. Each teacher uses four GPUs. Teacher HTTP services are intended for a trusted private network.

On a student node with eight GPUs, from this directory:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 train.py \
  --model nvidia/NLD-Uni --teacher-host teacher-node --output training-output
```

The default config runs ten updates each of T2I SFT, VLM SFT, VLM OPD, and image-edit OPD using two synthetic examples per task. To use your own data, copy `configs/default.yaml`, change its dataset paths, and pass `--config your-config.yaml`. Dataset mixtures accept YAML lists of JSON, JSONL, Parquet, or Lance files. Rows use `prompt`, `answer`, `images`, and `target`; images may be paths or `archive.tar::member` references. Mixture paths resolve relative to their YAML file.
