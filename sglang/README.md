# SGLang visual understanding

Use a compatible local checkout of the [SGLang DLLM fork](https://github.com/hutm/sglang) providing `FastDiffuser` and `NemotronLabsDiffusionModel`. From this directory, with CUDA 12.9 available:

```bash
export SGLANG_SOURCE=/path/to/sglang-dllm
conda env create -f environment-sglang.yml
conda activate nlduni-sglang
git -C "$SGLANG_SOURCE" apply "$PWD/sglang_runtime.patch"
pip install --no-deps -e "$SGLANG_SOURCE/python"
python demo_sglang.py nvidia/NLD-Uni --image input.png --prompt "Describe this image."
```

A local model directory can replace `nvidia/NLD-Uni`. Use `--tensor-parallel-size` to select the GPU count, `--max-new-tokens` to limit the response, and `--threshold` to set decoding confidence. Text-to-image and image editing use `../infer.py` or `../gradio_demo.py`.
