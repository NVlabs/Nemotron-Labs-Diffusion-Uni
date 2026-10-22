"""Run NLD-Uni inference from a Hugging Face model ID or local model directory."""
import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='nvidia/NLD-Uni')
    parser.add_argument('--task', choices=['text-to-image', 'image-edit', 'understand', 'grounding'], required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output.png'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, help='Image sampling steps; uses model defaults when omitted')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.task in ('image-edit', 'grounding') and args.image is None:
        parser.error('--image is required for this task')
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    image = Image.open(args.image).convert('RGB') if args.image else None
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    options = {'seed': args.seed}
    if args.steps is not None:
        options['steps'] = args.steps
    if args.task == 'understand':
        print(model.generate(args.prompt, image=image, max_new_tokens=args.max_new_tokens))
        return
    if args.task == 'text-to-image':
        result = model.text_to_image(args.prompt, **options)
    elif args.task == 'image-edit':
        result = model.image_edit(image, args.prompt, **options)
    else:
        from grounding import ground_objects
        result, boxes = ground_objects(model, image, args.prompt, steps=1)
        import json
        print(json.dumps(boxes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
