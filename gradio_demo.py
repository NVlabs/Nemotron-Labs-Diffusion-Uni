"""Image generation, editing, understanding, and grounding UI for a local or Hugging Face NLDUni checkpoint."""
import argparse


def make_callbacks(model):
    def image_options(resolution, steps, refinement, guidance, temperature, seed,
                      top_k, top_p, shift):
        return dict(resolution=int(resolution), steps=int(steps),
                    refinement_steps=int(refinement), guidance_scale=float(guidance),
                    temperature=float(temperature), seed=int(seed), top_k=int(top_k),
                    top_p=float(top_p), shift=float(shift))

    def text_to_image(prompt, *options):
        if not prompt or not prompt.strip():
            raise ValueError('Enter an image prompt.')
        return model.text_to_image(prompt, **image_options(*options))

    def image_edit(image, prompt, image_guidance, crop, *options):
        if image is None or not prompt or not prompt.strip():
            raise ValueError('Choose a source image and enter an edit instruction.')
        return model.image_edit(image, prompt, image_guidance_scale=float(image_guidance),
                                crop_to_source=bool(crop), **image_options(*options))

    def understand(image, prompt, max_tokens, block_length, steps, threshold, temperature):
        if not prompt or not prompt.strip():
            raise ValueError('Enter a question.')
        return model.generate(prompt, image=image, max_new_tokens=int(max_tokens),
                              block_length=int(block_length), steps_per_block=int(steps),
                              threshold=float(threshold), temperature=float(temperature))

    def ground(image, tags, steps):
        from grounding import ground_objects
        return ground_objects(model, image, tags, steps)

    return text_to_image, image_edit, understand, ground


def build_demo(model):
    import gradio as gr

    t2i, edit, understand, ground = make_callbacks(model)

    def image_controls(guidance):
        with gr.Accordion('Sampling settings', open=False):
            resolution = gr.Dropdown([512, 1024], value=1024, label='Resolution')
            steps = gr.Slider(1, 256, value=64, step=1, label='Activation steps')
            refinement = gr.Dropdown([1, 4, 8, 16], value=4, label='Refinement stages')
            cfg = gr.Slider(0, 10, value=guidance, step=0.1, label='Text guidance')
            temperature = gr.Slider(0, 2, value=0.8, step=0.05, label='Temperature')
            seed = gr.Number(value=42, precision=0, label='Seed')
            top_k = gr.Number(value=0, precision=0, minimum=0, label='Top k (0 disables)')
            top_p = gr.Slider(0.01, 1, value=1, step=0.01, label='Top p')
            shift = gr.Slider(0.1, 10, value=5, step=0.1, label='Schedule shift')
        return [resolution, steps, refinement, cfg, temperature, seed, top_k, top_p, shift]

    with gr.Blocks(title='NLDUni', analytics_enabled=False) as demo:
        gr.Markdown('# NLDUni\nGenerate images, edit a source image, ask a visual question, or locate objects.')
        with gr.Tab('Text to image'):
            prompt = gr.Textbox(label='Prompt', lines=3)
            options = image_controls(3.0)
            button = gr.Button('Generate')
            output = gr.Image(label='Generated image', type='pil')
            button.click(t2i, [prompt, *options], output,
                         concurrency_id='model', concurrency_limit=1)
        with gr.Tab('Image editing'):
            source = gr.Image(label='Source image', type='pil')
            prompt = gr.Textbox(label='Edit instruction', lines=3)
            guidance = gr.Slider(0, 10, value=2.0, step=0.1, label='Image guidance')
            crop = gr.Checkbox(value=True, label='Restore source dimensions')
            options = image_controls(1.6)
            button = gr.Button('Edit')
            output = gr.Image(label='Edited image', type='pil')
            button.click(edit, [source, prompt, guidance, crop, *options], output,
                         concurrency_id='model', concurrency_limit=1)
        with gr.Tab('Visual understanding'):
            source = gr.Image(label='Image (optional for text-only questions)', type='pil')
            prompt = gr.Textbox(label='Question', lines=3)
            with gr.Accordion('Generation settings', open=False):
                max_tokens = gr.Number(value=16000, precision=0, minimum=1, label='Maximum new tokens')
                block_length = gr.Number(value=32, precision=0, minimum=1, label='Block length')
                steps = gr.Number(value=32, precision=0, minimum=1, label='Steps per block')
                threshold = gr.Slider(0, 1, value=0.9, step=0.01, label='Confidence threshold')
                temperature = gr.Slider(0, 2, value=0, step=0.05, label='Temperature')
            button = gr.Button('Answer')
            output = gr.Textbox(label='Answer', lines=8)
            button.click(understand, [source, prompt, max_tokens, block_length, steps, threshold, temperature],
                         output, concurrency_id='model', concurrency_limit=1)
        with gr.Tab('Object grounding'):
            source = gr.Image(label='Image', type='pil')
            tags = gr.Textbox(label='Object tags', lines=3,
                              placeholder='dog, red collar',
                              info='Enter one tag, or multiple tags separated by commas or new lines. One box is predicted per tag.')
            steps = gr.Dropdown([1, 2, 4], value=1, label='Grounding steps')
            button = gr.Button('Locate objects')
            output = gr.Image(label='Located objects', type='pil')
            boxes = gr.JSON(label='Boxes (original-image pixels and model coordinates)')
            button.click(ground, [source, tags, steps], [output, boxes],
                         concurrency_id='model', concurrency_limit=1)
    return demo.queue(default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='nvidia/NLD-Uni',
                        help='Local checkpoint directory or Hugging Face model ID')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    build_demo(model).launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == '__main__':
    main()
