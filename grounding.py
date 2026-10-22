"""Object grounding for the exported semantic-tree model (no source-repo imports)."""
import importlib
import re

import torch
from PIL import Image, ImageDraw


def parse_tags(text):
    """Accept a tag or comma/newline-separated tags, preserving input order."""
    tags = [part.strip() for part in re.split(r'[,\n]', text or '') if part.strip()]
    if not tags:
        raise ValueError('Enter at least one object tag.')
    if len(tags) > 16:
        raise ValueError('Use at most 16 tags per request.')
    if any('<' in tag or '>' in tag for tag in tags):
        raise ValueError('Tags must be plain text, without token markup.')
    return tags


def _prompt_parts(tags, mask):
    # These strings are the checkpoint's grounding prompt format.
    slot = '<LOC_BEGIN>' + mask * 4 + '<LOC_END>'
    if len(tags) == 1:
        question = f' Please locate {tags[0]} in this image. Give bounding boxes in LOC format.'
        answer = slot
    else:
        question = ' Please describe this image with dense localized caption. Give bounding boxes in LOC format.'
        objects = [f'<box_p> {tag} </box_p>{slot}' for tag in tags]
        answer = 'There are following objects in the scene. ' + ', '.join(objects)
    return '<image>\n' + question, answer


def _image_prompt(model, image, tags):
    # Reuse the already-loaded HF model's own image/tokenizer utilities. This also
    # works when from_pretrained loaded a remote model into the HF module cache.
    api = importlib.import_module(type(model).__module__)
    mask = model.tokenizer.convert_ids_to_tokens(model.mask_token_id)
    question, answer = _prompt_parts(tags, mask)
    conversation = api.Conversation()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], answer)
    images = api.process_images([image], model.encoder.vision_tower.image_processor, model.config)
    grid = images[0][1][0]
    shape = (int(grid[1]) // 2, int(grid[2]) // 2)
    rendered = api.fix_one_prompt(conversation.get_prompt(), model.config, [shape])
    ids = api.tokenizer_image_token(rendered, model.tokenizer, return_tensors='pt')
    ids = ids.unsqueeze(0).to(model.device)
    embeds = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, images)[4]
    return ids, embeds


def _fill_coordinates(model, ids, embeddings, count, steps):
    """Fill only coordinate slots, reading a fixed causal prefix KV cache."""
    slots = ids[0].eq(model.mask_token_id).nonzero(as_tuple=True)[0]
    if len(slots) != count * 4:
        raise ValueError('The tokenized prompt does not contain four coordinate slots per tag.')
    boundary = int(slots[0]) - 1  # shifted head: coordinate t reads hidden state t-1
    if boundary < 1:
        raise ValueError('Grounding requires a nonempty image and instruction prefix.')
    attention = [layer.self_attn for layer in model.encoder.layers]
    saved = [(module, module.diffusion_lm, getattr(module, 'mode', None)) for module in attention]
    try:
        for module in attention:
            module.diffusion_lm = False
            if hasattr(module, 'mode'):
                module.mode = 'bidirectional'
        prefix = model.encoder(inputs_embeds=embeddings[:, :boundary], use_cache=True,
                               use_causal_mask=True, overwrite_attn_impl='flash_attn')
        cache = prefix.past_key_values
        if cache is None:
            raise RuntimeError('The checkpoint did not return a causal prefix cache.')
        for module in attention:
            module.diffusion_lm = True
        suffix = embeddings[:, boundary:].clone()
        offsets = slots - boundary
        pending = torch.ones(len(slots), dtype=torch.bool, device=ids.device)
        chosen = torch.empty(len(slots), dtype=torch.long, device=ids.device)
        for _ in range(steps):
            hidden = model.encoder(inputs_embeds=suffix, past_key_values=cache, use_cache=False,
                                   overwrite_attn_impl='flash_attn').last_hidden_state
            active = pending.nonzero(as_tuple=True)[0]
            logits = model.diffusion_head(hidden[0, offsets[active] - 1])
            scores, candidates = logits.max(dim=-1)
            # Highest-logit slots are committed first, matching the checkpoint recipe.
            selected = scores.topk(len(slots) // steps).indices
            destinations = active[selected]
            chosen[destinations] = candidates[selected]
            suffix[0, offsets[destinations]] = model.get_input_embeddings()(chosen[destinations])
            pending[destinations] = False
        return chosen.reshape(count, 4).cpu().tolist()
    finally:
        for module, diffusion, mode in saved:
            module.diffusion_lm = diffusion
            if hasattr(module, 'mode'):
                module.mode = mode


def pixel_box(coordinates, size):
    """Undo the 1024-unit centered-square coordinate convention, then clip."""
    width, height = size
    side = max(size)
    offset = ((side - width) / 2, (side - height) / 2)
    return [round(max(0., min(float(size[i % 2]), value * side / 1024 - offset[i % 2])), 2)
            for i, value in enumerate(coordinates)]


@torch.inference_mode()
def ground_objects(model, image, tags, steps=1):
    """Return a labeled PIL image and per-tag LOC/pixel xyxy records."""
    if not isinstance(image, Image.Image):
        raise ValueError('Upload an image to ground objects.')
    tags = parse_tags(tags)
    if steps not in (1, 2, 4):
        raise ValueError('Grounding steps must be 1, 2, or 4.')
    image = image.convert('RGB')
    ids, embeddings = _image_prompt(model, image, tags)
    predictions = _fill_coordinates(model, ids, embeddings, len(tags), int(steps))
    records = []
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    palette = ('#e6194b', '#008040', '#4363d8', '#f58231', '#911eb4', '#008b8b')
    for index, (tag, tokens) in enumerate(zip(tags, predictions, strict=True)):
        pieces = model.tokenizer.convert_ids_to_tokens(tokens)
        matches = [re.fullmatch(r'<LOC_(\d+)>', piece) for piece in pieces]
        if not all(matches):
            raise RuntimeError(f'No valid coordinate prediction for {tag!r}: {pieces}. Try a more specific tag.')
        coordinates = [int(match.group(1)) for match in matches]
        if any(value > 1024 for value in coordinates):
            raise RuntimeError(f'Coordinates for {tag!r} are outside the model coordinate range.')
        box = pixel_box(coordinates, image.size)
        valid = box[0] < box[2] and box[1] < box[3]
        color = palette[index % len(palette)]
        record = dict(tag=tag, loc_xyxy=coordinates, pixel_xyxy=box, valid=valid, color=color)
        records.append(record)
        if valid:
            draw.rectangle(box, outline=color, width=max(2, max(image.size) // 300))
            label = f'{index + 1}. {tag}'
            font_size = max(12, max(image.size) // 60)
            text_box = draw.textbbox((0, 0), label, font_size=font_size)
            x = max(0, min(box[0], image.width - text_box[2] - 6))
            y = max(0, box[1] - text_box[3] - 6)
            draw.rectangle((x, y, x + text_box[2] + 6, y + text_box[3] + 6), fill=color)
            draw.text((x + 3, y + 2), label, fill='white', font_size=font_size)
    return canvas, records
