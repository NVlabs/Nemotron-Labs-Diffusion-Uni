from __future__ import annotations
from sglang.srt.managers.schedule_batch import Modality, MultimodalInputFormat, MultimodalProcessorOutput
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang_nlduni.models.nlduni_radio_h import NLDUniForCausalLM

class NLDUniRadioHProcessor(BaseMultimodalProcessor):
    models = [NLDUniForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        tokenizer = getattr(_processor, 'tokenizer', _processor)
        self._image_token_id = tokenizer.convert_tokens_to_ids('<image>')
        if self._image_token_id is None:
            self._image_token_id = getattr(tokenizer, 'unk_token_id', 0) or 0

    async def process_mm_data_async(self, image_data, input_text, request_obj=None, **kwargs):
        del request_obj, kwargs
        if not isinstance(input_text, list) or any((not isinstance(token_id, int) for token_id in input_text)):
            raise TypeError('NLDUni SGLang evaluation requires pre-tokenized input_ids')
        present_images = [item for item in image_data or [] if item is not None]
        if not present_images:
            return MultimodalProcessorOutput(mm_items=[], input_ids=list(input_text))
        mm_items = []
        for (image_index, payload) in enumerate(present_images):
            if not isinstance(payload, dict) or payload.get('format') not in {'processor_output', MultimodalInputFormat.PROCESSOR_OUTPUT.name, MultimodalInputFormat.PROCESSOR_OUTPUT}:
                raise TypeError(f'image_data[{image_index}] must be a processor_output dict')
            items = self.collect_mm_items_from_processor_output(payload, modality=Modality.IMAGE)
            if len(items) != 1:
                raise ValueError(f'image_data[{image_index}] produced {len(items)} image items')
            item = items[0]
            item.format = MultimodalInputFormat.PROCESSOR_OUTPUT
            item.set_pad_value()
            mm_items.append(item)
        return MultimodalProcessorOutput(mm_items=mm_items, input_ids=list(input_text), im_token_id=int(self._image_token_id))
