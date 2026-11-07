"""Download only pinned processor/tokenizer assets, then verify real multimodal masking."""

from transformers import AutoProcessor

from wafervlm.core import LABELS
from wafervlm.data import render_map
from wafervlm.modeling import MODEL_ID, MODEL_REVISION, encode, mask_answer, messages


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    image = render_map([[0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 1, 1], [0, 1, 1, 0]])
    prompt = encode(processor, image)
    assert "pixel_values" in prompt and "image_grid_thw" in prompt
    text = processor.apply_chat_template(messages(image), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    assert text.endswith("<think>\n\n</think>\n\n"), "Expected explicit non-thinking assistant prefix"
    for answer in LABELS:
        full = encode(processor, image, answer)
        labels = mask_answer(full["input_ids"], prompt["input_ids"])
        visible = full["input_ids"][labels != -100]
        decoded = processor.tokenizer.decode(visible, skip_special_tokens=True).strip()
        assert decoded == answer, (decoded, answer)
        assert full["input_ids"].shape[-1] <= 1024
    print(f"PASS: real processor, all {len(LABELS)} labels, image tensors, exact answer-only masking")
    print(f"Model revision: {MODEL_REVISION}")


if __name__ == "__main__":
    main()
