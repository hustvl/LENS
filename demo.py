import torch
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
from transformers import AutoProcessor, logging
from qwen_vl_utils import smart_resize
from torchvision.transforms.functional import resize as resize_api

from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration_qwen2p5
from src.open_r1.constants import system_prompt_registry, question_template_registry
from src.open_r1.utils import parse_float_sequence_within

logging.set_verbosity_error()

# ----------- Model Load -------------
model_path = './weights/stage2/qwen2p5_refcoco'

model = SAMR1ForConditionalGeneration_qwen2p5.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    ignore_mismatched_sizes=True
).cuda()
processor = AutoProcessor.from_pretrained(model_path)


# ----------- Core functions -------------
def preprocess(image_path, instruction):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024
    question_template = question_template_registry["pr1_grounding"]
    system_template = system_prompt_registry["default"]
    
    image = Image.open(image_path).convert(mode="RGB")
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height, 
        width, 
        28, 
        max_pixels=1000000
    )
    llm_image = image.resize((resized_width, resized_height))

    sam_image = resize_api(image, (sam_img_size, sam_img_size))
    sam_image = torch.from_numpy(np.array(sam_image)).permute(2, 0, 1).contiguous()
    sam_image = (sam_image - pixel_mean) / pixel_std

    message = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question_template.format(question=instruction)}
        ]},
    ]

    return {
        "image": llm_image,
        "message": message,
        "sam_image": sam_image,
        "ori_hw": (height, width),
        "hw": (resized_height, resized_width)
    }


def evaluate_single(input_data, model, processor):
    message = input_data["message"]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
    image_inputs = [input_data["image"]] * len(texts)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device="cuda", dtype=torch.bfloat16)

    llm_out = model.generate(
        max_new_tokens=256, 
        use_cache=True, 
        do_sample=False, 
        **inputs
    )
    completion_text = processor.batch_decode(llm_out, skip_special_tokens=True)
    new_attention_mask = torch.ones_like(llm_out, dtype=torch.int64)
    pos = torch.where(llm_out == processor.tokenizer.pad_token_id)
    new_attention_mask[pos] = 0
    inputs.update({"input_ids": llm_out, "attention_mask": new_attention_mask})

    inputs.update({"sam_images": input_data["sam_image"].unsqueeze(0).repeat(len(texts), 1, 1, 1)})
    _, low_res_mask = model(output_hidden_states=True, use_learnable_query=True, **inputs)

    pred_mask = model.postprocess_masks(
        low_res_mask[0],
        orig_hw=input_data["ori_hw"],
    )
    pred_mask = (pred_mask[:, 0] > 0).int()

    return pred_mask, completion_text


def rescale_box(pred_box, from_hw, to_hw):
    from_h, from_w = from_hw
    to_h, to_w = to_hw
    scale_w = to_w / from_w
    scale_h = to_h / from_h
    x1, y1, x2, y2 = pred_box
    return [x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h]


def is_valid_box(box, image_hw=None):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2 or min(x1, y1, x2, y2) < 0:
        return False
    if image_hw:
        h, w = image_hw
        if x2 > w or y2 > h:
            return False
    return True

# ----------- Gradio UI Function -------------
def run_segmentation(image: Image.Image, instruction: str):
    image = image.convert("RGB")
    tmp_path = "/tmp/tmp_input.jpg"
    image.save(tmp_path)

    input_data = preprocess(tmp_path, instruction)
    mask, completion_text = evaluate_single(input_data, model, processor)

    pred_box = parse_float_sequence_within(completion_text[0])
    resized_hw = input_data["hw"]
    ori_hw = input_data["ori_hw"]
    box = rescale_box(pred_box, resized_hw, ori_hw)

    mask = mask.squeeze(0).to('cpu').numpy().astype('uint8')
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 3] = (mask > 0) * 128

    image_rgba = image.convert('RGBA')
    image_np = np.array(image_rgba)
    composite = Image.alpha_composite(Image.fromarray(image_np), Image.fromarray(overlay))

    if is_valid_box(box, ori_hw):
        draw = ImageDraw.Draw(composite)
        draw.rectangle(box, outline="green", width=4)

    return composite, completion_text[0]


# ----------- Launch Web UI -------------
demo = gr.Interface(
    fn=run_segmentation,
    inputs=[
        gr.Image(type="pil", label="upload image"),
        gr.Textbox(lines=1, label="Instruction（i.e. left person / man in green）")
    ],
    outputs=[
        gr.Image(type="pil", label="Segmentation Output"),
        gr.Textbox(label="Reasoning Output")
    ],
    title="LENS Instruct Segmentation",
    description="Instruct+Image->Box+Mask",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
