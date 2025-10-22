from transformers import AutoProcessor
from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration, SAMR1ForConditionalGeneration_qwen2p5
import re
import torch
import json
from tqdm import tqdm
import os
import argparse
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from PIL import Image
from eval.utils import AverageMeter, Summary, intersectionAndUnionGPU
from src.open_r1.constants import system_prompt_registry, question_template_registry
import cv2
import glob
from qwen_vl_utils import smart_resize
from evaluate_res2 import save_mask_visualization, parse_float_sequence_within


def get_mask_from_json(json_path, img):
    """
    Read polygon annotation JSON file and generate mask.

    Returns:
        mask (np.uint8): 0 for background, 1 for target, 255 for ignore regions.
        comments: annotation comments
        is_sentence: whether the annotation is sentence-level
    """
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.shape[:2]

    # Sort polygons by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  # Deprecated, meaningless annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_inform = [valid_poly_list[s] for s in sort_index]

    # Generate GT mask
    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # Ignored in evaluation
        else:
            label_value = 1  # Foreground

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence


def resize_longest(image: Image.Image, longest_side_length):
    """
    Resize an image so that its longest side matches `longest_side_length`,
    keeping aspect ratio.
    """
    original_width, original_height = image.size

    if original_width > original_height:
        scale_factor = longest_side_length / original_width
    else:
        scale_factor = longest_side_length / original_height

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    return image.resize((new_width, new_height))


class ReasonSegVal(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(self, base_image_dir, if_cot=False):
        if if_cot:
            self.images = sorted(glob.glob(os.path.join(base_image_dir, "reason_seg", "ReasonSeg", "val", "*.jpg")))
            self.question_template = question_template_registry["samr1_v4"]
        else:
            self.images = sorted(glob.glob(os.path.join(base_image_dir, "reason_seg", "ReasonSeg", "test", "*.jpg")))
            self.question_template = question_template_registry["pr1_grounding"]

        self.system_template = system_prompt_registry["default"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        json_path = image_path.replace(".jpg", ".json")

        # Prepare LLM input image
        image = Image.open(image_path).convert(mode="RGB")
        image = resize_longest(image, longest_side_length=640)
        llm_image = image

        # Prepare SAM input image
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        sam_image = cv2.resize(image_cv, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(sam_image).permute(2, 0, 1).contiguous()
        sam_image = (sam_image - self.pixel_mean) / self.pixel_std

        mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image_cv)
        problems = [sampled_sents[0]]

        message = [
            {"role": "system", "content": self.system_template},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.question_template.format(question=problems[0])}
            ]},
        ]

        masks = torch.from_numpy(np.stack([mask_json], axis=0))
        masks = (masks == 1).float()

        return {
            'image': llm_image,
            "messages": [message],
            "sam_image": sam_image,
            "mask": masks[0],
            "problems": problems,
            "box": [],
            "image_path": image_path
        }


class RefCOCOEvaluator:
    """Evaluator class for ReasonSeg dataset"""
    def __init__(self, args):
        self.args = args
        self.dtype = torch.bfloat16
        if args.coord_norm_type == "qwen2p5vl":
            self.model = SAMR1ForConditionalGeneration_qwen2p5.from_pretrained(
                args.model_path, torch_dtype=self.dtype, attn_implementation="flash_attention_2"
            ).cuda()
        elif args.coord_norm_type == "qwen2vl":
            self.model = SAMR1ForConditionalGeneration.from_pretrained(
                args.model_path, torch_dtype=self.dtype, attn_implementation="flash_attention_2"
            ).cuda()

        self.processor = AutoProcessor.from_pretrained(args.model_path)
        input_size = 1024
        self._bb_feat_sizes = [
            (input_size // 4, input_size // 4),
            (input_size // 8, input_size // 8),
            (input_size // 16, input_size // 16),
        ]

    @staticmethod
    def remove_last_think_part(text):
        """Remove the last <think>...</think> block from the string."""
        matches = list(re.finditer(r'<think>.*?</think>\n?', text, flags=re.DOTALL))
        if not matches:
            return text
        start, end = matches[-1].span()
        return text[:start] + text[end:]

    @torch.no_grad()
    def evaluate_single(self, input_data):
        """Evaluate segmentation for a single image."""
        messages = input_data["messages"]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

        image_inputs = [input_data["image"]] * len(texts)
        inputs = self.processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to(
            device="cuda", dtype=self.dtype
        )

        llm_out = self.model.generate(max_length=768, use_cache=True, do_sample=False, **inputs)
        completion_text = self.processor.batch_decode(llm_out, skip_special_tokens=True)

        new_attention_mask = torch.ones_like(llm_out, dtype=torch.int64)
        pos = torch.where(llm_out == self.processor.tokenizer.pad_token_id)
        new_attention_mask[pos] = 0
        inputs.update({"input_ids": llm_out, "attention_mask": new_attention_mask})
        inputs.update({"sam_images": input_data["sam_image"].unsqueeze(0).repeat(len(texts), 1, 1, 1)})

        output, low_res_masks = self.model(output_hidden_states=True, use_learnable_query=True, **inputs)

        pred_masks = [self.model.postprocess_masks(mask, orig_hw=input_data["mask"].shape[-2:]) for mask in low_res_masks]
        pred_masks = torch.cat(pred_masks, dim=0)
        pred_masks = (pred_masks[:, 0] > 0).int()

        masks_list = input_data["mask"].int().unsqueeze(0).repeat(pred_masks.size(0), 1, 1)
        intersection, union, acc_iou = 0.0, 0.0, 0.0
        index = 0

        for mask_i, output_i in zip(masks_list, pred_masks):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous().cuda(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            iou = intersection_i / (union_i + 1e-5)
            mask_iou = iou[1].cpu().item()

            pred_box = parse_float_sequence_within(completion_text[index])

            if self.args.vis and mask_iou > 0.8:
                good_dir = f'{self.args.model_path}/evaluations/reasonseg_good'
                os.makedirs(good_dir, exist_ok=True)
                save_path = os.path.join(good_dir, f"{index}th_{mask_iou:.4f}.png")
                save_mask_visualization(input_data["image"], pred_masks[index].cpu(), mask_i.cpu(), save_path, pred_box)
                with open(os.path.join(good_dir, f"{index}th_{mask_iou:.4f}.txt"), "w") as f:
                    f.write(f'{completion_text[index]}')

            index += 1
            acc_iou += iou
            acc_iou[union_i == 0] += 1.0  # Perfect score if no object present

        intersection = intersection.cpu().numpy() / len(masks_list)
        union = union.cpu().numpy() / len(masks_list)
        acc_iou = acc_iou.cpu().numpy() / len(masks_list)
        return intersection, union, acc_iou, len(masks_list)


def main(args):
    os.makedirs(f"{args.model_path}/evaluations", exist_ok=True)

    dist.init_process_group('nccl', init_method="env://")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    evaluator = RefCOCOEvaluator(args)

    dataset = ReasonSegVal(base_image_dir=args.image_dir, if_cot=args.cot)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=rank)
    dataloader = DataLoader(dataset, 1, False, sampler=sampler, collate_fn=lambda batch: list(batch))

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    for batch_data in tqdm(dataloader):
        assert len(batch_data) == 1, "Only batch_size=1 is supported"
        intersection, union, acc_iou, num_mask = evaluator.evaluate_single(batch_data[0])
        intersection_meter.update(intersection, n=num_mask)
        union_meter.update(union, n=num_mask)
        acc_iou_meter.update(acc_iou, n=num_mask)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-8)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    print(f"model path is: {args.model_path}")
    print(f"evaluation on reasonseg: giou={giou}, ciou={ciou}")
    with open(f"{args.model_path}/evaluations/reason_seg.txt", "w") as f:
        f.write(f"model path: {args.model_path}, evaluation on ReasonSeg Test: giou={giou}, ciou={ciou}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of samples (debugging)")
    parser.add_argument("--coord_norm_type", type=str, default="qwen2p5vl", choices=["qwen2vl", "qwen2p5vl"], help="Coordinate normalization type")
    parser.add_argument("--image_dir", type=str, default="./datasets", help="Path to reasonseg dir")
    parser.add_argument("--cot", action="store_true", help="Enable chain-of-thought prompt")
    parser.add_argument("--vis", action="store_true", help="Visualize segmentation results")
    args = parser.parse_args()
    print("args:", args)
    main(args)
