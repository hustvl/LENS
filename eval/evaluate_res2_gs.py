import json
import os
import argparse
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pycocotools import mask as maskUtils
from torchvision.transforms.functional import resize as resize_api
import torchvision.transforms.functional as TF
from transformers import AutoProcessor

from qwen_vl_utils import process_vision_info, smart_resize
from eval.utils import AverageMeter, Summary, intersectionAndUnionGPU
from src.open_r1.constants import system_prompt_registry, question_template_registry
from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration, SAMR1ForConditionalGeneration_qwen2p5


class GroundingSuiteEvalDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(
        self,
        anno_path="./datasets/GroundingSuite-Eval.jsonl",
        image_dir="./datasets/coco/unlabeled2017",
        coord_norm_type="qwen2p5vl",
        if_cot=False,
    ):
        self.coord_norm_type = coord_norm_type
        self.image_dir = image_dir
        self.list_data_dict = []

        if if_cot:
            self.question_template = question_template_registry["samr1_v4"]
        else:
            self.question_template = question_template_registry["pr1_grounding"]
        self.system_template = system_prompt_registry["default"]

        # Load JSONL annotations
        with open(anno_path, "r") as f:
            for line in f:
                if line.strip():
                    self.list_data_dict.append(json.loads(line))

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        ref = self.list_data_dict[idx]

        image_path = os.path.join(self.image_dir, ref["image_path"])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        resized_height, resized_width = smart_resize(
            height,
            width,
            28,
            max_pixels=1000000
        )
        llm_image = image.resize((resized_width, resized_height))

        sam_image = resize_api(image, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(np.array(sam_image)).permute(2, 0, 1).contiguous()
        sam_image = (sam_image - self.pixel_mean) / self.pixel_std

        # Convert bounding box format
        xmin, ymin, xmax, ymax = ref["box"]
        if self.coord_norm_type == "qwen2vl":
            box = [
                round(xmin / width * 1000),
                round(ymin / height * 1000),
                round(xmax / width * 1000),
                round(ymax / height * 1000)
            ]
        elif self.coord_norm_type == "qwen2p5vl":
            box = [
                round(xmin / width * resized_width),
                round(ymin / height * resized_height),
                round(xmax / width * resized_width),
                round(ymax / height * resized_height)
            ]

        # Parse segmentation (already in RLE format)
        segmentation = ref["segmentation"]
        if isinstance(segmentation, list):  # multiple RLEs
            for i in range(len(segmentation)):
                if not isinstance(segmentation[i]["counts"], bytes):
                    segmentation[i]["counts"] = segmentation[i]["counts"].encode()
        else:
            if not isinstance(segmentation["counts"], bytes):
                segmentation["counts"] = segmentation["counts"].encode()
            segmentation = [segmentation]

        mask = maskUtils.decode(segmentation)
        mask = np.sum(mask, axis=2) if mask.ndim == 3 else mask
        mask = torch.from_numpy(mask.astype(np.uint8))

        caption = ref["caption"]
        problem = caption.replace('is', '').replace('are', '').lower()
        problems = [problem]
        message = [
            {"role": "system", "content": self.system_template},
            {"role": "user", "content": [
                {"type": "image", "image": llm_image},
                {"type": "text", "text": self.question_template.format(question=problems[0])}
            ]},
        ]
        return {
            "image": llm_image,
            "messages": [message],
            "sam_image": sam_image,
            "mask": mask,
            "problems": problems,
            "box": box
        }


def save_mask_visualization(pil_image, pred_mask, gt_mask, save_path):
    """
    Save as a horizontally concatenated image: Original | GT Overlay | Pred Overlay
    Implemented using only PIL and torch, without matplotlib.

    Args:
        pil_image: PIL.Image, RGB image
        pred_mask: (H, W) int tensor, values 0/1
        gt_mask: (H, W) int tensor, values 0/1
        save_path: str, save path
    """
    image = TF.to_tensor(pil_image) * 255
    image = image.to(torch.uint8)

    # Resize masks to image size (PIL.size is (W,H), torch.Size is (H,W))
    size = pil_image.size[::-1]
    pred_mask = TF.resize(pred_mask.unsqueeze(0).float(), size, interpolation=Image.NEAREST)[0].int()
    gt_mask = TF.resize(gt_mask.unsqueeze(0).float(), size, interpolation=Image.NEAREST)[0].int()

    def overlay_mask(image, mask, color):
        color_layer = torch.zeros_like(image)
        if color == "green":
            color_layer[1] = 255
        elif color == "red":
            color_layer[0] = 255
        mask = mask.bool().unsqueeze(0)
        overlay = image * 0.6 + torch.where(mask, color_layer * 0.4, torch.zeros_like(image))
        return overlay.clamp(0, 255).byte()

    gt_overlay = overlay_mask(image, gt_mask, "green")
    pred_overlay = overlay_mask(image, pred_mask, "red")

    img_ori = TF.to_pil_image(image)
    img_gt = TF.to_pil_image(gt_overlay)
    img_pred = TF.to_pil_image(pred_overlay)

    total_width = img_ori.width * 3
    total_height = img_ori.height
    merged_img = Image.new("RGB", (total_width, total_height))

    merged_img.paste(img_ori, (0, 0))
    merged_img.paste(img_gt, (img_ori.width, 0))
    merged_img.paste(img_pred, (img_ori.width * 2, 0))

    merged_img.save(save_path)


class RefCOCOEvaluator:
    """Evaluator class for RefCOCO-style datasets"""
    def __init__(self, args):
        self.args = args
        self.dtype = torch.bfloat16
        if args.coord_norm_type == "qwen2p5vl":
            self.model = SAMR1ForConditionalGeneration_qwen2p5.from_pretrained(
                args.model_path,
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2",
            ).cuda()
        elif args.coord_norm_type == "qwen2vl":
            self.model = SAMR1ForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2",
            ).cuda()
        self.processor = AutoProcessor.from_pretrained(args.model_path)

    @torch.no_grad()
    def evaluate_single(self, input_data):
        """
        Evaluate one mask for one image.
        Each mask contains several referring expressions.
        """
        messages = input_data["messages"]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device="cuda", dtype=self.dtype)

        llm_out = self.model.generate(
            max_length=768,
            use_cache=True,
            do_sample=False,
            **inputs
        )
        completion_text = self.processor.batch_decode(llm_out, skip_special_tokens=True)

        new_attention_mask = torch.ones_like(llm_out, dtype=torch.int64)
        pos = torch.where(llm_out == self.processor.tokenizer.pad_token_id)
        new_attention_mask[pos] = 0
        inputs.update(
            {
                "input_ids": llm_out,
                "attention_mask": new_attention_mask
            }
        )
        inputs.update({"sam_images": input_data["sam_image"].unsqueeze(0).repeat(len(texts), 1, 1, 1)})
        _, low_res_masks = self.model(output_hidden_states=True, use_learnable_query=True, **inputs)

        pred_masks = [self.model.postprocess_masks(
            low_res_mask,
            orig_hw=input_data["mask"].shape[-2:],
        ) for low_res_mask in low_res_masks]
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
            acc_iou += iou
            acc_iou[union_i == 0] += 1.0  # no-object target

            if self.args.vis:
                mask_iou = iou[1].cpu().item()
                bad_dir = f'{self.args.model_path}/evaluations/bad_results_gs'
                os.makedirs(bad_dir, exist_ok=True)
                good_dir = f'{self.args.model_path}/evaluations/good_results_gs'
                os.makedirs(good_dir, exist_ok=True)

                image = input_data["image"]
                gt_mask = input_data["mask"].cpu()
                pred_mask = pred_masks[index].cpu()
                if mask_iou < 0.2:
                    print(f"mask_iou = {mask_iou}, completion = {completion_text[index]}")
                    save_path = os.path.join(bad_dir, f"bad_result_IoU_{mask_iou:.4f}.png")
                    save_mask_visualization(image, pred_mask, gt_mask, save_path)
                    with open(os.path.join(bad_dir, f"bad_result_IoU_{mask_iou:.4f}.txt"), "w") as f:
                        f.write(f'{completion_text[index]}')
                elif mask_iou > 0.9:
                    print(f"iou = {mask_iou}, completion = {completion_text[index]}")
                    save_path = os.path.join(good_dir, f"good_result_IoU_{mask_iou:.4f}.png")
                    save_mask_visualization(image, pred_mask, gt_mask, save_path)
                    with open(os.path.join(good_dir, f"good_result_IoU_{mask_iou:.4f}.txt"), "w") as f:
                        f.write(f'{completion_text[index]}')

            index += 1
        intersection = intersection.cpu().numpy() / len(masks_list)
        union = union.cpu().numpy() / len(masks_list)
        acc_iou = acc_iou.cpu().numpy() / len(masks_list)

        return intersection, union, acc_iou, len(masks_list)


def main(args):
    dist.init_process_group('nccl', init_method="env://")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    evaluator = RefCOCOEvaluator(args)
    dataset = GroundingSuiteEvalDataset(coord_norm_type=args.coord_norm_type, if_cot=args.cot, anno_path=args.anno_path, image_dir=args.image_dir)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=rank)
    dataloader = DataLoader(
        dataset,
        1,
        False,
        sampler=sampler,
        collate_fn=lambda batch: list(batch),
        num_workers=16)

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    for batch_data in tqdm(dataloader):
        assert len(batch_data) == 1, "Only batch_size=1 evaluation is supported"
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

    print("model path is: ", args.model_path)
    print(f"giou={giou}, ciou={ciou}")

    os.makedirs(f"{args.model_path}/evaluations", exist_ok=True)
    with open(f"{args.model_path}/evaluations/eval.txt", "w") as f:
        f.write(f"model path is: {args.model_path},\n giou={giou}, ciou={ciou}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of samples (for debugging)")
    parser.add_argument("--coord_norm_type", type=str, default="qwen2p5vl", choices=["qwen2vl", "qwen2p5vl"], help="Coordinate normalization type")
    parser.add_argument("--anno_path", type=str, default='./datasets/GroundingSuite-Eval.jsonl', help="Path to GroundingSuite Eval Annotation Json File")
    parser.add_argument("--image_dir", type=str, default='./datasets/coco/unlabeled2017', help="Path to COCO Unlabeled2017 Dir")
    parser.add_argument("--vis", action="store_true", help="Whether to visualize the results")
    parser.add_argument("--cot", action="store_true", help="Whether to use COT")
    args = parser.parse_args()
    main(args)
