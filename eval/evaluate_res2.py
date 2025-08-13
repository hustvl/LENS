from transformers import AutoProcessor
from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration, SAMR1ForConditionalGeneration_qwen2p5
import torch
import re
from tqdm import tqdm
import os
import argparse
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pycocotools import mask as maskUtils
from src.open_r1.refer import REFER
import numpy as np
from qwen_vl_utils import smart_resize
from torchvision.transforms.functional import resize as resize_api
from eval.utils import AverageMeter, Summary, intersectionAndUnionGPU
from src.open_r1.constants import system_prompt_registry, question_template_registry
from src.open_r1.utils import parse_float_sequence_within, compute_iou
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF


def resize_longest(image: Image.Image, longest_side_length):
    """
    Resize the image so that the longest side is scaled to the specified length.
    Keep the aspect ratio.
    """
    original_width, original_height = image.size
    if original_width > original_height:
        scale_factor = longest_side_length / original_width
    else:
        scale_factor = longest_side_length / original_height
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    return image.resize((new_width, new_height))


class ReferSegEvaluationSet(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(
        self,
        val_dataset,
        base_image_dir="./datasets",
        coord_norm_type="qwen2vl",
        if_cot=False,
    ):
        splits = val_dataset.split(".")
        assert len(splits) == 3
        ds, splitBy, split = splits
        assert ds in ["refcoco", "refcoco+", "refcocog"]
        assert splitBy in ["unc", "umd"]
        assert split in ["val", "test", "testA", "testB"]

        self.coord_norm_type = coord_norm_type
        self.list_data_dict = []
        self.refer_annotations = {}
        self.base_image_dir = base_image_dir
        base_image_dir = os.path.join(base_image_dir, "refer_seg")

        refer_api = REFER(base_image_dir, ds, splitBy)
        ref_ids = refer_api.getRefIds(split=split)
        refs = refer_api.loadRefs(ref_ids=ref_ids)
        self.list_data_dict.extend(refs)
        self.refer_annotations.update(refer_api.Anns)

        if if_cot:
            self.question_template = question_template_registry["samr1_v4"]
        else:
            self.question_template = question_template_registry["pr1_grounding"]
        self.system_template = system_prompt_registry["default"]

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        ref = self.list_data_dict[idx]
        image_path = f"COCO_train2014_{ref['image_id']:012}.jpg"
        image_path = os.path.join(self.base_image_dir, "refer_seg/images/mscoco/images/train2014", image_path)

        image = Image.open(image_path).convert(mode="RGB")
        width, height = image.size
        resized_height, resized_width = smart_resize(height, width, 28, max_pixels=1000000)
        llm_image = image.resize((resized_width, resized_height))

        sam_image = resize_api(image, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(np.array(sam_image)).permute(2, 0, 1).contiguous()
        sam_image = (sam_image - self.pixel_mean) / self.pixel_std

        messages = []
        problems = []
        for item in ref["sentences"]:
            problem = item["sent"]
            problems.append(problem)
            message = [
                {"role": "system", "content": self.system_template},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.question_template.format(question=problem)}
                ]},
            ]
            messages.append(message)
        ann = self.refer_annotations[ref["ann_id"]]

        xmin, ymin, box_w, box_h = ann["bbox"]
        xmax = xmin + box_w
        ymax = ymin + box_h
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

        if len(ann["segmentation"]) == 0:
            mask = np.zeros((height, width)).astype(np.uint8)

        if type(ann["segmentation"][0]) == list:  # polygon
            rle = maskUtils.frPyObjects(ann["segmentation"], height, width)
        else:
            rle = ann["segmentation"]
            for i in range(len(rle)):
                if not isinstance(rle[i]["counts"], bytes):
                    rle[i]["counts"] = rle[i]["counts"].encode()
        mask = maskUtils.decode(rle)
        mask = np.sum(mask, axis=2)  # sometimes there are multiple binary maps
        mask = mask.astype(np.uint8)
        mask = torch.from_numpy(mask)

        return {
            "image": llm_image,
            "messages": messages,
            "sam_image": sam_image,
            "mask": mask,
            "problems": problems,
            "box": box,
            "image_id": ref['image_id']
        }


def save_mask_visualization(pil_image, pred_mask, gt_mask, save_path, box):
    """
    Save as two separate images: original and prediction overlay.
    No matplotlib dependency, only PIL and torch.
    """
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
    img_w, img_h = pil_image.size
    x2, y2 = min(x2, img_w - 1), min(y2, img_h - 1)
    box = (x1, y1, x2, y2)

    image = TF.to_tensor(pil_image) * 255
    image = image.to(torch.uint8)

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

    pred_overlay = overlay_mask(image, pred_mask, "red")
    img_ori = TF.to_pil_image(image)
    img_pred = TF.to_pil_image(pred_overlay)

    draw = ImageDraw.Draw(img_pred)
    if y2 >= y1 and x2 >= x1:
        draw.rectangle(box, outline="green", width=3)

    base_name, ext = os.path.splitext(save_path)
    img_ori.save(base_name + "_ori" + ext)
    img_pred.save(base_name + "_pred" + ext)


class RefCOCOEvaluator:
    """Evaluator class encapsulating the evaluation logic"""
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
        input_size = 1024
        self._bb_feat_sizes = [
            (input_size // 4, input_size // 4),
            (input_size // 8, input_size // 8),
            (input_size // 16, input_size // 16),
        ]

    @staticmethod
    def remove_last_think_part(text):
        matches = list(re.finditer(r'<think>.*?</think>\n?', text, flags=re.DOTALL))
        if not matches:
            return text
        start, end = matches[-1].span()
        return text[:start] + text[end:]

    @torch.no_grad()
    def evaluate_single(self, input_data):
        """
        Evaluate one mask of one image.
        Each mask contains several referring expressions.
        """
        messages = input_data["messages"]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        image_inputs = [input_data["image"]] * len(texts)
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
        inputs.update({"input_ids": llm_out, "attention_mask": new_attention_mask})
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
        box_ious = []

        for index, (mask_i, output_i) in enumerate(zip(masks_list, pred_masks)):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous().cuda(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            iou = intersection_i / (union_i + 1e-5)
            mask_iou = iou[1].cpu().item()

            pred_box = parse_float_sequence_within(completion_text[index])
            gt_box = input_data["box"]
            box_iou = compute_iou(pred_box, gt_box)
            box_ious.append(box_iou)

            if self.args.vis and mask_iou > 0.95:
                good_dir = f'{self.args.model_path}/evaluations/good'
                os.makedirs(good_dir, exist_ok=True)
                image = input_data["image"]
                image_id = input_data["image_id"]
                gt_mask = input_data["mask"].cpu()
                pred_mask = pred_masks[index].cpu()
                save_path = os.path.join(good_dir, f"{image_id}_{index}th_{mask_iou:.4f}.png")
                save_mask_visualization(image, pred_mask, gt_mask, save_path, pred_box)
                completion_path = os.path.join(good_dir, f"{image_id}_{index}th_{mask_iou:.4f}.txt")
                with open(completion_path, "w") as f:
                    f.write(f'{completion_text[index]}')

            acc_iou += iou
            acc_iou[union_i == 0] += 1.0  # no-object target

        intersection = intersection.cpu().numpy() / len(masks_list)
        union = union.cpu().numpy() / len(masks_list)
        acc_iou = acc_iou.cpu().numpy() / len(masks_list)
        avg_box_iou = sum(box_ious) / len(box_ious)

        return intersection, union, acc_iou, len(masks_list), avg_box_iou


def main(args):
    """Main execution flow"""
    # Determine the datasets to evaluate
    target_datasets = [
        "refcoco.unc.testA", "refcoco.unc.testB", "refcoco.unc.val",
        'refcoco+.unc.val', 'refcoco+.unc.testA', 'refcoco+.unc.testB',
        'refcocog.umd.val', 'refcocog.umd.test'
    ]
    os.makedirs(f"{args.model_path}/evaluations", exist_ok=True)

    dist.init_process_group('nccl', init_method="env://")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    evaluator = RefCOCOEvaluator(args)

    for ds in target_datasets:
        print(f"Processing {ds}...")
        dataset = ReferSegEvaluationSet(ds, coord_norm_type=args.coord_norm_type, if_cot=args.cot, base_image_dir=args.image_dir)
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=rank)
        dataloader = DataLoader(dataset, 1, False, sampler=sampler,
                                collate_fn=lambda batch: list(batch), num_workers=16)

        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
        box_iou_meter = AverageMeter("boxIoU", ":6.3f", Summary.SUM)

        for batch_data in tqdm(dataloader):
            assert len(batch_data) == 1, "Only batch_size=1 evaluation is supported"
            intersection, union, acc_iou, num_mask, box_iou = evaluator.evaluate_single(batch_data[0])
            intersection_meter.update(intersection, n=num_mask)
            union_meter.update(union, n=num_mask)
            acc_iou_meter.update(acc_iou, n=num_mask)
            box_iou_meter.update(box_iou, n=num_mask)

        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()
        box_iou_meter.all_reduce()
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-8)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]
        box_iou = box_iou_meter.avg

        print("model path is:", args.model_path)
        print(f"evaluation on {ds}: giou={giou}, ciou={ciou}, box_iou={box_iou}")

        with open(f"{args.model_path}/evaluations/{ds}.txt", "w") as f:
            f.write(f"model path is: {args.model_path},\n evaluation on {ds}: giou={giou}, ciou={ciou}, box_iou={box_iou}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of samples (for debugging)")
    parser.add_argument("--coord_norm_type", type=str, default="qwen2p5vl", choices=["qwen2vl", "qwen2p5vl"], help="Coordinate normalization type")
    parser.add_argument("--image_dir", type=str, default="./datasets", help="Path to refer_seg dir")
    parser.add_argument("--vis", action="store_true", help="Whether to visualize the results")
    parser.add_argument("--cot", action="store_true", help="Whether to use COT")
    args = parser.parse_args()
    main(args)
