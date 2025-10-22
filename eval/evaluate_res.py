from transformers import AutoProcessor
from src.open_r1.trainer.samr1_wo_sam import SAMR1ForConditionalGeneration
from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration_qwen2p5
import torch
from tqdm import tqdm
import os
import argparse
from src.segment_anything_2.sam2.build_sam import build_sam2
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pycocotools import mask as maskUtils
from src.open_r1.refer import REFER
import numpy as np
from qwen_vl_utils import smart_resize
from PIL import Image
from torchvision.transforms.functional import resize as resize_api
from eval.utils import AverageMeter, Summary, intersectionAndUnionGPU
from src.open_r1.constants import system_prompt_registry, question_template_registry

"""
This script evaluates the performance of a Stage 1 trained model for RES tasks.
If you need to evaluate a Stage 2 model, please use evaluate_res2.py.

Stage 1 training only updates the learnable query and connector structures; 
LLM and SAM parameters are not trained.  
The Stage 1 model does not contain SAM parameters, so this script instantiates SAM 
in the evaluator and performs SAM inference inside evaluate_single().

In Stage 1, the model can access grounding ground-truth:
- If --if_stage1 is True: add GT grounding to messages, then perform segmentation.
- If --if_stage1 is False: LLM performs grounding first, then segmentation based on the generated result.

--coord_norm_type controls coordinate normalization for LLM grounding:
- "qwen2vl": normalizes model output coordinates to [0, 1000]
- "qwen2.5vl": uses absolute coordinates

Example:
Original image size: 640×480  
GT coordinate: (320, 240)  
After resizing to the nearest multiple of 28 (644×476):
- qwen2vl → (500, 500)
- qwen2.5vl → (322, 238)
"""

class ReferSegEvaluationSet(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(self, val_dataset, base_image_dir="./dataset", coord_norm_type="qwen2vl"):
        splits = val_dataset.split(".")
        assert len(splits) == 3
        ds, splitBy, split = splits
        assert ds in ["refcoco", "refcoco+", "refcocog"]
        assert splitBy in ["unc", "umd"]
        assert split in ["val", "test", "testA", "testB"]

        self.coord_norm_type = coord_norm_type
        print(f'coord_norm_type = {self.coord_norm_type}')
        self.list_data_dict = []
        self.refer_annotations = {}
        self.base_image_dir = base_image_dir
        base_image_dir = os.path.join(base_image_dir, "refer_seg")

        refer_api = REFER(base_image_dir, ds, splitBy)
        ref_ids = refer_api.getRefIds(split=split)
        refs = refer_api.loadRefs(ref_ids=ref_ids)
        self.list_data_dict.extend(refs)
        self.refer_annotations.update(refer_api.Anns)
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

        messages, problems = [], []
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
        xmax, ymax = xmin + box_w, ymin + box_h

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
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            if isinstance(ann["segmentation"][0], list):  # polygon
                rle = maskUtils.frPyObjects(ann["segmentation"], height, width)
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()

            mask = maskUtils.decode(rle)
            mask = np.sum(mask, axis=2)  # merge multiple binary masks
            mask = mask.astype(np.uint8)

        mask = torch.from_numpy(mask)

        return {
            "image": llm_image,
            "messages": messages,
            "sam_image": sam_image,
            "mask": mask,
            "problems": problems,
            "box": box,
            "image_path": image_path,
        }
    

class RefCOCOEvaluator:
    """Evaluator class encapsulating the evaluation logic"""
    def __init__(self, args):
        self.args = args
        self.dtype = torch.bfloat16
        if args.coord_norm_type=="qwen2p5vl":
            self.model = SAMR1ForConditionalGeneration_qwen2p5.from_pretrained(
                args.model_path, 
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2",
            ).cuda()
        elif args.coord_norm_type=="qwen2vl":
            self.model = SAMR1ForConditionalGeneration.from_pretrained(
                args.model_path, 
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2",
            ).cuda()
        self.sam = build_sam2(
            "sam2_hiera_l.yaml", 
            "./pretrained/sam2_hiera_large.pt", 
            device="cuda"
        ).to(self.dtype)

        self.processor = AutoProcessor.from_pretrained(args.model_path)
        input_size = 1024
        self._bb_feat_sizes = [
            (input_size//4, input_size//4),
            (input_size//8, input_size//8),
            (input_size//16, input_size//16),
        ]
        # self.stage1 = True if "stage2" not in args.model_path else False
        self.stage1 = args.if_stage1
        print("Current stage is stage1? ", self.stage1)

    @torch.no_grad()
    def evaluate_single(self, input_data):
        """
        Evaluate one mask of one image.
        Each mask contain several referrings.
        """
        # Prepare llm input data
        messages = input_data["messages"]
        if self.stage1:
            texts = []
            for idx in range(len(messages)):
                new_massage = messages[idx]
                problem = input_data["problems"][idx]
                if type(input_data["box"]) is str:
                    box = eval(input_data["box"])
                else:
                    box = input_data["box"]
                new_massage.append(
                    {
                        "role": "assistant",
                        "content": f"<|object_ref_start|>{problem}.<|object_ref_end|><|box_start|>({box[0]},{box[1]}),({box[2]},{box[3]})<|box_end|>"
                    }
                )
                prompt_text = self.processor.apply_chat_template(new_massage, tokenize=False, add_generation_prompt=False)
                # remove the last '\n' in prompts_text
                prompt_text = prompt_text[:-1] if prompt_text[-1] == "\n" else prompt_text
                texts.append(prompt_text)
                messages[idx] = new_massage
        else:
            texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

        image_inputs = [input_data["image"]]*len(texts)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device="cuda", dtype=self.dtype)
        if not self.stage1:
            llm_out = self.model.generate(
                max_length=768, 
                do_sample=False, 
                **inputs
            )
            # self.processor.batch_decode(llm_out, skip_special_tokens=False)
            new_attention_mask = torch.ones_like(llm_out, dtype=torch.int64)
            pos = torch.where(llm_out==self.processor.tokenizer.pad_token_id)
            new_attention_mask[pos] = 0
            inputs.update(
                {
                    "input_ids": llm_out,
                    "attention_mask": new_attention_mask             
                }
            )
        
        output = self.model(output_hidden_states=True, use_learnable_query=True, **inputs)
        box_end_embedding = self.model.get_sam_embedding(output.hidden_states[-1])

        sam_images = input_data["sam_image"].unsqueeze(0).to(device=self.sam.device, dtype=self.dtype)
        backbone_out = self.sam.forward_image(sam_images)
        # dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
        _, image_embeddings, _, _ = self.sam._prepare_backbone_features(backbone_out)
        image_embeddings = [_.to(sam_images.dtype) for _ in image_embeddings]
        batch_size = sam_images.shape[0]
        if self.sam.directly_add_no_mem_embed:
            image_embeddings[-1] = image_embeddings[-1] + self.sam.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(image_embeddings[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.sam.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=box_end_embedding,
        )
        sparse_embeddings = sparse_embeddings.to(box_end_embedding.dtype)
        high_res_features = _features["high_res_feats"]

        low_res_masks, iou_predictions, _, _ = self.sam.sam_mask_decoder(
            image_embeddings=_features["image_embed"],
            image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image = True,
            high_res_features=high_res_features,
        )
        # [num_instances, num_multimasks, h, w]
        pred_masks = self.model.postprocess_masks(
            low_res_masks,
            orig_hw=input_data["mask"].shape[-2:],
        )
        pred_masks = (pred_masks[:, 0]>0).int()
    
        masks_list = input_data["mask"].int().unsqueeze(0).repeat(pred_masks.size(0), 1, 1)
        for mask_i, output_i in zip(masks_list, pred_masks):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous().cuda(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection = intersection.cpu().numpy() / len(masks_list)
        union = union.cpu().numpy() / len(masks_list)
        acc_iou = acc_iou.cpu().numpy() / len(masks_list)
        return intersection, union, acc_iou, len(masks_list)


def main(args):
    """Main execution flow"""
    
    # Determine the datasets to evaluate
    target_datasets = [
        'refcoco.unc.val', 'refcoco.unc.testA', 'refcoco.unc.testB',
        'refcoco+.unc.val', 'refcoco+.unc.testA', 'refcoco+.unc.testB',
        'refcocog.umd.val', 'refcocog.umd.test'
    ]

    os.makedirs(f"{args.model_path}/evaluations", exist_ok=True)
    
    # Create evaluator actors
    dist.init_process_group('nccl', init_method="env://")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    evaluator = RefCOCOEvaluator(args)
    
    for ds in target_datasets:
        print(f"Processing {ds}...")
        dataset = ReferSegEvaluationSet(ds, base_image_dir=args.image_dir, coord_norm_type=args.coord_norm_type)
        print(dataset.__len__())
        sampler = DistributedSampler(dataset,shuffle=False, drop_last=False, rank=rank)
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
            # Assign batches to workers in round-robin fashion
            assert len(batch_data)==1, "current code only support batch_size=1 evaluation"
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
        print(f"evaluation on {ds}: giou={giou}, ciou={ciou}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of samples (for debugging)")
    parser.add_argument("--if_stage1", action="store_true", help="Whether to use stage1 model")
    parser.add_argument("--image_dir", type=str, default="./datasets", help="Path to refer_seg dir")
    parser.add_argument("--coord_norm_type", type=str, default="qwen2p5vl", choices=["qwen2vl", "qwen2p5vl"], help="Coordinate normalization type")
    args = parser.parse_args()
    main(args)
    