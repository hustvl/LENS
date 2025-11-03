import json
import random
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.open_r1.constants import system_prompt_registry, question_template_registry, answer_template_registry
from qwen_vl_utils import smart_resize
from src.open_r1.refer import REFER
from torchvision.transforms.functional import resize as resize_api
import numpy as np
from pycocotools import mask as maskUtils
import glob
import cv2

local_rank = int(os.environ.get("LOCAL_RANK", -1))

def get_mask_from_json(json_path, img):
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

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence


def resize_longest(image: Image.Image, longest_side_length=640):
    """
    Resize the image so that its longest side is scaled to the specified length,
    while maintaining the aspect ratio.

    :param image: The PIL.Image object to resize.
    :param longest_side_length: The length of the longest side after resizing.
    :return: The resized PIL.Image object.
    """
    # Get the original width and height of the image
    original_width, original_height = image.size

    # Determine which side is the longest
    if original_width > original_height:
        scale_factor = longest_side_length / original_width
    else:
        scale_factor = longest_side_length / original_height

    # Calculate the new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    return resized_image


def resize_shortest(image: Image.Image, shortest_side_length=640):
    """
    Resize the image so that its shortest side is scaled to the specified length,
    while maintaining the aspect ratio.

    :param image: The PIL.Image object to resize.
    :param shortest_side_length: The length of the shortest side after resizing.
    :return: The resized PIL.Image object.
    """
    original_width, original_height = image.size

    # Determine which side is the shortest
    if original_width < original_height:
        scale_factor = shortest_side_length / original_width
    else:
        scale_factor = shortest_side_length / original_height

    # Calculate the new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    return resized_image


def mask_to_bounding_box(mask):
    """
    Given a mask, compute the bounding box of the foreground.

    :param mask: 2D ndarray containing values 0, 1, 255
                 1: indicates foreground
                 0: indicates background
                 255: indicates ignore
    :return: The bounding box coordinates as a JSON-style list.
    """
    # Find the coordinates of the foreground pixels (value == 1)
    foreground_positions = np.argwhere(mask == 1)
    
    # If no foreground exists, return an empty box
    if foreground_positions.size == 0:
        return [0, 0, 0, 0]
    
    # Compute the minimum and maximum boundaries of the bounding box
    ymin, xmin = foreground_positions.min(axis=0)
    ymax, xmax = foreground_positions.max(axis=0)

    # Prepare the bounding box data
    bounding_box = [
        int(xmin),
        int(ymin),
        int(xmax),
        int(ymax),
    ]

    return bounding_box


class ReasonSegDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(
        self,
        script_args,
        base_image_dir = "./datasets",
        solution_format="str",
    ):
        self.script_args = script_args
        self.solution_format = solution_format
        self.images = glob.glob(os.path.join(base_image_dir, "reason_seg", "ReasonSeg", "train", "*.jpg"))
        self.images = sorted(self.images)

        self.question_template = question_template_registry[script_args.question_template]
        self.system_template = system_prompt_registry["default"]
        self.coord_norm_type = script_args.coord_norm_type
        print(f'coord_norm_type = {self.coord_norm_type}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        json_path = image_path.replace(".jpg", ".json")
        

        image = Image.open(image_path).convert(mode="RGB")
        origin_width, origin_height = image.size
        image = resize_longest(image, longest_side_length=640)
        width, height = image.size
        min_pixels = self.script_args.min_pixels
        max_pixels = self.script_args.max_pixels
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=28,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        llm_image = image.resize((resized_width, resized_height))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sam_image = cv2.resize(image, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(sam_image).permute(2, 0, 1).contiguous()
        sam_image = (sam_image-self.pixel_mean) / self.pixel_std
        
        mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
        box = mask_to_bounding_box(mask_json)
        xmin, ymin, xmax, ymax = box
        if self.coord_norm_type=="qwen2vl":
            solution = [
                round(xmin / origin_width * 1000),
                round(ymin / origin_height * 1000),
                round(xmax / origin_width * 1000),
                round(ymax / origin_height * 1000)
            ]
        elif self.coord_norm_type=="qwen2p5vl":
            solution = [
                round(xmin / origin_width * resized_width),
                round(ymin / origin_height * resized_height),
                round(xmax / origin_width * resized_width),
                round(ymax / origin_height * resized_height),
            ]
        else:
            raise NotImplementedError("Unknown coord_norm_type")
        if self.solution_format == "str":
            solution = str(solution)
        elif self.solution_format == "list":
            solution = solution
        else:
            raise ValueError(f"Unknown solution format: {self.solution_format}")


        problem = random.choice(sampled_sents)
        prompt = json.dumps(
            [
                {"role": "system", "content": self.system_template},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.question_template.format(question=problem)}
                ]},
            ]
        )

        masks = [mask_json]
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        masks = (masks==1).float()

        return {
            "image": llm_image,
            "problem": problem,
            "solution": solution,
            "prompt": prompt,
            "sam_image": sam_image,
            "mask": masks[0],
            "image_path": image_path
        }


class ReferSegCLDataset(torch.utils.data.Dataset):
    '''
    Each item corresponds to a referring mask.
    During __getitem__, one referring sentence is randomly selected from multiple candidates.
    Implements a Curriculum Learning (simple → complex) loading strategy.
    '''
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(
        self,
        script_args,
        base_image_dir = "./datasets",
        solution_format="str",
    ):
        super().__init__()
        self.script_args = script_args
        self.base_image_dir = base_image_dir  
        self.system_prompt_template = system_prompt_registry[script_args.system_prompt_template]
        self.question_template = question_template_registry[script_args.question_template]
        self.answer_template = answer_template_registry[script_args.answer_template]
        self.refer_seg_ds_list = ["refcoco", "refcoco+", "refcocog"]
        self.list_data_dict = []
        self.refer_annotations = {}
        self.total_images = 0

        # format solution
        self.solution_format = solution_format

        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            refer_api = REFER(os.path.join(base_image_dir, "refer_seg"), ds, splitBy)
            curriculum = json.load(open(os.path.join(base_image_dir, "refer_seg", ds, "curriculum.json")))
            ref_ids_train = [x["ref_id"] for x in curriculum if 0.2<x["total_reward"]<1.8]
            ref_rewards = [x["total_reward"] for x in curriculum if 0.2<x["total_reward"]<1.8]

            # ref_ids_train = refer_api.getRefIds(split="train")
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
            for i in range(len(refs_train)):
                refs_train[i]["ref_reward"] = ref_rewards[i]
            
            self.list_data_dict.extend(refs_train)

            self.refer_annotations.update(refer_api.Anns)
        self.list_data_dict = sorted(self.list_data_dict, key=lambda x: x["ref_reward"], reverse=True)      

        if self.script_args.train_sample_size is not None and self.script_args.train_sample_size < len(self.list_data_dict):
            self.list_data_dict = random.sample(self.list_data_dict, self.script_args.train_sample_size)
            print(f"Loaded {len(self.list_data_dict)} samples from {self.base_image_dir}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        ref = self.list_data_dict[idx]
        if "file_name" in ref:
            image_path = f"COCO_train2014_{ref['image_id']:012}.jpg"
            image_path = os.path.join(self.base_image_dir, "refer_seg/images/mscoco/images/train2014", image_path)
        else:
            # refclef
            image_id = ref["image_id"]
            image_path = os.path.join(self.base_image_dir, "refer_seg/images/saiapr_tc-12/{:02d}/images/{}.jpg".format(image_id//1000, image_id))

        image = Image.open(image_path).convert(mode="RGB")
        width, height = image.size
        min_pixels = self.script_args.min_pixels
        max_pixels = self.script_args.max_pixels
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=28,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        llm_image = image.resize((resized_width, resized_height))

        sam_image = resize_api(image, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(np.array(sam_image)).permute(2, 0, 1).contiguous()
        sam_image = (sam_image-self.pixel_mean) / self.pixel_std
        
        problem = random.choice(ref["sentences"])["sent"]
        prompt = json.dumps(
            [
                {"role": "system", "content": self.system_prompt_template},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.question_template.format(question=problem)},
                ]},
            ],
        )
        ann = self.refer_annotations[ref["ann_id"]]
        xmin, ymin, box_w, box_h = ann["bbox"]
        xmax = xmin + box_w
        ymax = ymin + box_h
        box = [
            round(xmin / width * 1000),
            round(ymin / height * 1000),
            round(xmax / width * 1000),
            round(ymax / height * 1000)
        ]

        if self.solution_format == "str":
            box = str(box)
        elif self.solution_format == "list":
            box = box
        else:
            raise ValueError(f"Unknown solution format: {self.solution_format}")

        if len(ann["segmentation"]) == 0:
            mask = np.zeros((height, width)).astype(
                np.uint8
            )

        if type(ann["segmentation"][0]) == list:  # polygon
            rle = maskUtils.frPyObjects(
                ann["segmentation"], height, width
            )
        else:
            rle = ann["segmentation"]
            for i in range(len(rle)):
                if not isinstance(rle[i]["counts"], bytes):
                    rle[i]["counts"] = rle[i]["counts"].encode()
        mask = maskUtils.decode(rle)
        mask = np.sum(
            mask, axis=2
        )  # sometimes there are multiple binary map (corresponding to multiple segs)
        mask = mask>0
        mask = mask.astype(np.uint8)  # convert to np.uint8
        mask = torch.from_numpy(mask)
        return {
            "image": llm_image,
            "problem": problem,
            "solution": box,
            "prompt": prompt,
            "sam_image": sam_image,
            "mask": mask,
            "image_path": image_path
        }


class ReferSegDataset(torch.utils.data.Dataset):
    '''
    Each item corresponds to a referring mask.
    During __getitem__, one referring sentence is randomly selected from multiple candidates.
    '''
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(
        self,
        script_args,
        base_image_dir = "./datasets",
        solution_format="str",
    ):
        super().__init__()
        self.script_args = script_args
        self.base_image_dir = base_image_dir  
        self.system_prompt_template = system_prompt_registry[script_args.system_prompt_template]
        self.question_template = question_template_registry[script_args.question_template]
        self.answer_template = answer_template_registry[script_args.answer_template]
        self.refer_seg_ds_list = script_args.refer_seg_ds.split(",")
        self.coord_norm_type = script_args.coord_norm_type
        self.list_data_dict = []
        self.refer_annotations = {}
        self.total_images = 0

        # format solution
        self.solution_format = solution_format

        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            refer_api = REFER(os.path.join(base_image_dir, "refer_seg"), ds, splitBy)

            ref_ids_train = refer_api.getRefIds(split="train")
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
            self.list_data_dict.extend(refs_train)

            self.refer_annotations.update(refer_api.Anns)

        if self.script_args.train_sample_size is not None and self.script_args.train_sample_size < len(self.list_data_dict):
            self.list_data_dict = random.sample(self.list_data_dict, self.script_args.train_sample_size)
            print(f"Loaded {len(self.list_data_dict)} samples from {self.base_image_dir}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        ref = self.list_data_dict[idx]
        if "file_name" in ref:
            image_path = f"COCO_train2014_{ref['image_id']:012}.jpg"
            image_path = os.path.join(self.base_image_dir, "refer_seg/images/mscoco/images/train2014", image_path)
        else:
            # refclef
            image_id = ref["image_id"]
            image_path = os.path.join(self.base_image_dir, "refer_seg/images/saiapr_tc-12/{:02d}/images/{}.jpg".format(image_id//1000, image_id))

        image = Image.open(image_path).convert(mode="RGB")
        width, height = image.size
        min_pixels = self.script_args.min_pixels
        max_pixels = self.script_args.max_pixels
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=28,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        llm_image = image.resize((resized_width, resized_height))

        sam_image = resize_api(image, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(np.array(sam_image)).permute(2, 0, 1).contiguous()
        sam_image = (sam_image-self.pixel_mean) / self.pixel_std
        
        problem = random.choice(ref["sentences"])["sent"]
        prompt = json.dumps(
            [
                {"role": "system", "content": self.system_prompt_template},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.question_template.format(question=problem)},
                ]},
            ],
        )
        ann = self.refer_annotations[ref["ann_id"]]
        xmin, ymin, box_w, box_h = ann["bbox"]
        xmax = xmin + box_w
        ymax = ymin + box_h
        if self.coord_norm_type=="qwen2vl":
            box = [
                round(xmin / width * 1000),
                round(ymin / height * 1000),
                round(xmax / width * 1000),
                round(ymax / height * 1000)
            ]
        elif self.coord_norm_type=="qwen2p5vl":
            box = [
                round(xmin / width * resized_width),
                round(ymin / height * resized_height),
                round(xmax / width * resized_width),
                round(ymax / height * resized_height),
            ]
        else:
            raise NotImplementedError

        if self.solution_format == "str":
            box = str(box)
        elif self.solution_format == "list":
            box = box
        else:
            raise ValueError(f"Unknown solution format: {self.solution_format}")

        if len(ann["segmentation"]) == 0:
            # import pdb;pdb.set_trace()
            mask = np.zeros((height, width)).astype(
                np.uint8
            )

        if type(ann["segmentation"][0]) == list:  # polygon
            rle = maskUtils.frPyObjects(
                ann["segmentation"], height, width
            )
        else:
            rle = ann["segmentation"]
            for i in range(len(rle)):
                if not isinstance(rle[i]["counts"], bytes):
                    rle[i]["counts"] = rle[i]["counts"].encode()
        mask = maskUtils.decode(rle)
        mask = np.sum(
            mask, axis=2
        )  # sometimes there are multiple binary map (corresponding to multiple segs)
        mask = mask>0
        mask = mask.astype(np.uint8)  # convert to np.uint8
        mask = torch.from_numpy(mask)
        return {
            "image": llm_image,
            "problem": problem,
            "solution": box,
            "prompt": prompt,
            "sam_image": sam_image,
            "mask": mask,
            "image_path": image_path,
            "ref_meta": ref
        }


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_a, dataset_b, prob_a=0.5):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.prob_a = prob_a
        self.len_a = len(dataset_a)
        self.len_b = len(dataset_b)
        self.total_len = self.len_a + self.len_b

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if random.random() < self.prob_a:
            return self.dataset_a[random.randint(0, self.len_a - 1)]
        else:
            return self.dataset_b[random.randint(0, self.len_b - 1)]


class GroundingSuiteDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(
        self,
        script_args,
        anno_path="./datasets/GroundingSuite-Eval.jsonl",
        image_dir="./datasets/coco/unlabeled2017",
    ):
        self.script_args = script_args
        self.coord_norm_type = script_args.coord_norm_type
        self.image_dir = image_dir
        self.list_data_dict = []

        self.system_prompt_template = system_prompt_registry[script_args.system_prompt_template]
        self.question_template = question_template_registry[script_args.question_template]

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

        segmentation = ref["segmentation"]
        if isinstance(segmentation, list):
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
        problem = "\"" + caption + "\""
        prompt = json.dumps(
            [
                {"role": "system", "content": self.system_prompt_template},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.question_template.format(question=problem)},
                ]},
            ],
        )

        return {
            "image_path": image_path,
            "image": llm_image,
            "prompt": prompt,
            "sam_image": sam_image,
            "mask": mask,
            "problem": problem,
            "solution": box
        }

class GSTrainDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(
        self,
        script_args,
        anno_path="./datasets/output_jsonl",
        image_dir="./datasets/SAM_region_caption/origin",
    ):
        self.script_args = script_args
        self.coord_norm_type = script_args.coord_norm_type
        self.image_dir = image_dir
        self.list_data_dict = []

        self.system_prompt_template = system_prompt_registry[script_args.system_prompt_template]
        self.question_template = question_template_registry[script_args.question_template]
        
        jsonl_files = glob.glob(os.path.join(anno_path, "*.jsonl"))        
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r") as f:
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

        segmentation = ref["segmentation"]
        if isinstance(segmentation, list):
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
        problem = "\"" + caption + "\""
        prompt = json.dumps(
            [
                {"role": "system", "content": self.system_prompt_template},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.question_template.format(question=problem)},
                ]},
            ],
        )

        return {
            "image_path": image_path,
            "image": llm_image,
            "prompt": prompt,
            "sam_image": sam_image,
            "mask": mask,
            "problem": problem,
            "solution": box
        }
