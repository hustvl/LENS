import json
import re
import os


def get_qa_pairs(conversation: list):
    qa_pairs = []
    for i in range(0, len(conversation), 2):
        question = conversation[i]['value']
        answer = conversation[i+1]['value']
        qa_pairs.append((question, answer))
    return qa_pairs

def load_image(image_path):
    from PIL import Image
    import megfile
    from io import BytesIO
    import os
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    if 's3://' in image_path:
        with megfile.smart_open(image_path, "rb") as f:
            bytes_data = f.read()
        image = Image.open(BytesIO(bytes_data), "r").convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image

# make conversation for text hf-dataset
def make_conversation(
        example,
        system_prompt,
        question_template=None,
        answer_template=None
    ):
    question = example["problem"] if question_template is None else question_template.format(question=example["problem"])
    answer = example["solution"] if answer_template is None else answer_template.format(answer=example["solution"])
    return {
        "prompt": json.dumps([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]),
    }

# make conversation for multi-modal hf-dataset
def make_conversation_image(
        example,
        system_prompt,
        question_template=None,
        answer_template=None
    ):
    question = example["problem"] if question_template is None else question_template.format(question=example["problem"])
    question = question + "." if not question.endswith(".") else question
    return {
        "prompt": json.dumps([
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]),
    }

# make conversation for json dataset
def json_map(
        example,
        system_prompt,
        question_template=None,
        answer_template=None
    ):
    '''
    1. example:
        {
            "problem": <str>,
            "image": <image_path>,
            "solution": <int/str>
        }
    2. system_prompt: <str>
    3. question_template: <str>, e.g. "xx {question}?"
    4. answer_template: <str>, e.g. "<answer> {answer} </answer>."
    '''
    image_path = example['image']
    question = example['problem'] if question_template is None else question_template.format(question=example['problem'])
    solution = example['solution'] if answer_template is None else answer_template.format(answer=example['solution'])
    
    rst = {
        "image": load_image(image_path),
        "prompt": json.dumps([
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]),
        "solution": solution,
        "problem": question,
    }
    if 'clicks' in example:
        rst['clicks'] = example['clicks']
    return rst

def parse_float_sequence_within(input_str):
    """
    Extract the first sequence of four floating-point numbers within square brackets from a string.

    Args:
    input_str (str): A string that may contain a sequence of four floats within square brackets.

    Returns:
    list: A list of four floats if the pattern is found, or a list of four zeros if the pattern is not found.
    """
    # Define the regex pattern to find the first instance of four floats within square brackets
    # TODO: add more patterns to support various formats
    # pattern1 [num, num, num, num]
    pattern = r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"

    # Use re.search to find the first match of the pattern in the input string
    match = re.search(pattern, input_str)

    # If a match is found, convert the captured groups into a list of floats
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    # pattern2 (num, num, num, num)
    pattern = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\)"
    match = re.search(pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    # pattern3 (num, num), (num, num)
    pattern = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\),\s*\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)"
    match = re.search(pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    # If the input does not contain the pattern, return the null float sequence
    return [0, 0, 0, 0]

def extract_bbox_answer(content):
    # is_qwen2vl = False
    # if "<|box_start|>" in content:
    #     is_qwen2vl = True
    bbox = parse_float_sequence_within(content)
    # if not is_qwen2vl:
    #     bbox = [int(x * 1000) for x in bbox]
    return bbox

def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou

def save_dict_to_json(dict_data, filename):
    """
    Save a dictionary to a JSON file.
    
    Args:
        dict_data (dict): Dictionary to be saved
        filename (str): Path to the output JSON file
    """
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict_data, indent=4))
    except Exception as e:
        print(f"error saving dictionary to {filename}: {e}")

def save_args_to_txt(args, filename):
    """
    Save the parsed arguments to a txt file.
    
    Args:
        args (argparse.Namespace): The parsed arguments
        filename (str): The path to the output txt file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")