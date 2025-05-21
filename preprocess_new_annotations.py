from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import florence.modeling_florence2 as flor2
from florence.configuration_florence2 import *
from florence.florence_attn import *
from florence.utils import unnormalize
import argparse
from tqdm import tqdm 

def run_task(flo_model, processor, image, task, device, text_prompt=''):
    prompt = f'<{task}>{text_prompt}'

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device)
    generated_ids = flo_model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)

    # print(generated_text)

    parsed_answer = processor.post_process_generation(
        generated_text[0],
        task=prompt,
        image_size=(image.size[0], image.size[1])
    )
    return parsed_answer, prompt

def convert_str_to_bb(phrase):
    word = ''
    loc = ''
    word_dict = {}
    find_bb = False

    for char in phrase:
        if not find_bb:
            if char == '<':
                if word not in word_dict:
                    word_dict[word] = []
                loc = ''
                find_bb = True
            else:
                if word in word_dict:
                    word = ''
                word += char
        else:
            if char == '>':
                word_dict[word].append(int(loc.replace('loc_', '')))
                find_bb = False
            else:
                loc += char
    return word_dict

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: A list or tuple of four numbers representing the coordinates of the first bounding box in the format (x1, y1, x2, y2).
        box2: A list or tuple of four numbers representing the coordinates of the second bounding box in the format (x1, y1, x2, y2).

    Returns:
        The IoU of the two bounding boxes, a float between 0 and 1.
    """

    x1_intersect = max(box1[0], box2[0])
    y1_intersect = max(box1[1], box2[1])
    x2_intersect = min(box1[2], box2[2])
    y2_intersect = min(box1[3], box2[3])

    intersection_area = max(0, x2_intersect - x1_intersect) * max(0, y2_intersect - y1_intersect)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def get_best_iou(teacher_bb, bb_dict):
    iou = 0
    best = ''
    for key, bb in bb_dict.items():
        curr_iou = calculate_iou(bb, teacher_bb)
        if curr_iou > iou:
            iou = curr_iou
            best = key 

    return best, iou

def test(image, teacher_bb, flo_model, processor, device):

    task = 'CAPTION_TO_PHRASE_GROUNDING'


    grounding, gnd_key = run_task(flo_model, processor, image, task, device, 'a teacher teaching in front of students') #caption[cap_key])
    phrase = grounding[gnd_key]

    word_dict = convert_str_to_bb(phrase)
    bb_dict = {}
    for word, bb in word_dict.items():
        bb_dict[word] = unnormalize(bb, image.width, image.height)

    best, iou = get_best_iou(teacher_bb, bb_dict)

    if best == '':
        return teacher_bb, '', 0
    return bb_dict[best], best, iou

def get_caption(image, flo_model, processor, device, level=0):
    if level == 0:
        task = 'CAPTION'
    elif level == 1:
        task = 'DETAILED_CAPTION'
    elif level == 2:
        task = 'MORE_DETAILED_CAPTION'

    caption, cap_key = run_task(flo_model, processor, image, task, device)
    return caption[cap_key]

def main(class_name, vid_num, device):
    flo_model, processor = flor2.load("BASE_FT", device)

    base_dir = os.path.join('./', 'frames')
    class_dir = os.path.join(base_dir, class_name)

    sorted_list = sorted(os.listdir(class_dir))
    if vid_num > len(sorted_list):
        return 0
    if vid_num > 0:
        sorted_list = [f'{vid_num:04d}']

    for vid in sorted_list:
        vid_path = os.path.join(class_dir, vid)
        with open(os.path.join(vid_path, 'annotation.json'), 'r') as f:
            json_dict = json.load(f)
        new_json_dict = copy.deepcopy(json_dict)
        sorted_frame_list = sorted(json_dict["frames"].keys())
        for frame in tqdm(sorted_frame_list, total=len(sorted_frame_list)):
            teacher_bb = json_dict['frames'][frame]["teacher_box"]
            image = Image.open(os.path.join(vid_path, f'{int(frame):04d}.jpg'))
            new_bb, best_key, iou_score = test(image, teacher_bb, flo_model, processor, device)
            new_json_dict['frames'][frame]["teacher_box"] = new_bb
            new_json_dict['frames'][frame]["caption"] = {}
            new_json_dict['frames'][frame]["caption"][0] = get_caption(image, flo_model, processor, device, level=0)
            new_json_dict['frames'][frame]["caption"][1] = get_caption(image, flo_model, processor, device, level=1)
            new_json_dict['frames'][frame]["caption"][2] = get_caption(image, flo_model, processor, device, level=2)

        new_ann_path = os.path.join(vid_path, 'new_annotation.json')
        with open(new_ann_path, 'w') as f:
            json.dump(new_json_dict, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('class_name')
    parser.add_argument('--vid_num', default=-1, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    args = parser.parse_args()
    main(args.class_name, args.vid_num, args.device)