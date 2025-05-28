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

def main(class_name, vid_num, vid_list, device, threshold, freq):

    flo_model, processor = flor2.load("BASE_FT", device)

    base_dir = os.path.join('./', 'frames')
    class_dir = os.path.join(base_dir, class_name)

    sorted_list = sorted(os.listdir(class_dir))
    if vid_num > len(sorted_list):
        return 0
    if vid_list != None:
        tmp_sorted_list = vid_list.split(',')
        sorted_list = []
        for element in tmp_sorted_list:
            if '-' in element:
                range_start = int(element.split('-')[0])
                range_end = int(element.split('-')[1])+1
                for i in range(range_start, range_end):
                    sorted_list.append(f'{i:04d}')
            else:
                sorted_list.append(f'{int(element):04d}')
    elif vid_num > 0:
        sorted_list = [f'{vid_num:04d}']

    for vid in sorted_list:
        vid_path = os.path.join(class_dir, vid)
        ann_path = os.path.join(vid_path, 'annotation.json')
        if not os.path.exists(ann_path):
            print(f"Error: path {ann_path} does not exist")
            continue
        new_ann_path = os.path.join(vid_path, 'new_annotation.json')
        if os.path.exists(ann_path):
            print(f"Error: path {new_ann_path} exists")
            with open(new_ann_path, 'r') as f:
                json_dict = json.load(f)
            try:
                # cheap way of saying "if it exists in the dict skip"
                t = json_dict["frames"]["1"]["caption"]
                continue
            except:
                pass
        with open(ann_path, 'r') as f:
            json_dict = json.load(f)
        new_json_dict = copy.deepcopy(json_dict)
        sorted_frame_list = sorted(json_dict["frames"].keys())
        skip = False
        if len(sorted_frame_list) > threshold:
            skip = True
        cnt = 0
        last_caption = ''
        last_detailed=''
        last_more = ''
        last_bb = []
        for frame in tqdm(sorted_frame_list, total=len(sorted_frame_list)):
            if skip and cnt % freq != 0 and cnt != 0:
                new_json_dict['frames'][frame]["teacher_box"]  = last_bb
                new_json_dict['frames'][frame]["caption"] = {}
                new_json_dict['frames'][frame]["caption"][0] = last_caption
                new_json_dict['frames'][frame]["caption"][1] = last_detailed
                new_json_dict['frames'][frame]["caption"][2] = last_more
            else:
                teacher_bb = json_dict['frames'][frame]["teacher_box"]
                image = Image.open(os.path.join(vid_path, f'{int(frame):04d}.jpg'))
                new_bb, best_key, iou_score = test(image, teacher_bb, flo_model, processor, device)
                last_bb = new_bb
                new_json_dict['frames'][frame]["teacher_box"]  = last_bb
                new_json_dict['frames'][frame]["caption"] = {}
                last_caption = get_caption(image, flo_model, processor, device, level=0)
                last_detailed = get_caption(image, flo_model, processor, device, level=1)
                last_more = get_caption(image, flo_model, processor, device, level=2)
                new_json_dict['frames'][frame]["caption"][0] = last_caption
                new_json_dict['frames'][frame]["caption"][1] = last_detailed
                new_json_dict['frames'][frame]["caption"][2] = last_more
            cnt += 1

        new_ann_path = os.path.join(vid_path, 'new_annotation.json')
        with open(new_ann_path, 'w') as f:
            json.dump(new_json_dict, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('class_name')
    parser.add_argument('--vid_num', default=-1, type=int)
    parser.add_argument('--vid_list', default=None, type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--threshold', default=50, type=int)
    parser.add_argument('--freq', default=240, type=int)
    args = parser.parse_args()

    if args.vid_num != -1 and args.vid_list != None:
        print('ERROR: vid num and vid list cannot be set at the same time')


    main(args.class_name, args.vid_num, args.vid_list, args.device, args.threshold, args.freq)