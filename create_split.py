import argparse
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import random

def write_file(file_name, frame_dir, pairs, classes, videos):
    with open(file_name, 'w') as f:
        for pair in tqdm(pairs, total=len(pairs)):
            idx, video_dir = pair
            missing_val = 0
            if videos[classes[idx]]["videos"][video_dir]["missing"]:
                missing_val = 1
            video_path = os.path.join(frame_dir, classes[idx], video_dir)
            num_frames = videos[classes[idx]]["videos"][video_dir]["num_frames"]
            if missing_val:
                # pass over any missing values
                # f.write(f'{video_path} {idx} {missing_val}\n')
                continue
            else:
                f.write(f'{video_path} {num_frames} {idx}\n')

def create_new_split(frame_dir='./frames', output_folder='./', seed=100, split_file_prefix='split', fold=1, val=False, percent_real_data=1):

    random.seed(seed)
    classes = sorted(os.listdir(frame_dir))

    videos = {}
    cnt = 0
    labels = []
    missing = {}
    seq = []
    print("Looping files...")
    for i, class_name in enumerate(classes):
        videos[class_name] = {}
        videos[class_name]["idx"] = i
        videos[class_name]["videos"] = {}

        print(f"Looping {class_name}")
        
        total_frame_cnt = 0
        video_dir = os.path.join(frame_dir, class_name)
        video_list = os.listdir(video_dir)
        num_elements = int(percent_real_data*len(video_list))
        reduced_indices = random.sample(list(range(len(video_list))), num_elements)
        for index in tqdm(reduced_indices, total=len(reduced_indices)):
            video = video_list[index]
            frame_files_dir = os.path.join(video_dir, video)
            curr_frame_cnt = len(os.listdir(frame_files_dir))
            total_frame_cnt += curr_frame_cnt
            videos[class_name]["videos"][video] = {}
            videos[class_name]["videos"][video]["missing"] = False
            videos[class_name]["videos"][video]["num_frames"] = curr_frame_cnt - 1
            if curr_frame_cnt == 0:
                videos[class_name]["videos"][video]["missing"] = True
            # Create sequence of videos and labels
            seq.append(video)
            labels.append(i)

        videos[class_name]["frames"] = total_frame_cnt

    print("Creating splits...")
    X_train = np.array(seq)
    y_train = labels
    if val:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=100)
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=100)

    print("Sorting pairs...")
    paired_train = sorted(list(zip(y_train, X_train)))
    paired_test = sorted(list(zip(y_test, X_test)))

    train_file = f'train{split_file_prefix}{fold:02}.txt'
    test_file = f'test{split_file_prefix}{fold:02}.txt'
    if output_folder != '':
        os.makedirs(output_folder, exist_ok=True)
        train_file = os.path.join(output_folder, train_file)
        test_file = os.path.join(output_folder, test_file)

    print("Creating train file...")
    write_file(train_file, frame_dir, paired_train, classes, videos)
    print("Creating test file...")
    write_file(test_file, frame_dir, paired_test, classes, videos)

    if val:
        paired_val = sorted(list(zip(y_val, X_val)))
        val_file = f'val{split_file_prefix}{fold:02}.txt'
        if output_folder != '':
            val_file = os.path.join(output_folder, val_file)
        print("Creating val file...")
        write_file(val_file, frame_dir, paired_val, classes, videos)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frame_dir', default='./frames/')
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('-p', '--percent_real_data', default=1, type=float)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--split_file_prefix', default='split', type=str)
    parser.add_argument('--output_folder', default='', type=str)
    parser.add_argument('--val', action='store_true', default=False)
    args = parser.parse_args()

    create_new_split(args.frame_dir, args.output_folder, args.seed, args.split_file_prefix, args.fold, args.val, args.percent_real_data)