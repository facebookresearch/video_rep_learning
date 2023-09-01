# coding=utf-8
import os
import json
import math
import shutil
from sys import version
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import torch

# NEW - ONLY_CHECK mode: check for all files without processing yet
ONLY_CHECK = False

def main(split="train", classes="gym99", version="v1.0"):
    # data_root = "/home/username/datasets/finegym"
    data_root = "/fsx/mwalmer/carl_workdir/finegym"
    output_dir = os.path.join(data_root, "processed_videos")
    os.makedirs(output_dir, exist_ok=True)
    if split == "train":
        save_file = os.path.join(data_root, f"{classes}_train_{version}.pkl")
    else:
        save_file = os.path.join(data_root, f"{classes}_val.pkl")

    annotation_file = os.path.join(data_root, f"finegym_annotation_info_{version}.json")
    train_file = os.path.join(data_root, f"{classes}_train_element_{version}.txt")
    val_file = os.path.join(data_root, f"{classes}_val_element.txt")
    video_dir = os.path.join(data_root, "FineGym_Raw_database")

    with open(annotation_file, 'r') as f:
        data=json.load(f)
    with open(train_file, 'r') as f:
        train_lines = f.readlines()
    with open(val_file, 'r') as f:
        val_lines = f.readlines()
    if split == "train":
        lines = train_lines
    else:
        lines = val_lines
    labels = {}
    video_ids = set()
    event_ids = set()
    for line in lines:
        full_id = line.split(" ")[0]
        label = int(line.split(" ")[1])
        labels[full_id] = label
        video_id = full_id.split("_E_")[0]
        video_ids.add(video_id)
        event_id = full_id.split("_A_")[0]
        event_ids.add(event_id)

    # NEW: convert event_ids to a list and sort
    event_ids = list(event_ids)
    event_ids = sorted(event_ids)
    print('loaded %i event_ids to process'%len(event_ids))

    warnings = []

    dataset = []
    for i, event_id in tqdm(enumerate(event_ids), total=len(event_ids)):
        
        output_file = os.path.join(output_dir, event_id) + ".mp4"
        video_id = event_id.split("_E_")[0]
        video_data = data[video_id]
        event_data = video_data[event_id.split(video_id+"_")[-1]]
        start_time = event_data['timestamps'][0][0]
        end_time = event_data['timestamps'][0][1]
        
        if not os.path.exists(output_file):
            video_path = os.path.join(video_dir, video_id)
            suffix = [".mp4", ".mkv", ".webm"]
            found = False
            for s in suffix:
                if os.path.exists(video_path+s):
                    video_file = video_path+s
                    found = True
                    break

            # NEW check for missing files
            if not found:
                w = 'WARNING: MISSING FILE: %s'%video_path
                print(w)
                warnings.append(w)

            if ONLY_CHECK: continue

            # temp_output_file = os.path.join(output_dir, event_id) + "_temp.mp4"
            # cmd = f'ffmpeg -hide_banner -loglevel panic -y -ss {start_time}s -to {end_time}s -i {video_file} -c:v copy -c:a copy {output_file}'
            # os.system(cmd)
            # cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {output_file} -strict -2 -vf scale=640:360 {temp_output_file}'
            # os.system(cmd)
            # cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {temp_output_file} -filter:v fps=25 {output_file}'
            # os.system(cmd)
            # os.remove(temp_output_file)

            # NEW - use two temp files to avoid issues
            temp_output_file_1 = os.path.join(output_dir, event_id) + "_temp1.mp4"
            temp_output_file_2 = os.path.join(output_dir, event_id) + "_temp2.mp4"
            # cmd = f'ffmpeg -hide_banner -loglevel panic -y -ss {start_time}s -to {end_time}s -i {video_file} -c:v copy -c:a copy {temp_output_file_1}'
            # cmd = f'ffmpeg -y -ss {start_time}s -to {end_time}s -i {video_file} -c:v copy -c:a copy {temp_output_file_1}'
            
            # NEW: NEED -strict -2
            cmd = f'ffmpeg -hide_banner -loglevel panic -y -ss {start_time}s -to {end_time}s -i {video_file} -strict -2 -c:v copy -c:a copy {temp_output_file_1}'
            # cmd = f'ffmpeg -y -ss {start_time}s -to {end_time}s -i {video_file} -strict -2 -c:v copy -c:a copy {temp_output_file_1}'

            os.system(cmd)
            cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {temp_output_file_1} -strict -2 -vf scale=640:360 {temp_output_file_2}'
            os.system(cmd)
            cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {temp_output_file_2} -filter:v fps=25 {output_file}'
            os.system(cmd)

            # NEW - handle errors
            if not os.path.isfile(temp_output_file_1):
                w = 'WARNING: MISSING TEMP FILE: %s'%temp_output_file_1
                print(w)
                warnings.append(w)
            else:
                os.remove(temp_output_file_1)
            if not os.path.isfile(temp_output_file_2):
                w = 'WARNING: MISSING TEMP FILE: %s'%temp_output_file_2
                print(w)
                warnings.append(w)
            else:
                os.remove(temp_output_file_2)

        video = cv2.VideoCapture(output_file)
        fps =int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(video_file, "\n", output_file)
        print(event_id, start_time, end_time, num_frames, fps, width, height)
        if num_frames == 0:
            print('WARNING: EMPTY PROCESSED VIDEO: ' + output_file)
            print(w)
            warnings.append(w)

        frame_label = -1 * torch.ones(num_frames)
        for action_id in event_data['segments']:
            full_id = event_id + '_' + action_id
            if full_id in labels:
                label = labels[full_id]
                start = event_data['segments'][action_id]['timestamps'][-1][0]
                end = event_data['segments'][action_id]['timestamps'][-1][1]
                frame_label[int(start*fps):int(end*fps)+1] = label
        data_dict = {"id": i, "name": event_id, "video_file": os.path.join("processed_videos", event_id+".mp4"), \
            "seq_len": num_frames, "frame_label": frame_label, "event_label": event_data['event']}
        dataset.append(data_dict)

    if ONLY_CHECK: return warnings
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"{len(dataset)} {split} samples of Finegym dataset have been writen.")
    return warnings




if __name__ == '__main__':
    if ONLY_CHECK:
        print('ONLY_CHECK MODE: checking for missing files')
    
    ALL_WARNINGS = []
    # ALL_WARNINGS += main(split="train", classes="gym99", version="v1.0")
    ALL_WARNINGS += main(split="val", classes="gym99", version="v1.0")
    # ALL_WARNINGS += main(split="train", classes="gym288", version="v1.0")
    # ALL_WARNINGS += main(split="val", classes="gym288", version="v1.0")

    print('TOTAL WARNING COUNT: %i'%len(ALL_WARNINGS))
    for W in ALL_WARNINGS:
        print(W)

    # main(split="train", classes="gym99", version="v1.1")
    # main(split="train", classes="gym288", version="v1.1")