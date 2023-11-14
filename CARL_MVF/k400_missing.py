# check for missing files in K400
import os
import csv
import argparse

from tqdm import tqdm
from torchvision.io import read_video
import torch

from utils.parser import parse_args, load_config, setup_train_dir
from datasets import construct_dataloader


WORKDIR = "/datasets01/"

train_dataset = os.path.join(WORKDIR, "kinetics_400/k400/annotations/train.csv")

video_dir = os.path.join(WORKDIR, "kinetics_400/k400/train")
# video_dir = os.path.join(WORKDIR, "kinetics_400/k400/videos/train")
# rep_dir = os.path.join(WORKDIR, "kinetics_400/k400/replacement/replacement_for_corrupted_k400")



# read annotation file and check for missing files
def check_annotation_file():
    in_train = 0
    in_replace = 0
    missing_files = []
    with open(train_dataset, 'r') as f:
        reader = csv.reader(f)
        dataset = []
        for r, row in enumerate(reader):
            if r == 0: continue
            if r%1000==0: print(r)
            label = row[0]
            video_file = "%s_%06i_%06i.mp4"%(row[1], int(row[2]), int(row[3]))
            
            # old locations:
            video_path = os.path.join(video_dir, video_file)
            # rep_path = os.path.join(rep_dir, video_file)
            # new locations:
            # video_path = os.path.join(video_dir, label, video_file)

            if os.path.isfile(video_path):
                in_train += 1
            # elif os.path.isfile(rep_path):
            #     in_replace += 1
            else:
                missing_files.append(video_file)
            # debug
            # if r > 2000: break

    print('Missing:')
    # print(missing_files)
    print(len(missing_files))
    print('In Train:')
    print(in_train)
    print('In Replacement:')
    print(in_replace)

    with open('k400_missing.txt', 'w') as f:
        for mf in missing_files:
            f.write(mf+'\n')


# load video files and look for corrupted files
def test_video_loading(start_fn=0, fn_cap=0):
    files = os.listdir(video_dir)
    # for fn, f in enumerate(tqdm(files)):
    files = sorted(files)
    cor_fn = 'k400_corrupted_%i.txt'%start_fn
    print('found %i video files'%len(files))
    if start_fn > 0:
        print('starting at file %i'%start_fn)
        files = files[start_fn:]
    if fn_cap > 0:
        print('limiting to %i files'%fn_cap)
        files = files[:fn_cap]
    # for f in tqdm(files):
    for fn, f in enumerate(files):
        if fn%100==0: print(fn)
        video_file = os.path.join(video_dir, f)
        video, _, info = read_video(video_file, pts_unit='sec')
        seq_len = len(video)
        if seq_len == 0:
            # empty video = corrupted download
            # cor_files.append(f)
            print(f)
            with open(cor_fn, 'a') as cf:
                cf.write(f+'\n')
        # fn += 1
        # debug
        # if fn > 100: break
    
    # print('corrupted files:')
    # print(len(cor_files))
    # with open('k400_corrupted.txt', 'w') as f:
    #     for cf in cor_files:
    #         f.write(cf+'\n')


# wrapper for the loader
def test_loader():
    # arg and config handling
    args = parse_args()
    cfg = load_config(args)
    setup_train_dir(cfg, cfg.LOGDIR, args.continue_train)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.NUM_GPUS = torch.cuda.device_count() # num_gpus_per_machine
    # args.world_size = int(os.getenv('WORLD_SIZE')) # total_gpus
    if os.environ.get('OMPI_COMM_WORLD_SIZE') is None:
        args.rank = args.local_rank
    else:
        args.node_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        args.rank = args.node_rank * torch.cuda.device_count() + args.local_rank
    # logger.info(f'Node info: rank {args.rank} of world size {args.world_size}')
    cfg.args = args

    # prep data loader
    train_loader, _ = construct_dataloader(cfg, "train", no_eval=True)
    train_loader = tqdm(train_loader)

    # run through data
    for cur_iter, (videos, _labels, seq_lens, chosen_steps, video_masks, names) in enumerate(train_loader):
        # debug
        if cur_iter > 10: break



def main():
    parser = argparse.ArgumentParser('check for corrupted files')
    parser.add_argument('--start', type=int, default=0, help='start file number (default: 0)')
    parser.add_argument('--cap', type=int, default=0, help='limit to processing this many files (default: no limit)')
    args = parser.parse_args()

    test_video_loading(args.start, args.cap)
    # check_annotation_file()
    # test_loader()


if __name__ == "__main__":
    main()
    