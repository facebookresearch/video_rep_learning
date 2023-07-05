# coding=utf-8
import os
import sys
import pprint
import torch
import random
from tqdm import tqdm
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils.distributed as du
import utils.logging as logging
from utils.parser import parse_args, load_config, setup_train_dir
from models import build_model, save_checkpoint, load_checkpoint
from utils.optimizer import construct_optimizer, construct_scheduler, get_lr
from datasets import construct_dataloader, unnorm
from algos import get_algo
from evaluation import get_tasks

# NEW - externalize data preproc to run on GPU
from datasets.data_augment import get_data_preprocess

logger = logging.get_logger(__name__)

# NEW: temporary bypass to turn off all val and eval
TRAIN_ONLY = False
# NEW: turn off TQDM
USE_TQDM = False



# NEW - apply preprocessing ops to views on GPU
def preproc_views(view_0, view_1, data_preprocess):
    bsize = view_0.size()[0]
    view_0 = view_0.cuda()
    view_1 = view_1.cuda()
    videos = []
    for bidx in range(bsize):
        view_0_b = data_preprocess(view_0[bidx,...])
        view_1_b = data_preprocess(view_1[bidx,...])
        vid = torch.stack([view_0_b, view_1_b], dim=0)
        videos.append(vid)
    if bsize == 1:
        videos = torch.unsqueeze(videos[0], dim=0)
    else:
        videos = torch.stack(videos, dim=0)
    return videos



def train(cfg, train_loader, model, optimizer, scheduler, algo, cur_epoch, summary_writer, data_preprocess):
    model.train()
    optimizer.zero_grad()
    data_size = len(train_loader)
    # DistributedSampler shuffle based on epoch and seed
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(cur_epoch)
        logger.info(f"update the training sampler to epoch {cur_epoch}")
    if hasattr(train_loader.batch_sampler, 'set_epoch'):
        train_loader.batch_sampler.set_epoch(cur_epoch)
        logger.info(f"update the training batch sampler to epoch {cur_epoch}")
    total_loss = {}


    if USE_TQDM and du.is_root_proc():
        train_loader = tqdm(train_loader, total=len(train_loader))
    
    t1 = time.time()
    tmt = {} # timing marker tracker
    tmc = 0
    for i in range(10):
        tmt[i] = 0.0

    for cur_iter, (videos, _labels, seq_lens, chosen_steps, video_masks, names) in enumerate(train_loader):
        # NEW shifted video preproc to GPU-side
        view_0, view_1 = videos

        tmc += 1
        tmt[0] += time.time() - t1
        t1 = time.time()

        videos = preproc_views(view_0, view_1, data_preprocess)

        tmt[1] += time.time() - t1
        t1 = time.time()

        t0 = time.time()
        optimizer.zero_grad()
        if cfg.USE_AMP:
            # torch.autograd.set_detect_anomaly(True)
            torch.autograd.set_detect_anomaly(False)
            scaler = algo.scaler
            with torch.cuda.amp.autocast():
                if cfg.TRAINING_ALGO == 'classification':
                    loss_dict = algo.compute_loss(model, videos, _labels, seq_lens, chosen_steps, video_masks)
                else:
                    loss_dict = algo.compute_loss(model, videos, seq_lens, chosen_steps, video_masks)

            tmt[2] += time.time() - t1
            t1 = time.time()

            loss = loss_dict["loss"]
            scaler.scale(loss).backward()
            if cfg.OPTIMIZER.GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
            
            tmt[3] += time.time() - t1
            t1 = time.time()

            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()

            tmt[4] += time.time() - t1
            t1 = time.time()

        else:
            if cfg.TRAINING_ALGO == 'classification':
                loss_dict = algo.compute_loss(model, videos, _labels, seq_lens, chosen_steps, video_masks)
            else:
                loss_dict = algo.compute_loss(model, videos, seq_lens, chosen_steps, video_masks)
            loss = loss_dict["loss"]
            # Perform the backward pass.
            loss.backward()
            # Update the parameters.
            if cfg.OPTIMIZER.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
            optimizer.step()

        for key in loss_dict:
            loss_dict[key][torch.isnan(loss_dict[key])] = 0
            if key not in total_loss:
                total_loss[key] = 0
            total_loss[key] += du.all_reduce([loss_dict[key]])[0].item() / data_size

        tmt[5] += time.time() - t1
        t1 = time.time()

        if cfg.NUM_GPUS == 1 and cur_iter % cfg.LOGGING.REPORT_INTERVAL == 0:
            # print(names)
            logger.info(f"iter {data_size * cur_epoch + cur_iter}, training loss: {loss.item():.3f}")
            # visual_video = videos[0]
            # if cfg.SSL:
            #     for i, v in enumerate(visual_video):
            #         summary_writer.add_video(f'{names[0]}_view{i}', unnorm(v[::cfg.DATA.NUM_CONTEXTS]).unsqueeze(0), 0, fps=4)
            # else:
            #     summary_writer.add_video(f'{names[0]}', unnorm(visual_video[::cfg.DATA.NUM_CONTEXTS]).unsqueeze(0), 0, fps=4)

        tmt[6] += time.time() - t1
        t1 = time.time()

    # print timing markers
    for i in range(10):
        if tmt[i] > 0.0:
            tm_avg = tmt[i] / tmc
            print('marker %i: %f'%(i, tm_avg))
    print('loops: %i'%tmc)

    summary_writer.add_scalar('train/learning_rate', get_lr(optimizer)[0], cur_epoch)
    for key in total_loss:
        summary_writer.add_scalar(f'train/{key}', total_loss[key], cur_epoch)
    logger.info("epoch {}, train loss: {:.3f}".format(cur_epoch, total_loss["loss"]))
    
    if cur_epoch != cfg.TRAIN.MAX_EPOCHS-1:
        scheduler.step()

def val(cfg, val_loader, model, algo, cur_epoch, summary_writer, data_preprocess):
    model.eval()
    data_size = len(val_loader)
    total_loss = {}

    # NEW run preproc on GPU side
    # data_preprocess = val_loader.dataset.get_preproc()

    with torch.no_grad():
        for cur_iter, (videos, labels, seq_lens, chosen_steps, video_masks, names) in enumerate(val_loader):
            
            # TODO - this currently only supports batch size = 1

            # # NEW shifted video preproc to GPU-side
            # view_0 = videos[0][0,...]
            # view_1 = videos[1][0,...]
            # view_0 = view_0.cuda()
            # view_0 = data_preprocess(view_0)
            # view_1 = view_1.cuda()
            # view_1 = data_preprocess(view_1)
            # videos = torch.stack([view_0, view_1], dim=0)
            # videos = torch.unsqueeze(videos, 0)
            # TODO - clean up

            # NEW shifted video preproc to GPU-side
            view_0 = videos[0]
            view_1 = videos[1]
            videos = preproc_views(view_0, view_1, data_preprocess)

            if cfg.USE_AMP:
                with torch.cuda.amp.autocast():
                    if cfg.TRAINING_ALGO == 'classification':
                        loss_dict = algo.compute_loss(model, videos, labels, seq_lens, chosen_steps, video_masks, training=False)
                    else:
                        loss_dict = algo.compute_loss(model, videos, seq_lens, chosen_steps, video_masks, training=False)
            else:
                if cfg.TRAINING_ALGO == 'classification':
                    loss_dict = algo.compute_loss(model, videos, labels, seq_lens, chosen_steps, video_masks, training=False)
                else:
                    loss_dict = algo.compute_loss(model, videos, seq_lens, chosen_steps, video_masks, training=False)

            for key in loss_dict:
                loss_dict[key][torch.isnan(loss_dict[key])] = 0
                if key not in total_loss:
                    total_loss[key] = 0
                total_loss[key] += du.all_reduce([loss_dict[key]])[0].item() / data_size

        if cfg.NUM_GPUS == 1:
            print(names)
            visual_video = videos[0]
            if cfg.SSL:
                for i, v in enumerate(visual_video):
                    summary_writer.add_video(f'{names}_view{i}', unnorm(v[::2]).unsqueeze(0), 0, fps=4)
            else:
                summary_writer.add_video(f'{names}', unnorm(visual_video[::2]).unsqueeze(0), 0, fps=4)

    for key in total_loss:
        summary_writer.add_scalar(f'val/{key}', total_loss[key], cur_epoch)
    logger.info("epoch {}, val loss: {:.3f}".format(cur_epoch, total_loss["loss"]))

def main():
    args = parse_args()
    cfg = load_config(args)
    setup_train_dir(cfg, cfg.LOGDIR, args.continue_train)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.NUM_GPUS = torch.cuda.device_count() # num_gpus_per_machine
    args.world_size = int(os.getenv('WORLD_SIZE')) # total_gpus
    print('NUM_GPUS: ' + str(cfg.NUM_GPUS))
    print('WORLD_SIZE:' + str(args.world_size))

    if os.environ.get('OMPI_COMM_WORLD_SIZE') is None:
        args.rank = args.local_rank
    else:
        args.node_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        args.rank = args.node_rank * torch.cuda.device_count() + args.local_rank
    logger.info(f'Node info: rank {args.rank} of world size {args.world_size}')
    cfg.args = args
    print('args.rank: ' + str(args.rank))

    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)

    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # distributed logging and ignore warning message
    logging.setup_logging(cfg.LOGDIR)
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model
    model = build_model(cfg)
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    # model = model.to(f'cuda:{args.local_rank}')
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print('args.local_rank:' + str(args.local_rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank], 
            output_device = args.local_rank, find_unused_parameters=True)

    optimizer = construct_optimizer(model, cfg)
    algo = get_algo(cfg)

    # Setup Dataset Iterators from train and val datasets.
    # NEW - externalize data preproc to run on GPU
    train_loader, train_emb_loader = construct_dataloader(cfg, "train", no_eval=TRAIN_ONLY)
    train_preproc = get_data_preprocess(cfg, "train")
    if not TRAIN_ONLY:
        val_loader, val_emb_loader = construct_dataloader(cfg, "val")
        val_preproc = get_data_preprocess(cfg, "val")
    iterator_tasks, embedding_tasks = get_tasks(cfg)

    if cfg.USE_AMP:
        algo.scaler = torch.cuda.amp.GradScaler()
        logger.info("Initializing mixed precision done.")

    """Trains model and evaluates on relevant downstream tasks."""
    start_epoch = load_checkpoint(cfg, model, optimizer)
    cfg.TRAIN.MAX_ITERS = cfg.TRAIN.MAX_EPOCHS * len(train_loader)
    scheduler = construct_scheduler(optimizer, cfg)

    for cur_epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCHS):
        logger.info(f"Traning epoch {cur_epoch}/{cfg.TRAIN.MAX_EPOCHS}, {len(train_loader)} iters each epoch")
        print('running train...')
        t0 = time.time()
        train(cfg, train_loader, model, optimizer, scheduler, algo, cur_epoch, summary_writer, train_preproc)
        print('train done in (m): ' + str((time.time()-t0)/60.0))
        if not TRAIN_ONLY and ((cur_epoch+1) % cfg.EVAL.VAL_INTERVAL == 0 or cur_epoch == cfg.TRAIN.MAX_EPOCHS-1):
            print('running val...')
            t0 = time.time()
            val(cfg, val_loader, model, algo, cur_epoch, summary_writer, val_preproc)
            print('val done in (m): ' + str((time.time()-t0)/60.0))
            print('running evaluate_once...')
            t0 = time.time()
            if cfg.DATASETS[0] == "finegym":
                from evaluate_finegym import evaluate_once
                evaluate_once(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader, 
                                iterator_tasks, embedding_tasks, cur_epoch, summary_writer)
            elif du.is_root_proc():
                from evaluate import evaluate_once
                evaluate_once(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader, 
                                iterator_tasks, embedding_tasks, cur_epoch, summary_writer)
            print('evaluate_once done in (m): ' + str((time.time()-t0)/60.0))
        if du.is_root_proc() and ((cur_epoch+1) % cfg.CHECKPOINT.SAVE_INTERVAL == 0 or cur_epoch == cfg.TRAIN.MAX_EPOCHS-1):
            save_checkpoint(cfg, model, optimizer, cur_epoch)
        du.synchronize()

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
