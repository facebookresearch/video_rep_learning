# helper script to automate the generation of job launching scripts
# NEW: use the "--micro" flag to make a micro job for debugging (2 training epochs)
import argparse
import os
import random



def gen_trial(args, trial_num=None, trial_seed=None):
    if args.micro and args.eval:
        print('ERROR: cannot run with both --micro and --eval')
        return
    if args.micro and trial_num is not None:
        print('ERROR: cannot run with --micro and trial_mode')
    if trial_num is not None and trial_seed is None:
        print('ERROR: must specify trial_seed if using trial_num')

    # check config path valid
    config_path = args.cfg
    assert os.path.isfile(config_path)
    config_name = os.path.basename(config_path).replace('.yml','')
    
    # select template file
    if args.one:
        template_name = "job_template_1gpu.slurm"
    else:
        template_name = "job_template_4gpu.slurm"
    
    # generate destination name
    dest_folder = args.dst
    os.makedirs(dest_folder, exist_ok=True)
    if args.micro:
        dest_file = os.path.join(dest_folder, "job_micro_%s.slurm"%config_name)
    elif args.eval:
        dest_file = os.path.join(dest_folder, "job_eval_%s.slurm"%config_name)
    else:
        dest_file = os.path.join(dest_folder, "job_%s.slurm"%config_name)
    if trial_num is not None:
        dest_file = dest_file.replace('.slurm','_trial%02i.slurm'%trial_num)
    
    # generate job file
    with open(dest_file, 'w') as fout:
        with open(template_name, 'r') as fin:
            for line in fin:
                # if (args.eval or args.micro) and "#SBATCH --time" in line:
                #     line = "#SBATCH --time=2:00:00\n" # reduce time requested
                
                # special names for micro, eval, or trials
                if args.micro and ("--logdir" in line or "--output" in line):
                    line = line.replace('CONFIGNAME','micro_CONFIGNAME')
                if trial_num is not None and ("--logdir" in line or "--output" in line):
                    line = line.replace('CONFIGNAME','CONFIGNAME-trial%02i'%trial_num)
                if args.eval and "--output" in line:
                    line = line.replace('CONFIGNAME','eval_CONFIGNAME')
                
                # insert config name
                line = line.replace('CONFIGNAME',config_name)

                # append special settings for micro, eval, or trials
                if "--logdir" in line:
                    special = '\n'
                    if args.micro:
                        special = special.replace('\n',' EVAL.VAL_INTERVAL 1 CHECKPOINT.SAVE_INTERVAL 1 TRAIN.MAX_EPOCHS 2\n')
                    if trial_num is not None:
                        special = special.replace('\n',' RNG_SEED %i\n'%trial_seed)
                    if args.one:
                        special = special.replace('\n',' TRAIN.BATCH_SIZE 4\n')
                    if special != '\n':
                        special = " --opts" + special
                        line = line.replace('\n',special)

                # change to eval
                if args.eval and "train.py" in line:
                    line = line.replace("train.py", "evaluate.py")
                fout.write(line)
    
    # provide launch command
    print('sbatch %s'%dest_file)



def main(args):
    random.seed(args.seed)
    # micro mode will not run with multiple trials, as it is only for debugging
    if args.micro:
        gen_trial(args)
        return
    # generate multiple trials with different random seeds
    for t in range(args.trials):
        ts = random.randint(1,10000)
        gen_trial(args, trial_num=t+1, trial_seed=ts)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="config file to launch with")
    parser.add_argument("--dst", help="destination folder for job script", default="slurm_jobs")
    parser.add_argument("--one", help="run as a 1 GPU job", action="store_true")
    parser.add_argument("--micro", help="make a micro job for debugging", action="store_true")
    parser.add_argument("--eval", help="calls evaluate.py instead of train.py", action="store_true")
    parser.add_argument("--trials", help="how many trials to create", default=3)
    parser.add_argument("--seed", help="random seed to generating per-trial seeds", default=1)
    args = parser.parse_args()
    main(args)