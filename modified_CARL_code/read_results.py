# helper to read multiple log files for multiple trials and report the combined results ± 2stdev
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# exclude all logs containing these terms
EXCLUDE_LOGS = ['micro']

def gather_res(log_dir, folder, all_res, metrics, fg=False):
    fp = os.path.join(log_dir, folder, 'stdout.log')
    if 'trial' in folder:
        tn = int(folder.split('-')[-1].replace('trial',''))
    else:
        tn = 0
    no_res = True
    print('reading: %s'%fp)
    max_e = -1
    with open(fp, 'r') as f:
        for line in f:
            # check for epoch on train line
            if 'Traning epoch' in line:
                l = line.split('Traning epoch ')[-1]
                l = l.split('/')[0]
                cur_e = int(l) + 1
            # check for results
            if not fg and 'metrics/all' in line:
                for m in metrics:
                    if m in line:
                        # parse result
                        v = line.split(m+': ')[-1]
                        v = v.replace('\n','')
                        v = float(v)
                        # log result
                        if tn not in all_res:
                            all_res[tn] = {}
                        if m not in all_res[tn]:
                            all_res[tn][m] = {}
                        all_res[tn][m][cur_e] = v
                        # log max result epoch
                        max_e = max(cur_e, max_e)
                        # register result loaded
                        no_res = False
                        break
            # check for finegym results
            elif fg and 'tensor(' in line:
                m = None
                if 'classification_1.0/train' in line:
                    m = 'classification_train'
                elif 'classification_1.0/val:' in line:
                    m = 'classification_val'
                if m is not None:
                    # parse result
                    v = line.split('tensor(')[-1].replace(')','')
                    v = v.replace('\n','')
                    v = float(v)
                    # log result
                    if tn not in all_res:
                        all_res[tn] = {}
                    if m not in all_res[tn]:
                        all_res[tn][m] = {}
                    all_res[tn][m][cur_e] = v
                    # log max result epoch
                    max_e = max(cur_e, max_e)
                    # register result loaded
                    no_res = False
                        
    if max_e > 0:
        print('max result epoch: %i'%max_e)
    if no_res: print('NO RESULTS')
    return max_e



def plot_results(all_res, tns, epcs, config_name, metrics, mult=100, plot_dir='plots'):
    os.makedirs(plot_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(32, 8))
    for m_i, m in enumerate(metrics):
        m_data = np.zeros([len(tns),len(epcs)], dtype=float)
        for e_i, e in enumerate(epcs):
            for t_i, t in enumerate(tns): 
                m_data[t_i, e_i] = all_res[t][m][e]
        m_data *= mult
        m_mean = np.mean(m_data, axis=0)
        m_std = np.std(m_data, axis=0)
        axs[m_i].errorbar(epcs, m_mean, yerr=(2*m_std))
        title = '%s - last: %.2f±%.2f'%(m, m_mean[-1], 2*m_std[-1])
        axs[m_i].set_title(title, fontsize=20)
        axs[m_i].tick_params(axis='both', which='major', labelsize=15)
        axs[m_i].tick_params(axis='both', which='minor', labelsize=15)
    fig.suptitle(config_name, fontsize=20)
    outname = os.path.join(plot_dir, config_name + '.png')
    plt.tight_layout()
    plt.savefig(outname)
    


def main(args):
    # check config path valid
    config_path = args.cfg
    assert os.path.isfile(config_path)
    config_name = os.path.basename(config_path).replace('.yml','')

    # check for matching logs
    files = os.listdir(args.ld)
    f_keep = []
    for f in files:
        f_base = f.split('-')[0]
        if f_base == config_name:
            f_keep.append(f)

    # check dataset
    is_fg = False
    if args.fg:
        is_fg = True
    elif 'finegym' in args.cfg:
        print('parsing as FineGym results')
        is_fg = True

    # select metrics to search for in logs
    if is_fg:
        metrics = ['classification_train', 'classification_val']
        mult = 1
    else:
        metrics = ['all_classification', 'all_event_completion', 'all_kendalls_tau', 'all_retrieval']
        mult = 100

    # display found files
    if len(f_keep) == 0:
        print('ERROR: found no matching trials for %s'%config_name)
        return
    f_keep = sorted(f_keep)
    print('Found %i matching trials for %s'%(len(f_keep), config_name))
    for f in f_keep:
        print(f)
    print('===')

    # gather results in this dictionary
    all_res = {}
    uniform_max = True
    max_all = None
    for f in f_keep:
        max_e = gather_res(args.ld, f, all_res, metrics, is_fg)
        if max_all is None:
            max_all = max_e
        if max_all != max_e:
            uniform_max = False
    print('===')
    if not uniform_max:
        print('WARNING: not all trials have finished')
        return

    # identity trials and epochs
    tns = sorted(list(all_res.keys()))
    epcs = sorted(list(all_res[tns[0]][metrics[0]].keys()))
    print('found %i trials'%len(tns))
    print('found results at epochs:')
    print(epcs)
    if args.emax > 0:
        epcs_new = []
        for e in epcs:
            if e <= args.emax:
                epcs_new.append(e)
        epcs = epcs_new
        print('limiting to epochs:')
        print(epcs)
    print('===')

    # show all results in dictionary
    if args.all:
        for m in metrics:
            print('---')
            print(m)
            for tn in tns:
                print('trial %i'%tn)
                res = all_res[tn][m]
                for e in epcs:
                    v = res[e] * mult
                    print('%03i - %.2f'%(e, v))
        print('===')

    # show average results with ±2 stdev
    if args.verbose:
        for m in metrics:
            print(m)
            for e in epcs:
                cur = []
                for tn in tns:
                    cur.append(all_res[tn][m][e])
                cur = np.array(cur) * mult
                cur_m = np.mean(cur)
                cur_s = np.std(cur)
                print('%03i : %.2f ± %.2f'%(e, cur_m, cur_s*2))
            print('---')
        print('===')

    # final results
    print('FINAL RESULTS (%i TRIALS) (%s)'%(len(f_keep), config_path))
    for t in ['AVG', 'MAX', 'MIN']:
        print(t)
        final_summary = ''
        for m in metrics:
            e = epcs[-1]
            cur = []
            for tn in tns:
                cur.append(all_res[tn][m][e] * mult)
            if args.all and t == 'AVG':
                for c in cur: print('%.2f'%c)
            if t == 'AVG':
                cur = np.array(cur)
                cur_m = np.mean(cur)
                cur_s = np.std(cur)
                print('%s : %.2f ± %.2f'%(m.ljust(20), cur_m, cur_s*2))
                final_summary += '%.2f ± %.2f\t'%(cur_m, cur_s*2)
            elif t == 'MAX':
                cur = np.array(cur)
                cur = np.max(cur)
                print('%s : %.2f'%(m.ljust(20), cur))
                final_summary += '%.2f\t'%(cur)
            else: # MIN
                cur = np.array(cur)
                cur = np.min(cur)
                print('%s : %.2f'%(m.ljust(20), cur))
                final_summary += '%.2f\t'%(cur)
        final_summary = final_summary[:-2]
        print('copy:')
        print(final_summary)
        print('---')

    # plot results
    if args.plot:
        plot_results(all_res, tns, epcs, config_name, metrics, mult)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="config file to search for results for")
    parser.add_argument("--ld", help="log dir to search for results in", default="/fsx/mwalmer/carl_logs")
    parser.add_argument("--all", help="print all results", action="store_true")
    parser.add_argument("--emax", help="limit epoch reading to a certain number, for partial result files", default=-1, type=int)
    parser.add_argument("--verbose", help="enable detailed printing", action="store_true")
    parser.add_argument("--plot", help="generate plots", action="store_true")
    parser.add_argument("--fg", help="parse FineGym Results", action="store_true")
    args = parser.parse_args()
    main(args)