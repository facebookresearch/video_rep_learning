# read multiple log files for multiple trials and report the combined results +- 2stdev
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# exclude all logs containing these terms
EXCLUDE_LOGS = ['micro']

# metrics to search for in logs
METRICS = ['all_classification', 'all_event_completion', 'all_kendalls_tau', 'all_retrieval']




def gather_res(log_dir, folder, all_res):
    fp = os.path.join(log_dir, folder, 'stdout.log')
    if 'trial' in folder:
        tn = int(folder.split('-')[-1].replace('trial',''))
    else:
        tn = 0
    no_res = True
    print('reading: %s'%fp)
    with open(fp, 'r') as f:
        for line in f:
            # check for epoch on val loss line
            if 'val loss' in line:
                l = line.split(', val loss')[0]
                l = l.split('epoch ')[-1]
                cur_e = int(l) + 1
            # check for results
            if 'metrics/all' in line:
                for m in METRICS:
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
                        # register result loaded
                        no_res = False
                        break
    if no_res: print('NO RESULTS')



def plot_results(all_res, tns, epcs, config_name, plot_dir='plots'):
    os.makedirs(plot_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(32, 8))
    for m_i, m in enumerate(METRICS):
        m_data = np.zeros([len(tns),len(epcs)], dtype=float)
        for e_i, e in enumerate(epcs):
            for t_i, t in enumerate(tns): 
                m_data[t_i, e_i] = all_res[t][m][e]
        m_data *= 100
        m_mean = np.mean(m_data, axis=0)
        m_std = np.std(m_data, axis=0)
        axs[m_i].errorbar(epcs, m_mean, yerr=(2*m_std))
        title = '%s - last: %.2f+-%.2f'%(m, m_mean[-1], 2*m_std[-1])
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
    for f in f_keep:
        gather_res(args.ld, f, all_res)
    print('===')

    # identity trials and epochs
    tns = sorted(list(all_res.keys()))
    epcs = sorted(list(all_res[tns[0]][METRICS[0]].keys()))
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
        for m in METRICS:
            print('---')
            print(m)
            for tn in tns:
                print('trial %i'%tn)
                res = all_res[tn][m]
                for e in epcs:
                    v = res[e] * 100
                    print('%03i - %.2f'%(e, v))
        print('===')

    # show average results with +-2 stdev
    for m in METRICS:
        print(m)
        for e in epcs:
            cur = []
            for tn in tns:
                cur.append(all_res[tn][m][e])
            cur = np.array(cur) * 100
            cur_m = np.mean(cur)
            cur_s = np.std(cur)
            print('%03i : %.2f +- %.2f'%(e, cur_m, cur_s*2))
        print('---')

    # show last epochs
    print('===')
    print('FINAL RESULTS')
    print('===')
    for m in METRICS:
        print(m)
        e = epcs[-1]
        cur = []
        for tn in tns:
            cur.append(all_res[tn][m][e] * 100)
        if args.all:
            for c in cur: print('%.2f'%c)
        cur = np.array(cur)
        cur_m = np.mean(cur)
        cur_s = np.std(cur)
        print('%03i : %.2f +- %.2f'%(e, cur_m, cur_s*2))
        print('---')


    # plot results
    plot_results(all_res, tns, epcs, config_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="config file to search for results for")
    parser.add_argument("--ld", help="log dir to search for results in", default="/fsx/mwalmer/carl_logs")
    parser.add_argument("--all", help="print all results", action="store_true")
    parser.add_argument("--emax", help="limit epoch reading to a certain number, for partial result files", default=-1, type=int)
    args = parser.parse_args()
    main(args)