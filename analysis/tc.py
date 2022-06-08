import matplotlib.pyplot as plt
import os
import os.path as osp
import operator
import numpy as np

def cpu_time(folder, dataset="Reddit"):
    outfile_path = osp.join(osp.dirname(osp.realpath(__file__)), "tc", dataset, 'cpu_time.txt')
    os.makedirs(osp.dirname(outfile_path), exist_ok=True)
    outfile = open(outfile_path, 'w')
    outfile.write('prune rate: cpu_time (s), Relabel time (s), Total time (s)\n\n')
    
    plot_dict = {}

    num_trail = 0
    cpu_time = []
    relabel_time = []
    x = []
    y = []
    y1 = []
    outfile.write(f"------------------baseline--------------------\n")
    with open(osp.join(folder, dataset, 'baseline/stdout.txt')) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if 'CPU Time' in lines[i] and not 'Relabel' in lines[i]:
                cpu_time.append(float(lines[i].strip().split(' ')[2]))
                if 'Relabel' in lines[i-1]:
                    relabel_time.append(float(lines[i-1].strip().split(' ')[3]))
                else:
                    relabel_time.append(0)
                num_trail += 1
        cpu_time_mean = np.mean(cpu_time)
        cpu_time_std = np.std(cpu_time)
        cpu_time = [x for x in cpu_time if x < cpu_time_mean + 1.5 * cpu_time_std and x > cpu_time_mean - 1.5 * cpu_time_std]
        relabel_time_mean = np.mean(relabel_time)
        relabel_time_std = np.std(relabel_time)
        relabel_time = [x for x in relabel_time if x < relabel_time_mean + 1.5 * relabel_time_std and x > relabel_time_mean - 1.5 * relabel_time_std]
        outfile.write(f"baseline: {np.mean(cpu_time) :.3f} ({num_trail - len(cpu_time)} Outliers), {np.mean(relabel_time) :.3f} ({num_trail - len(cpu_time)} Outliers), {np.mean(cpu_time) + np.mean(relabel_time) :.3f}\n\n")
        x.append(0)
        y.append(np.mean(cpu_time)) 
        y1.append(np.mean(cpu_time)+np.mean(relabel_time))
        plot_dict['baseline'] = [x, y, y1]
    
    for prune_algo in ["random", "in_degree", "out_degree", "er"]:
        outfile.write(f"------------------{prune_algo}--------------------\n")
        x = []
        y = []
        y1 = []
        for subdir in sorted(os.listdir(osp.join(folder, dataset, prune_algo))):
            num_trail = 0
            cpu_time = []
            relabel_time = []
            with open(osp.join(folder, dataset, prune_algo, subdir, 'stdout.txt')) as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if 'CPU Time' in lines[i] and not 'Relabel' in lines[i]:
                        cpu_time.append(float(lines[i].strip().split(' ')[2]))
                        if 'Relabel' in lines[i-1]:
                            relabel_time.append(float(lines[i-1].strip().split(' ')[3]))
                        else:
                            relabel_time.append(0)
                        num_trail += 1
                cpu_time_mean = np.mean(cpu_time)
                cpu_time_std = np.std(cpu_time)
                cpu_time = [x for x in cpu_time if x < cpu_time_mean + 1.5 * cpu_time_std and x > cpu_time_mean - 1.5 * cpu_time_std]
                relabel_time_mean = np.mean(relabel_time)
                relabel_time_std = np.std(relabel_time)
                relabel_time = [x for x in relabel_time if x < relabel_time_mean + 1.5 * relabel_time_std and x > relabel_time_mean - 1.5 * relabel_time_std]
                relabel_time = [0] * num_trail if len(relabel_time) == 0 else relabel_time
                outfile.write(f"{subdir}: {np.mean(cpu_time) :.3f} ({num_trail - len(cpu_time)} Outliers), {np.mean(relabel_time) :.3f} ({num_trail - len(relabel_time)} Outliers), {np.mean(cpu_time) + np.mean(relabel_time) :.3f}\n")
                x.append(float(subdir))
                y.append(np.mean(cpu_time))
                y1.append(np.mean(cpu_time)+np.mean(relabel_time))
        plot_dict[prune_algo] = [plot_dict['baseline'][0] + x, plot_dict['baseline'][1] + y, plot_dict['baseline'][2] + y1]
        outfile.write("\n")
    outfile.close()
    
    del plot_dict['baseline']
    fig, ax = plt.subplots(figsize=(5, 5))
    for key, value in plot_dict.items():
        ax.plot(value[0], value[1], label=key)
    ax.set_xlabel('prune rate')
    ax.set_ylabel('cpu time (s)')
    ax.legend()
    ax.set_title(f"CPU time (not include relabel time) for {dataset}")
    fig.savefig(osp.join(osp.dirname(outfile_path), 'cpu_time.png'))

    fig, ax = plt.subplots(figsize=(5, 5))
    for key, value in plot_dict.items():
        ax.plot(value[0], value[2], label=key)
    ax.set_xlabel('prune rate')
    ax.set_ylabel('cpu time (s)')
    ax.legend()
    ax.set_title(f"CPU time (include relabel time) for {dataset}")
    fig.savefig(osp.join(osp.dirname(outfile_path), 'cpu_time+relabel_time.png'))
    

def triangle_count(folder, dataset='Reddit'):
    outfile_path = osp.join(osp.dirname(osp.realpath(__file__)), "tc", dataset, 'triangle_count.txt')
    os.makedirs(osp.dirname(outfile_path), exist_ok=True)
    outfile = open(outfile_path, 'w')
    outfile.write('Triangle Count\n\n')
    
    plot_dict = {}

    outfile.write(f"------------------baseline--------------------\n")
    x = []
    y = []
    with open(osp.join(folder, dataset, 'baseline/analysis.txt')) as f:
        triangle = int(f.readline().strip().split(' ')[0])
        outfile.write(f"baseline: {triangle:,}\n\n")
    x.append(0)
    y.append(triangle) 
    plot_dict['baseline'] = [x, y]

    for prune_algo in ["random", "in_degree", "out_degree", "er"]:
        outfile.write(f"------------------{prune_algo}--------------------\n")
        x = []
        y = []
        for subdir in sorted(os.listdir(osp.join(folder, dataset, prune_algo))):
            with open(osp.join(folder, dataset, prune_algo, subdir, 'analysis.txt')) as f:
                triangle = int(f.readline().strip().split(' ')[0])
                outfile.write(f"{subdir}: {triangle:,}\n")
                x.append(float(subdir))
                y.append(triangle)
        plot_dict[prune_algo] = [plot_dict['baseline'][0] + x, plot_dict['baseline'][1] + y]
        outfile.write("\n")
    outfile.close()
    
    del plot_dict['baseline']
    fig, ax = plt.subplots(figsize=(5, 5))
    for key, value in plot_dict.items():
        ax.plot(value[0], value[1], label=key)
    ax.set_xlabel('prune rate')
    ax.set_ylabel('Triangle Count')
    ax.legend()
    ax.set_title(f"Triangle for {dataset}")
    fig.savefig(osp.join(osp.dirname(outfile_path), 'triangle_count.png'))


if __name__ == "__main__":
    cpu_time('../experiments/tc/', dataset='Reddit')
    cpu_time('../experiments/tc/', dataset='Reddit2')
    cpu_time('../experiments/tc/', dataset='ogbn_products')
    
    triangle_count('../experiments/tc/', dataset='Reddit')
    triangle_count('../experiments/tc/', dataset='Reddit2')
    triangle_count('../experiments/tc/', dataset='ogbn_products')
    
