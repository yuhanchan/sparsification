import json
import os
import sys
import os.path as osp

import shlex
import subprocess
import multiprocessing
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import myLogger
import sparsifier
import dataLoader
import workload

import argparse
from typing import Union


# def call_shell_cmd(i, cmd):
#     myLogger.info(f'Process {i} started')
#     myLogger.info(f'Process {i} cmd: {cmd}')
#     p = subprocess.Popen(cmd, shell=True)
#     our, err = p.communicate()
#     myLogger.info(f'Process {i} finished')
#     return (i, our, err)


def init_data():
    """
    This function generates all pruned dataset files with parameters specified in config.json.
    This function need only run once on new machine.
    """    
    # setup logger
    myLogger.set_level('INFO')
    myLogger.get_logger(__name__)
    
    # Load config file
    myLogger.info(f'Loading config')
    config = json.load(open('config.json'))
    
    reddit = dataLoader.Reddit()
    reddit2 = dataLoader.Reddit2()
    ogbn_products = dataLoader.ogbn_products()
    
    # random prune 
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(sparsifier.random_sparsify, dataset=reddit, dataset_name="Reddit", drop_rate=drop_rate): drop_rate for drop_rate in config["Reddit"]['drop_rate']}
        futures.update({executor.submit(sparsifier.random_sparsify, dataset=reddit2, dataset_name="Reddit2", drop_rate=drop_rate): drop_rate for drop_rate in config["Reddit2"]['drop_rate']})
        futures.update({executor.submit(sparsifier.random_sparsify, dataset=ogbn_products, dataset_name="ogbn_products", drop_rate=drop_rate): drop_rate for drop_rate in config["ogbn_products"]['drop_rate']})
        
        for future in futures:
            drop_rate = futures[future]
            try:
                myLogger.info(f"Processing drop rate {drop_rate}")
                future.result()
            except:
                myLogger.error(f"Error processing drop rate {drop_rate}")
                sys.exit(1)
            else:
                myLogger.info(f"Finished with drop rate {drop_rate}")

    # degree
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(sparsifier.in_degree_sparsify, dataset=reddit, dataset_name="Reddit", degree_thres=thres, config=config): thres for thres in config["Reddit"]['degree_thres']}
        futures.update({executor.submit(sparsifier.in_degree_sparsify, dataset=reddit2, dataset_name="Reddit2", degree_thres=thres, config=config): thres for thres in config["Reddit2"]['degree_thres']})
        futures.update({executor.submit(sparsifier.in_degree_sparsify, dataset=ogbn_products, dataset_name="ogbn_products", degree_thres=thres, config=config): thres for thres in config["ogbn_products"]['degree_thres']})

        futures.update({executor.submit(sparsifier.out_degree_sparsify, dataset=reddit, dataset_name="Reddit", degree_thres=thres, config=config): thres for thres in config["Reddit"]['degree_thres']})
        futures.update({executor.submit(sparsifier.out_degree_sparsify, dataset=reddit2, dataset_name="Reddit2", degree_thres=thres, config=config): thres for thres in config["Reddit2"]['degree_thres']})
        futures.update({executor.submit(sparsifier.out_degree_sparsify, dataset=ogbn_products, dataset_name="ogbn_products", degree_thres=thres, config=config): thres for thres in config["ogbn_products"]['degree_thres']})

        for future in futures:
            thres = futures[future]
            try:
                myLogger.info(f'Processing with thres {thres}')
                future.result()
            except:
                myLogger.error(f'Error with thres {thres}') 
                sys.exit(1)
            else:
                myLogger.info(f'Finished with thres {thres}')
    
    # er
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(sparsifier.er_sparsify, dataset=reddit, dataset_name="Reddit", epsilon=epsilon, config=config): epsilon for epsilon in config["Reddit"]['er_epsilon']}
        futures.update({executor.submit(sparsifier.er_sparsify, dataset=reddit2, dataset_name="Reddit2", epsilon=epsilon, config=config): epsilon for epsilon in config["Reddit2"]['er_epsilon']})
        futures.update({executor.submit(sparsifier.er_sparsify, dataset=ogbn_products, dataset_name="ogbn_products", epsilon=epsilon, config=config): epsilon for epsilon in config["ogbn_products"]['er_epsilon']})
        
        for future in futures:
            epsilon = futures[future]
            try:
                myLogger.info(f'Processing with epsilon {epsilon}')
                future.result()
            except:
                myLogger.error(f'Error with epsilon {epsilon}')
                sys.exit(1)
            else:
                myLogger.info(f'Finished with epsilon {epsilon}')
                
                
def main():
    # setup logger
    myLogger.set_level('INFO')
    myLogger.get_logger(__name__)
    
    # setup dir
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir_root = osp.join(current_file_dir, "experiments")
    os.makedirs(experiment_dir_root, exist_ok=True)
    
    # parse args
    parser = argparse.ArgumentParser(description='Top level script')
    parser.add_argument('-w', '--workload', type=str, required=True, help='workload to run')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset to use')
    parser.add_argument('-s', '--sparsifier', type=str, required=True, help='sparsifier to use')
    parser.add_argument('-p', '--prune', type=str, required=True, 
                        help="Prune argument to use." +
                             "For random, it's used as prune rate." + 
                             "For in_degree/out_degree, it's used as degree threshold." + 
                             "For er, it's used as epsilon.")
    parser.add_argument('-f', '--force', action='store_true', default=False, 
                        help='Set this option to run prune arguments not in config.json')
    args = parser.parse_args()
    
    # Sanity check
    assert args.workload in ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc', 'ClusterGCN'], f"Unknown workload: {args.workload}"
    assert args.dataset in ['Reddit', 'Reddit2', 'ogbn_products'], f"Unknown dataset: {args.dataset}"
    assert args.sparsifier in ['baseline', 'random', 'in_degree', 'out_degree', 'er'], f"Unknown sparsifier: {args.sparsifier}"
    args.prune = float(args.prune) if '.' in args.prune else int(args.prune)
    
    # Load config file
    myLogger.info(f'Loading config')
    config = json.load(open('config.json'))
    
    if args.workload == 'ClusterGCN':
        # Load dataset
        if args.dataset == 'Reddit':
            dataset = dataLoader.Reddit()
        elif args.dataset == 'Reddit2':
            dataset = dataLoader.Reddit2()
        elif args.dataset == 'ogbn_products':
            dataset = dataLoader.ogbn_products()
        else:
            myLogger.error(f"Unknown dataset: {args.dataset}")
            
        # Apply sparsifier
        if args.sparsifier == 'baseline':
            experiment_dir = osp.join(experiment_dir_root, f"{args.workload}/{args.dataset}/{args.sparsifier}")
        elif args.sparsifier == 'random':
            if not args.force and args.prune not in config[args.dataset]['drop_rate']:
                myLogger.error(f"Prune rate {args.prune} for random prune for {args.dataset} not found in config.json, if you want to force this, please set -f. Exiting...")
                sys.exit(1)
            dataset = sparsifier.random_sparsify(dataset=dataset, dataset_name=args.dataset, drop_rate=args.prune)
            experiment_dir = osp.join(experiment_dir_root, f"{args.workload}/{args.dataset}/{args.sparsifier}/{args.prune}")
        elif args.sparsifier == 'in_degree':
            if not args.force and args.prune not in config[args.dataset]['degree_threshold']:
                myLogger.error(f"Degree threshold {args.prune} for in_degree prune for {args.dataset} not found in config.json, if you want to force this, please set -f. Exiting...")
                sys.exit(1)
            dataset = sparsifier.in_degree_sparsify(dataset=dataset, dataset_name=args.dataset, degree_thres=args.prune, config=config)
            experiment_dir = osp.join(experiment_dir_root, f"{args.workload}/{args.dataset}/{args.sparsifier}/{config[args.dataset]['degree_thres_to_drop_rate_map'][str(args.prune)]}")
        elif args.sparsifier == 'out_degree':
            if not args.force and args.prune not in config[args.dataset]['degree_threshold']:
                myLogger.error(f"Degree threshold {args.prune} for out_degree prune for {args.dataset} not found in config.json, if you want to force this, please set -f. Exiting...")
                sys.exit(1)
            dataset = sparsifier.out_degree_sparsify(dataset=dataset, dataset_name=args.dataset, degree_thres=args.prune, config=config)
            experiment_dir = osp.join(experiment_dir_root, f"{args.workload}/{args.dataset}/{args.sparsifier}/{config[args.dataset]['degree_thres_to_drop_rate_map'][str(args.prune)]}")
        elif args.sparsifier == 'er':
            if not args.force and args.prune not in config[args.dataset]['er_epsilon']:
                myLogger.error(f"Epsilon {args.prune} for er prune for {args.dataset} not found in config.json, if you want to force this, please set -f. Exiting...")
            dataset = sparsifier.er_sparsify(dataset=dataset, dataset_name=args.dataset, epsilon=args.prune, config=config)
            experiment_dir = osp.join(experiment_dir_root, f"{args.workload}/{args.dataset}/{args.sparsifier}/{config[args.dataset]['er_epsilon_to_drop_rate_map'][str(args.prune)]}")
            
        print(f"Experiment dir: {experiment_dir}")
        os.makedirs(experiment_dir, exist_ok=True)
        # Invoke workload
        
    else:
        # set dataset path
        if args.sparsifier == 'baseline':
            edge_list_path = osp.join(current_file_dir, 'data', args.dataset, 'raw', 'edge_list.el')
            experiment_dir = osp.join(experiment_dir_root, f"{args.workload}/{args.dataset}/{args.sparsifier}")
        elif args.sparsifier == 'random':
            if args.prune not in config[args.dataset]['drop_rate']:
                myLogger.error(f"Prune rate {args.prune} for random prune for {args.workload} not found in config.json. Exiting...")
                sys.exit(1)
            edge_list_path = osp.join(current_file_dir, 'data', args.dataset, 'pruned', args.sparsifier, str(args.prune), 'edge_list.el')
            experiment_dir = osp.join(experiment_dir_root, f"{args.workload}/{args.dataset}/{args.sparsifier}/{args.prune}")
        elif args.sparsifier == 'in_degree' or args.sparsifier == 'out_degree':
            if args.prune not in config[args.dataset]['degree_thres']:
                myLogger.error(f"Degree threshold {args.prune} for in_degree/out_degree prune for {args.workload} not found in config.json. Exiting...")
                sys.exit(1)
            drop_rate = config[args.dataset]['degree_thres_to_drop_rate_map'][str(args.prune)]
            edge_list_path = osp.join(current_file_dir, 'data', args.dataset, 'pruned', args.sparsifier, str(drop_rate), 'edge_list.el')
            experiment_dir = osp.join(experiment_dir_root, f"{args.workload}/{args.dataset}/{args.sparsifier}/{drop_rate}")
        elif args.sparsifier == 'er':
            if args.prune not in config[args.dataset]['er_epsilon']:
                myLogger.error(f"Epsilon {args.prune} for er prune for {args.workload} not found in config.json. Exiting...")
                sys.exit(1)
            drop_rate = config[args.dataset]['er_epsilon_to_drop_rate_map'][str(args.prune)]
            edge_list_path = osp.join(current_file_dir, 'data', args.dataset, 'pruned', args.sparsifier, str(drop_rate), 'edge_list.wel')
            experiment_dir = osp.join(experiment_dir_root, f"{args.workload}/{args.dataset}/{args.sparsifier}/{drop_rate}")

        print(f"Experiment dir: {experiment_dir}")
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"edge_list_path: {edge_list_path}")
        assert osp.exists(edge_list_path), f"Edge list file {edge_list_path} not found."
        
        # Invoke workload
        if args.workload == 'bc':
            pass
        elif args.workload == 'bfs':
            pass
        elif args.workload == 'cc':
            pass
        elif args.workload == 'cc_sv':
            pass
        elif args.workload == 'pr':
            workload.pr(**{"-f": edge_list_path, "-n": "1", "-v": "", "-a": "", "-z": osp.join(experiment_dir, 'analysis.txt'), ">": osp.join(experiment_dir, "stdout.txt")})
        elif args.workload == 'pr_spmv':
            pass
        elif args.workload == 'sssp':
            pass
        elif args.workload == 'tc':
            pass
        elif args.workload == 'ClusterGCN':
            pass
        
        
if __name__ == '__main__':
    main()