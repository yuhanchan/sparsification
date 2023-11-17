import glob
from collections import defaultdict
from collections import OrderedDict
import os
import os.path as osp
import re
import argparse
import pandas as pd

PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ")
    print("please source env.sh at the top level of the project")
    exit(1)



def degreeDistribution(dataset_name):
    infile = osp.join(PROJECT_HOME, f"output_metric_raw/{dataset_name}/degreeDistribution/log")
    outfile = osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/degreeDistribution/log")
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    if osp.exists(infile):
        fin = open(infile, 'r')
        if not fin.read(1):
            fin.close()
            return
        fin.seek(0)
        df = pd.DataFrame(columns=['prune_algo', 'num_nodes', 'num_edges', 'Bhattacharyya_distance'])
        for line in fin:
            if line:
                line = line.strip().split()
                df = pd.concat([df, pd.DataFrame([[line[0], int(line[2]), int(line[4]), float(line[6])]], 
                                                columns=['prune_algo', 'num_nodes', 'num_edges', 'Bhattacharyya_distance'])], 
                                                ignore_index=True)
        fin.close()
        original_edge = int(df[df['prune_algo'] == 'original']['num_edges'].values[0])
        # add prune_rate column
        df['prune_rate'] = (1-df['num_edges'].astype(int)/original_edge).round(2)
        # for prune_algo == ForestFire, round prune_rate to closest 0.1 if prune rate is smaller than 0.93
        # df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'] = df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'].round(1)
        # group by same prune_algo and prune_rate, count occurance, and calculate mean and std, and add count, mean std columns to df
        df = df.groupby(['prune_algo', 'prune_rate']).agg({'Bhattacharyya_distance': ['count', 'mean', 'std']})
        # flatten multi-index columns, and merge multiple hearders
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
        # output to file
        df.to_csv(outfile, index=False, sep=',')

def Centrality(dataset_name):
    for c in ["EstimateBetweenness", "TopCloseness", "Katz", "Eigenvector", "PageRank"]:
        infile = osp.join(PROJECT_HOME, f"output_metric_raw/{dataset_name}/{c}Centrality/log")
        outfile = osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/{c}Centrality/log")
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        if osp.exists(infile):
            fin = open(infile, 'r')
            if not fin.read(1):
                fin.close()
                continue
            fin.seek(0)
            df = pd.DataFrame(columns=['prune_algo', 'num_nodes', 'num_edges', 'top_100_precision', 'top_100_correlation'])
            for line in fin:
                if line:
                    line = line.strip().split()
                    df = pd.concat([df, pd.DataFrame([[line[0], int(line[2]), int(line[4]), float(line[6]), float(line[8])]], 
                                                    columns=['prune_algo', 'num_nodes', 'num_edges', 'top_100_precision', 'top_100_correlation'])], 
                                                    ignore_index=True)
            fin.close()
            original_edge = int(df[df['prune_algo'] == 'original']['num_edges'].values[0])
            # add prune_rate column
            df['prune_rate'] = (1-df['num_edges'].astype(int)/original_edge).round(2)
            # for prune_algo == ForestFire, round prune_rate to closest 0.1 if prune rate is smaller than 0.93
            # df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'] = df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'].round(1)
            # group by same prune_algo and prune_rate, count occurance, and calculate mean and std, and add count, mean std columns to df
            df = df.groupby(['prune_algo', 'prune_rate']).agg({'top_100_precision': ['count', 'mean', 'std'], 'top_100_correlation': ['count', 'mean', 'std']})
            # flatten multi-index columns, and merge multiple hearders
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            df = df.reset_index()
            # output to file
            df.to_csv(outfile, index=False, sep=',')
        
def DetectCommunity(dataset_name):
    infile = osp.join(PROJECT_HOME, f"output_metric_raw/{dataset_name}/DetectCommunity/log")
    outfile = osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/DetectCommunity/log")
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    if osp.exists(infile):
        fin = open(infile, 'r')
        if not fin.read(1):
            fin.close()
            return
        fin.seek(0)
        df = pd.DataFrame(columns=['prune_algo', 'num_nodes', 'num_edges', 'num_community', 'modularity'])
        for line in fin:
            if line:
                line = line.strip().split()
                df = pd.concat([df, pd.DataFrame([[line[0], int(line[2]), int(line[4]), float(line[6]), float(line[8])]], 
                                                columns=['prune_algo', 'num_nodes', 'num_edges', 'num_community', 'modularity'])], 
                                                ignore_index=True)
        fin.close()
        original_edge = int(df[df['prune_algo'] == 'original']['num_edges'].values[0])
        # add prune_rate column
        df['prune_rate'] = (1-df['num_edges'].astype(int)/original_edge).round(2)
        # for prune_algo == ForestFire, round prune_rate to closest 0.1 if prune rate is smaller than 0.93
        # df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'] = df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'].round(1)
        # group by same prune_algo and prune_rate, count occurance, and calculate mean and std, and add count, mean std columns to df
        df = df.groupby(['prune_algo', 'prune_rate']).agg({'num_community': ['count', 'mean', 'std'], 'modularity': ['count', 'mean', 'std']})
        # flatten multi-index columns, and merge multiple hearders
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
        # output to file
        df.to_csv(outfile, index=False, sep=',')

def ClusteringF1Similarity(dataset_name):
    infile = osp.join(PROJECT_HOME, f"output_metric_raw/{dataset_name}/ClusteringF1Similarity/log")
    outfile = osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/ClusteringF1Similarity/log")
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    if osp.exists(infile):
        fin = open(infile, 'r')
        if not fin.read(1):
            fin.close()
            return
        fin.seek(0)
        df = pd.DataFrame(columns=['prune_algo', 'num_nodes', 'num_edges', 'F1_Similarity'])
        for line in fin:
            if line:
                line = line.strip().split()
                df = pd.concat([df, pd.DataFrame([[line[0], int(line[2]), int(line[4]), float(line[6])]], 
                                                columns=['prune_algo', 'num_nodes', 'num_edges', 'F1_Similarity'])], 
                                                ignore_index=True)
        fin.close()
        original_edge = int(df[df['prune_algo'] == 'original']['num_edges'].values[0])
        # add prune_rate column
        df['prune_rate'] = (1-df['num_edges'].astype(int)/original_edge).round(2)
        # for prune_algo == ForestFire, round prune_rate to closest 0.1 if prune rate is smaller than 0.93
        # df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'] = df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'].round(1)
        # group by same prune_algo and prune_rate, count occurance, and calculate mean and std, and add count, mean std columns to df
        df = df.groupby(['prune_algo', 'prune_rate']).agg({'F1_Similarity': ['count', 'mean', 'std']})
        # flatten multi-index columns, and merge multiple hearders
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
        # output to file
        df.to_csv(outfile, index=False, sep=',')

def QuadraticFormSimilarity(dataset_name):
    infile = osp.join(PROJECT_HOME, f"output_metric_raw/{dataset_name}/QuadraticFormSimilarity/log")
    outfile = osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/QuadraticFormSimilarity/log")
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    if osp.exists(infile):
        fin = open(infile, 'r')
        if not fin.read(1):
            fin.close()
            return
        fin.seek(0)
        df = pd.DataFrame(columns=['prune_algo', 'num_nodes', 'num_edges', 'QuadraticFormSimilarity'])
        for line in fin:
            if line:
                line = line.strip().split()
                df = pd.concat([df, pd.DataFrame([[line[0], int(line[2]), int(line[4]), float(line[17])]], 
                                                columns=['prune_algo', 'num_nodes', 'num_edges', 'QuadraticFormSimilarity'])], 
                                                ignore_index=True)
        fin.close()
        original_edge = int(df[df['prune_algo'] == 'original']['num_edges'].values[0])
        # add prune_rate column
        df['prune_rate'] = (1-df['num_edges'].astype(int)/original_edge).round(2)
        # for prune_algo == ForestFire, round prune_rate to closest 0.1 if prune rate is smaller than 0.93
        # df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'] = df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'].round(1)
        # group by same prune_algo and prune_rate, count occurance, and calculate mean and std, and add count, mean std columns to df
        df = df.groupby(['prune_algo', 'prune_rate']).agg({'QuadraticFormSimilarity': ['count', 'mean', 'std']})
        # flatten multi-index columns, and merge multiple hearders
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
        # output to file
        df.to_csv(outfile, index=False, sep=',')

def ApproximateDiameter(dataset_name):
    infile = osp.join(PROJECT_HOME, f"output_metric_raw/{dataset_name}/ApproximateDiameter/log")
    outfile = osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/ApproximateDiameter/log")
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    if osp.exists(infile):
        fin = open(infile, 'r')
        if not fin.read(1):
            fin.close()
            return
        fin.seek(0)
        df = pd.DataFrame(columns=['prune_algo', 'num_nodes', 'num_edges', 'ApproximateDiameter'])
        for line in fin:
            if line:
                line = line.strip().split()
                df = pd.concat([df, pd.DataFrame([[line[0], int(line[5]), int(line[8]), float(line[17].replace("(","").replace(",",""))]], 
                                                columns=['prune_algo', 'num_nodes', 'num_edges', 'ApproximateDiameter'])], 
                                                ignore_index=True)
        fin.close()
        original_edge = int(df[df['prune_algo'] == 'original']['num_edges'].values[0])
        # add prune_rate column
        df['prune_rate'] = (1-df['num_edges'].astype(int)/original_edge).round(2)
        # for prune_algo == ForestFire, round prune_rate to closest 0.1 if prune rate is smaller than 0.93
        # df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'] = df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'].round(1)
        # eliminate outliers
        # df = df.groupby(['prune_algo', 'prune_rate']).apply(lambda x: x[(x.ApproximateDiameter-x.ApproximateDiameter.mean()).abs() < 2*x.ApproximateDiameter.std()])
        # df = df.reset_index(drop=True)
        # group by same prune_algo and prune_rate, count occurance, and calculate mean and std, and add count, mean std columns to df
        df = df.groupby(['prune_algo', 'prune_rate']).agg({'ApproximateDiameter': ['count', 'mean', 'std']})
        # flatten multi-index columns, and merge multiple hearders
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
        # output to file
        df.to_csv(outfile, index=False, sep=',')

def SPSP_Eccentricity(dataset_name):
    infile = osp.join(PROJECT_HOME, f"output_metric_raw/{dataset_name}/SPSP_Eccentricity/log")
    outfile = osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/SPSP_Eccentricity/log")
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    if osp.exists(infile):
        fin = open(infile, 'r')
        if not fin.read(1):
            fin.close()
            return
        fin.seek(0)
        df = pd.DataFrame(columns=['prune_algo', 'num_nodes', 'num_edges', 'Distance', 'Unreachable', 'Eccentricity', 'Isolated'])
        lines = fin.readlines()
        l = 0
        while l < len(lines):
            line1, line2 = lines[l], lines[l+1]
            l += 2
            if line1 and line2:
                line1 = line1.strip().split()
                line2 = line2.strip().split()
                df = pd.concat([df, pd.DataFrame([[line1[0], int(line1[5]), int(line1[8]), 
                                                float("nan") if line1[29]=="--" else float(line1[29]), 
                                                float(line1[34]),
                                                float("nan") if line2[29]=="--" else float(line2[29]),
                                                float(line2[34])]],
                                                columns=['prune_algo', 'num_nodes', 'num_edges', 'Distance', 'Unreachable', 'Eccentricity', 'Isolated'])], 
                                                ignore_index=True)
        fin.close()
        original_edge = int(df[df['prune_algo'] == 'original']['num_edges'].values[0])
        # add prune_rate column
        df['prune_rate'] = (1-df['num_edges'].astype(int)/original_edge).round(2)
        # for prune_algo == ForestFire, round prune_rate to closest 0.1 if prune rate is smaller than 0.93
        # df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'] = df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'].round(1)
        # eliminate outliers
        # df = df.groupby(['prune_algo', 'prune_rate']).apply(lambda x: x[(x.ApproximateDiameter-x.ApproximateDiameter.mean()).abs() < 2*x.ApproximateDiameter.std()])
        # df = df.reset_index(drop=True)
        # group by same prune_algo and prune_rate, count occurance, and calculate mean and std, and add count, mean std columns to df
        df = df.groupby(['prune_algo', 'prune_rate']).agg({'Distance': ['count', 'mean', 'std'], 'Unreachable': ['mean', 'std'], 'Eccentricity': ['mean', 'std'], 'Isolated': ['mean', 'std']})
        # flatten multi-index columns, and merge multiple hearders
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
        # output to file
        df.to_csv(outfile, index=False, sep=',')

def GlobalClusteringCoefficient(dataset_name):
    infile = osp.join(PROJECT_HOME, f"output_metric_raw/{dataset_name}/GlobalClusteringCoefficient/log")
    outfile = osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/GlobalClusteringCoefficient/log")
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    if osp.exists(infile):
        fin = open(infile, 'r')
        if not fin.read(1):
            fin.close()
            return
        fin.seek(0)
        df = pd.DataFrame(columns=['prune_algo', 'num_nodes', 'num_edges', 'GlobalClusteringCoefficient'])
        for line in fin:
            if line:
                line = line.strip().split()
                df = pd.concat([df, pd.DataFrame([[line[0], int(line[5]), int(line[8]), float(line[17])]], 
                                                columns=['prune_algo', 'num_nodes', 'num_edges', 'GlobalClusteringCoefficient'])], 
                                                ignore_index=True)
        fin.close()
        original_edge = int(df[df['prune_algo'] == 'original']['num_edges'].values[0])
        # add prune_rate column
        df['prune_rate'] = (1-df['num_edges'].astype(int)/original_edge).round(2)
        # for prune_algo == ForestFire, round prune_rate to closest 0.1 if prune rate is smaller than 0.93
        # df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'] = df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'].round(1)
        # group by same prune_algo and prune_rate, count occurance, and calculate mean and std, and add count, mean std columns to df
        df = df.groupby(['prune_algo', 'prune_rate']).agg({'GlobalClusteringCoefficient': ['count', 'mean', 'std']})
        # flatten multi-index columns, and merge multiple hearders
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
        # output to file
        df.to_csv(outfile, index=False, sep=',')
    
def LocalClusteringCoefficient(dataset_name):
    infile = osp.join(PROJECT_HOME, f"output_metric_raw/{dataset_name}/LocalClusteringCoefficient/log")
    outfile = osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/LocalClusteringCoefficient/log")
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    if osp.exists(infile):
        fin = open(infile, 'r')
        if not fin.read(1):
            fin.close()
            return
        fin.seek(0)
        df = pd.DataFrame(columns=['prune_algo', 'num_nodes', 'num_edges', 'MeanClusteringCoefficient'])
        for line in fin:
            if line:
                line = line.strip().split()
                df = pd.concat([df, pd.DataFrame([[line[0], int(line[5]), int(line[8]), float(line[35])]], 
                                                columns=['prune_algo', 'num_nodes', 'num_edges', 'MeanClusteringCoefficient'])], 
                                                ignore_index=True)
        fin.close()
        original_edge = int(df[df['prune_algo'] == 'original']['num_edges'].values[0])
        # add prune_rate column
        df['prune_rate'] = (1-df['num_edges'].astype(int)/original_edge).round(2)
        # for prune_algo == ForestFire, round prune_rate to closest 0.1 if prune rate is smaller than 0.93
        # df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'] = df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'].round(1)
        # group by same prune_algo and prune_rate, count occurance, and calculate mean and std, and add count, mean std columns to df
        df = df.groupby(['prune_algo', 'prune_rate']).agg({'MeanClusteringCoefficient': ['count', 'mean', 'std']})
        # flatten multi-index columns, and merge multiple hearders
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
        # output to file
        df.to_csv(outfile, index=False, sep=',')

def MaxFlow(dataset_name):
    infile = osp.join(PROJECT_HOME, f"output_metric_raw/{dataset_name}/MaxFlow/log")
    outfile = osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/MaxFlow/log")
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    if osp.exists(infile):
        fin = open(infile, 'r')
        if not fin.read(1):
            fin.close()
            return
        fin.seek(0)
        df = pd.DataFrame(columns=['prune_algo', 'num_nodes', 'num_edges', 'MaxFlow', 'Unreachable'])
        for line in fin:
            if line:
                line = line.strip().split()
                df = pd.concat([df, pd.DataFrame([[line[0], int(line[2]), int(line[4]), float('nan') if line[17]=='--' else float(line[17]), float('nan') if line[21]=='--' else float(line[21])]], 
                                                columns=['prune_algo', 'num_nodes', 'num_edges', 'MaxFlow', 'Unreachable'])], 
                                                ignore_index=True)
        fin.close()
        original_edge = int(df[df['prune_algo'] == 'original']['num_edges'].values[0])
        # add prune_rate column
        df['prune_rate'] = (1-df['num_edges'].astype(int)/original_edge).round(2)
        # for prune_algo == ForestFire, round prune_rate to closest 0.1 if prune rate is smaller than 0.93
        # df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'] = df.loc[(df['prune_algo'] == 'ForestFire') & (df['prune_rate'] < 0.93), 'prune_rate'].round(1)
        # group by same prune_algo and prune_rate, count occurance, and calculate mean and std, and add count, mean std columns to df
        df = df.groupby(['prune_algo', 'prune_rate']).agg({'MaxFlow': ['count', 'mean', 'std'], 'Unreachable': ['mean', 'std']})
        # flatten multi-index columns, and merge multiple hearders
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        df = df.reset_index()
        # output to file
        df.to_csv(outfile, index=False, sep=',')


def SAGE(dataset_name):
    if dataset_name != "ogbn-proteins":
        return
    outfile = osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/SAGE/log")
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    fout = open(outfile, 'w')
    fout.write("prune_algo,prune_rate,test_acc\n")
    for prune_algo in ["empty", "original"]:
        folder = osp.join(PROJECT_HOME, f"experiments/SAGE/{dataset_name}/{prune_algo}")
        if not osp.exists(folder):
            continue
        if osp.exists(osp.join(folder, "test.csv")):
            with open(osp.join(folder, "test.csv"), 'r') as fin:
                lines = fin.readlines()
                top_10 = [0] * 10
                for i in range(1, len(lines)):
                    line = lines[i].strip().split(',')
                    if float(line[-1]) > min(top_10):
                        top_10[top_10.index(min(top_10))] = float(line[-1])
            prune_rate = 0
            fout.write(f"{prune_algo},{prune_rate},{sum(top_10)/10}\n")
    for prune_algo in ["ER", "ForestFire", "GSpar", "KNeighbor", "LocalDegree", "LocalSimilarity", "LSpar", "Random", "RankDegree", "SCAN", "SpanningForest", "Spanner-3", "Spanner-5", "Spanner-7"]:
        folder = osp.join(PROJECT_HOME, f"experiments/SAGE/{dataset_name}/{prune_algo}")
        if not osp.exists(folder):
            continue
        for subfolder in os.listdir(folder):
            if osp.exists(osp.join(folder, subfolder, "test.csv")):
                with open(osp.join(folder, subfolder, "test.csv"), 'r') as fin:
                    lines = fin.readlines()
                    top_10 = [0] * 10
                    for i in range(1, len(lines)):
                        line = lines[i].strip().split(',')
                        if float(line[-1]) > min(top_10):
                            top_10[top_10.index(min(top_10))] = float(line[-1])
                prune_rate = float(subfolder)
                fout.write(f"{prune_algo},{prune_rate},{sum(top_10)/10}\n")
    fout.close()

def ClusterGCN(dataset_name):
    if dataset_name != "Reddit":
        return
    outfile = osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/ClusterGCN/log")
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    fout = open(outfile, 'w')
    fout.write("prune_algo,prune_rate,test_acc\n")
    for prune_algo in ["empty", "original"]:
        folder = osp.join(PROJECT_HOME, f"experiments/ClusterGCN/{dataset_name}/{prune_algo}")
        if not osp.exists(folder):
            continue
        if osp.exists(osp.join(folder, "test.csv")):
            with open(osp.join(folder, "test.csv"), 'r') as fin:
                lines = fin.readlines()
                top_10 = [0] * 10
                for i in range(1, len(lines)):
                    line = lines[i].strip().split(',')
                    if float(line[-1]) > min(top_10):
                        top_10[top_10.index(min(top_10))] = float(line[-1])
            prune_rate = 0
            fout.write(f"{prune_algo},{prune_rate},{sum(top_10)/10}\n")
    for prune_algo in ["ER", "ForestFire", "GSpar", "KNeighbor", "LocalDegree", "LocalSimilarity", "LSpar", "Random", "RankDegree", "SCAN", "SpanningForest", "Spanner-3", "Spanner-5", "Spanner-7"]:
        folder = osp.join(PROJECT_HOME, f"experiments/ClusterGCN/{dataset_name}/{prune_algo}")
        if not osp.exists(folder):
            continue
        for subfolder in os.listdir(folder):
            if osp.exists(osp.join(folder, subfolder, "test.csv")):
                with open(osp.join(folder, subfolder, "test.csv"), 'r') as fin:
                    lines = fin.readlines()
                    top_10 = [0] * 10
                    for i in range(1, len(lines)):
                        line = lines[i].strip().split(',')
                        if float(line[-1]) > min(top_10):
                            top_10[top_10.index(min(top_10))] = float(line[-1])
                prune_rate = float(subfolder)
                fout.write(f"{prune_algo},{prune_rate},{sum(top_10)/10}\n")
    fout.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True)
    args = parser.parse_args()

    if args.dataset_name == "all":
        for dataset_name in ["ego-Facebook",
                             "ego-Twitter", 
                             "human_gene2", 
                             "com-DBLP", 
                             "com-Amazon", 
                             "email-Enron", 
                             "ca-AstroPh", 
                             "ca-HepPh", 
                             "web-BerkStan", 
                             "web-Google", 
                             "web-NotreDame", 
                             "web-Stanford", 
                             "Reddit", 
                             "ogbn-proteins"]:
            degreeDistribution(dataset_name)
            Centrality(dataset_name)
            DetectCommunity(dataset_name)
            ClusteringF1Similarity(dataset_name)
            QuadraticFormSimilarity(dataset_name)
            ApproximateDiameter(dataset_name)
            SPSP_Eccentricity(dataset_name)
            GlobalClusteringCoefficient(dataset_name)
            LocalClusteringCoefficient(dataset_name)
            MaxFlow(dataset_name)
            SAGE(dataset_name)
            ClusterGCN(dataset_name)
    else:
        degreeDistribution(args.dataset_name)
        Centrality(args.dataset_name)
        DetectCommunity(args.dataset_name)
        ClusteringF1Similarity(args.dataset_name)
        QuadraticFormSimilarity(args.dataset_name)
        ApproximateDiameter(args.dataset_name)
        SPSP_Eccentricity(args.dataset_name)
        GlobalClusteringCoefficient(args.dataset_name)
        LocalClusteringCoefficient(args.dataset_name)
        MaxFlow(args.dataset_name)
        SAGE(args.dataset_name)
        ClusterGCN(args.dataset_name)
    