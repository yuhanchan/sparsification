from matplotlib import pyplot as plt
import argparse
import os
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.lines as mlines
import json
import numpy as np
import matplotlib.patches as mpatches
import math

PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ")
    print("please source env.sh at the top level of the project")
    exit(1)

color_map = {
    "RankDegree": "#6baed6",
    # "ER-Min": "#0570b0",
    # "ER-Min_unweighted": "#0570b0",
    # "ER-Min_weighted": "#0570b0",
    "ER": "#0570b0",
    "ER-Max": "#0570b0",
    "ER-Max_unweighted": "#0570b0",
    "ER-Max_weighted": "#addd8e",
    "ER-unweighted": "#0570b0",
    "ER-weighted": "#addd8e",
    "ForestFire": "#238443",
    "GSpar": "#e31a1c",
    "KNeighbor": "#ff7f00",
    "LocalSimilarity": "#6a3d9a",
    "LocalDegree": "#b15928",
    "LSpar": "#6baed6",
    "Random": "#0570b0",
    "SCAN": "#addd8e",
    "Spanner-3": "#238443",
    "Spanner-5": "#e31a1c",
    "Spanner-7": "#ff7f00",
    "SpanningForest": "#6a3d9a",

    "RD": "#6baed6",
    "FF": "#238443",
    "GS": "#e31a1c",
    "KN": "#ff7f00",
    "LSim": "#6a3d9a",
    "LD": "#b15928",
    "LS": "#6baed6",
    "RN": "#0570b0",
    "SCAN": "#addd8e",
    "SP-3": "#238443",
    "SP-5": "#e31a1c",
    "SP-7": "#ff7f00",
    "SF": "#6a3d9a",
    "ER-uw": "#0570b0",
    "ER-w": "#addd8e",
}

marker_map = {
    "RankDegree": "o",
    # "ER-Min": "o",
    # "ER-Min_unweighted": "X",
    # "ER-Min_weighted": "o",
    "ER": "o",
    "ER-Max": "o",
    "ER-Max_unweighted": "o",
    "ER-Max_weighted": "o",
    "ER-unweighted": "o",
    "ER-weighted": "o",
    "ForestFire": "o",
    "GSpar": "o",
    "KNeighbor": "o",
    "LocalSimilarity": "o",
    "LocalDegree": "o",
    "LSpar": "^",
    "Random": "^",
    "SCAN": "^",
    "Spanner-3": "^",
    "Spanner-5": "^",
    "Spanner-7": "^",
    "SpanningForest": "^",
    "RD": "o",
    "FF": "o",
    "GS": "o",
    "KN": "o",
    "LSim": "o",
    "LD": "o",
    "LS": "^",
    "RN": "^",
    "SCAN": "^",
    "SP-3": "^",
    "SP-5": "^",
    "SP-7": "^",
    "SF": "^",
    "ER-uw": "o",
    "ER-w": "o",
}

text_map = {
    "RankDegree": "RD",
    "ER": "ER",
    "ER-Max": "ER",
    "ER-Max_unweighted": "ER-uw",
    "ER-Max_weighted": "ER-w",
    "ER-unweighted": "ER-uw",
    "ER-weighted": "ER-w",
    "ForestFire": "FF",
    "GSpar": "GS",
    "KNeighbor": "KN",
    "LocalSimilarity": "LSim",
    "LocalDegree": "LD",
    "LSpar": "LS",
    "Random": "RN",
    "SCAN": "SCAN",
    "Spanner-3": "SP-3",
    "Spanner-5": "SP-5",
    "Spanner-7": "SP-7",
    "SpanningForest": "SF",
}

hatch_map = {
    "RankDegree": "/",
    # "ER-Min": "o",
    # "ER-Min_unweighted": "X",
    # "ER-Min_weighted": "o",
    "ER": "/",
    "ER-Max": "",
    "ER-Max_unweighted": "",
    "ER-Max_weighted": "",
    "ER-unweighted": "",
    "ER-weighted": "",
    "ForestFire": "",
    "GSpar": "",
    "KNeighbor": "",
    "LocalSimilarity": "",
    "LocalDegree": "",
    "LSpar": "",
    "Random": "",
    "SCAN": "",
    "Spanner-3": "",
    "Spanner-5": "",
    "Spanner-7": "",
    "SpanningForest": "",
    "RD": "/",
    "FF": "",
    "GS": "",
    "KN": "",
    "LSim": "",
    "LD": "",
    "LS": "",
    "RN": "",
    "SCAN": "",
    "SP-3": "",
    "SP-5": "",
    "SP-7": "",
    "SF": "",
    "ER-uw": "",
    "ER-w": "",
}



def sparsifier_time(dataset_name):
    config = json.load(open(osp.join(PROJECT_HOME, "config.json"), "r"))
    original_edges = config[dataset_name]["num_edges"]
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_sparsifier_parsed/{dataset_name}.csv"), header=0, sep=",")
    except:
        return
    
    df["prune_rate"] = 1 - df["num_edge"] / original_edges

    # remove prune rate < 0
    df = df[df.prune_rate >= 0]

    prune_algo_map = {"randomEdgeSparsifier": "RN",
                      "localDegreeSparsifier": "LD",
                      "fireforestSparsifier": "FF",
                      "forestfireSparsifier": "FF",
                      "localSimilaritySparsifier": "LSim",
                      "GSpar": "GS",
                      "scanSparsifier": "SCAN",
                      "KNeighbor": "KN",
                      "LSpar": "LS",
                      "RankDegree": "RD",
                      "python_er_sparsify": "ER",
                      }

    # # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in prune_algo_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    df = df.sort_values(by=["prune_rate"])
    grouped = df.groupby('prune_algo')

    # plot
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="wall_time", ax=ax, marker=marker_map[key], label=key, 
                                    color=color_map[key], markersize=markersize, linewidth=linewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Time (s)", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    # plt.yscale("log")
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    # for key in ["Random", "KNeighbor", "RankDegree", "LocalDegree", "SpanningForest", "Spanner-3", "Spanner-5", 
    #             "Spanner-7", "ForestFire", "LSpar", "GSpar", "LocalSimilarity", "SCAN", "ER"]:
    for key in ["Random", "KNeighbor", "LSpar", "GSpar", "LocalDegree", "SCAN", "LocalSimilarity", "ForestFire", "RankDegree", "ER"]:
        handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                linewidth=linewidth, label=text_map[key])
        handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=legendfontsize)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_time.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

    ### plot bar chart with all sparsifier with prune rate 0.1, 0.5, 0.9
    prune_rate_list = [0.1, 0.5, 0.9]
    # filter df by prune rate, consider floating point precision
    df["prune_rate"] = df["prune_rate"].apply(lambda x: round(x, 1))
    df = df[df["prune_rate"].isin(prune_rate_list)]

    fig, ax = plt.subplots(figsize=(figwidth+2, figheight))

    # shift right for each bar
    width = 0.1
    ind = [0.05, 1.55, 3.05]
    ind = np.array(ind)
    for prune_algo in ["RN", "KN", "LS", "GS", "LD", "SCAN", "LSim", "FF", "RD", "ER"]:
        df_prune_algo = df[df["prune_algo"] == prune_algo]
        # if there are more than 1 row with prune rate 0.9, keep the one with higher num_edge
        df_prune_algo = df_prune_algo.sort_values(by=["prune_rate", "num_edge"], ascending=[True, False])
        df_prune_algo = df_prune_algo.drop_duplicates(subset=["prune_rate"], keep="first")
        ax.bar(ind, df_prune_algo["wall_time"], width=width, label=prune_algo, color=color_map[prune_algo], hatch=hatch_map[prune_algo])
        ind = ind + width
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Time (s)", fontsize=ylabelfontsize)
    ax.set_xticks([0.5, 2, 3.5])
    ax.set_xticklabels(["0.1", "0.5", "0.9"], fontsize=xtickfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.yscale("log")
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    for key in ["Random", "KNeighbor", "LSpar", "GSpar", "LocalDegree", "SCAN", "LocalSimilarity", "ForestFire", "RankDegree", "ER"]:
        handle = mpatches.Patch(facecolor=color_map[key], label=text_map[key], hatch=hatch_map[key])
        handles.append(handle)
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
    # plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=legendfontsize)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_time_bar.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

def degreeDistribution(dataset_name, prune_algos=None):
    # read csv file, seprated by , and space
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/degreeDistribution/log"), header=0, sep=",")
    except:
        return

    # remove some algorithms
    df = df[df.prune_algo.isin(prune_algos)]
    
    # remove prune rate < 0
    df = df[df.prune_rate >= 0]
    df = df[df.prune_rate < 0.93]
    df = df[~((df.prune_algo == "ForestFire") & (df.prune_rate > 0.75))]
    df = df[~((df.prune_algo == "KNeighbor") & (df.prune_rate > 0.92))]

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in text_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    df.loc[df.prune_algo == "ER-Max_unweighted", "prune_algo"] = "ER"

    grouped = df.groupby('prune_algo')

    # plot
    fig, ax = plt.subplots(figsize=(figwidth, figheight+1.5))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="Bhattacharyya_distance_mean", yerr="Bhattacharyya_distance_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Bhattacharyya Distance", fontsize=ylabelfontsize-5)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    if len(prune_algos) > 3:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=text_map[key])
            handles.append(handle)
    else:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=key)
            handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=4, fontsize=legendfontsize)
    plt.tight_layout()

    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_degreeDistribution.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

def Diameter(dataset_name, prune_algos=None):
    # read csv file, seprated by , and space
    for algo in ["Diameter", "ApproximateDiameter"]:
        try:
            df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/{algo}/log"), header=0, sep=",")
        except:
            continue

        # get ground truth
        ground_truth = df[df.prune_algo == "original"][f"{algo}_mean"].values[0]
        # remove some algorithms
        df = df[df.prune_algo.isin(prune_algos)]

        # remove prune rate < 0
        df = df[df.prune_rate >= 0]
        df = df[df.prune_rate < 0.93]

        # rename er_min_weighted and er_max_weighted to er_min and er_max
        for key, value in text_map.items():
            df.loc[df.prune_algo == key, "prune_algo"] = value

        grouped = df.groupby('prune_algo')

        # plot
        fig, ax = plt.subplots(figsize=(figwidth+figwidth_legend_compensation, figheight))
        for key in grouped.groups.keys():
            grouped.get_group(key).plot(x="prune_rate", y=f"{algo}_mean", yerr=f"{algo}_std", 
                                        ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                        linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
        # plot ground truth line
        plt.axhline(y=ground_truth, color='g', linestyle='--', linewidth=linewidth)
        plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
        plt.ylabel("Diameter", fontsize=ylabelfontsize)
        plt.xticks(fontsize=xtickfontsize)
        plt.yticks(fontsize=ytickfontsize)
        plt.grid(gridon)

        # make legend without error bar
        # handles = []
        # if len(prune_algos) > 3:
        #     for key in prune_algos:
        #         handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
        #                                 linewidth=linewidth, label=text_map[key])
        #         handles.append(handle)
        # else:
        #     for key in prune_algos:
        #         handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
        #                                 linewidth=linewidth, label=key)
        #         handles.append(handle)
        # # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
        # plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=4, fontsize=legendfontsize-3)
        # plt.tight_layout()
        handles = []
        for key in ["RN", "KN", "RD", "LD", "SF", "SP-3", "SP-5", "SP-7", "FF", "LS", "GS", "LSim", "SCAN", "ER-uw"]:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize-5, 
                                    linewidth=linewidth, label=key)
            handles.append(handle)
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize-8)
        # plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=legendfontsize)
        plt.tight_layout()

        # save plot
        figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_Diameter.{saveformat}")
        os.makedirs(osp.dirname(figpath), exist_ok=True)
        plt.savefig(figpath)

def SPSP_Eccentricity(dataset_name, prune_algos=None):
    # read csv file, seprated by , and space
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/SPSP_Eccentricity/log"), header=0, sep=",")
    except:
        return
    original_unreachable = df[df.prune_algo == "original"]["Unreachable_mean"].values[0]
    original_isolated = df[df.prune_algo == "original"]["Isolated_mean"].values[0]

    df = df[~df.prune_algo.isin(["original", "ER-Min", "ER-Min_weighted", "ER-Min_unweighted", "ER-Max_weighted"])]

    # remove prune rate < 0
    df = df[df.prune_rate >= 0]
    df = df[df.prune_rate < 0.93]

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in text_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    grouped = df.groupby('prune_algo')

    ### plot SPSP unreachable ratio
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="Unreachable_mean", yerr="Unreachable_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    ax.fill_between(np.arange(0, 1.01, 0.1), original_unreachable+0.2, 1, facecolor='gray', alpha=0.3)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Unreachable Ratio", fontsize=ylabelfontsize-2)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    for key in ["RN", "KN", "RD", "LD", "SF", "SP-3", "SP-5", "SP-7", "FF", "LS", "GS", "LSim", "SCAN", "ER"]:
        handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize-5, 
                                linewidth=linewidth, label=key)
        handles.append(handle)
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize-8)
    # plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=legendfontsize)
    plt.legend().set_visible(False)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_SPSP_unreachable.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

    ### plot SPSP stretch factor
    fig, ax = plt.subplots(figsize=(figwidth-0.5, figheight))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="Distance_mean", yerr="Distance_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Stretch Factor", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    plt.legend().set_visible(False)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_SPSP_stretch_factor.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)


    ### plot SPSP stretch factor with unreachable ratio < original unreachable ratio + 0.2
    # plot_legend = True
    # fig, ax = plt.subplots(figsize=(figwidth+figwidth_legend_compensation+0.5, figheight))
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    filtered = df[df.Unreachable_mean < original_unreachable+0.2]
    filtered_grouped = filtered.groupby('prune_algo')
    for key in filtered_grouped.groups.keys():
        filtered_grouped.get_group(key).plot(x="prune_rate", y="Distance_mean", yerr="Distance_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Stretch Factor", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar, include all prune_algo, even filtered out
    # handles = []
    # for key in ["RN", "KN", "RD", "LD", "SF", "SP-3", "SP-5", "SP-7", "FF", "LS", "GS", "LSim", "SCAN", "ER-uw"]:
    #     handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize-5, 
    #                             linewidth=linewidth, label=key)
    #     handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize-5)
    # # plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=legendfontsize)
    plt.legend().set_visible(False)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_SPSP_stretch_factor_with_unreachable_constraint.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

    ### plot Eccentricity isolated ratio
    fig, ax = plt.subplots(figsize=(figwidth+figwidth_legend_compensation, figheight))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="Isolated_mean", yerr="Isolated_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    ax.fill_between(np.arange(0, 1.01, 0.1), original_isolated+0.2, 1, facecolor='gray', alpha=0.3)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Isolated Ratio", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    for key in ["RN", "KN", "RD", "LD", "SF", "SP-3", "SP-5", "SP-7", "FF", "LS", "GS", "LSim", "SCAN", "ER"]:
        handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize-5, 
                                linewidth=linewidth, label=key)
        handles.append(handle)
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize-8)
    # plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=legendfontsize)
    # plt.legend().set_visible(False)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_Eccentricity_isolated.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

    ### plot Eccentricity stretch factor
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="Eccentricity_mean", yerr="Eccentricity_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Stretch Factor", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    plt.legend().set_visible(False)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_Eccentricity_stretch_factor.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

    ### plot Eccentricity stretch factor with unreachable ratio < original unreachable ratio + 0.2
    # fig, ax = plt.subplots(figsize=(figwidth+figwidth_legend_compensation, figheight))
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    filtered = df[df.Unreachable_mean < original_unreachable+0.2]
    filtered_grouped = filtered.groupby('prune_algo')
    for key in filtered_grouped.groups.keys():
        filtered_grouped.get_group(key).plot(x="prune_rate", y="Eccentricity_mean", yerr="Eccentricity_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)

    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Stretch Factor", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    # handles = []
    # for key in ["RN", "KN", "RD", "LD", "SF", "SP-3", "SP-5", "SP-7", "FF", "LS", "GS", "LSim", "SCAN", "ER-uw"]:
    #     handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize-5, 
    #                             linewidth=linewidth, label=key)
    #     handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize-5)
    # # plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=legendfontsize)
    plt.legend().set_visible(False)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_Eccentricity_stretch_factor_with_unreachable_constraint.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

def LocalClusteringCoefficient(dataset_name, prune_algos=None):
    # read csv file, seprated by , and space
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/LocalClusteringCoefficient/log"), header=0, sep=",")
    except:
        return
    mcc_ground_truth = df[df.prune_algo == "original"]["MeanClusteringCoefficient_mean"].values[0]
    df = df[df.prune_algo.isin(prune_algos)]

    # remove prune rate < 0
    df = df[df.prune_rate >= 0]
    df = df[df.prune_rate < 0.93]

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in text_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    grouped = df.groupby('prune_algo')

    # plot
    fig, ax = plt.subplots(figsize=(figwidth, figheight+figheight_legend_compensation*2))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="MeanClusteringCoefficient_mean", yerr="MeanClusteringCoefficient_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    ax.axhline(y=mcc_ground_truth, color='g', linestyle='--', linewidth=linewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Mean Clustering Coeff", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    if len(prune_algos) > 3:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=text_map[key])
            handles.append(handle)
    else:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=key)
            handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=4, fontsize=legendfontsize-3)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_MeanClusteringCoefficient.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

def GlobalClusteringCoefficient(dataset_name, prune_algos=None):
    # read csv file, seprated by , and space
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/GlobalClusteringCoefficient/log"), header=0, sep=",")
    except:
        return
    gcc_ground_truth = df[df.prune_algo == "original"]["GlobalClusteringCoefficient_mean"].values[0]
    df = df[df.prune_algo.isin(prune_algos)]

    # remove prune rate < 0
    df = df[df.prune_rate >= 0]
    df = df[df.prune_rate < 0.93]

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in text_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    grouped = df.groupby('prune_algo')

    # plot
    fig, ax = plt.subplots(figsize=(figwidth, figheight+figheight_legend_compensation))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="GlobalClusteringCoefficient_mean", yerr="GlobalClusteringCoefficient_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    ax.axhline(y=gcc_ground_truth, color='g', linestyle='--', linewidth=linewidth)
    if addTitle:
        plt.title(f"Global Clustering Coefficient ({dataset_name})", fontsize=titlefontsize)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Global Clustering Coeff", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    if len(prune_algos) > 3:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=text_map[key])
            handles.append(handle)
    else:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=key)
            handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=4, fontsize=legendfontsize-3)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_GlobalClusteringCoefficient.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

def ClusteringF1Similarity(dataset_name, prune_algos=None):
    # read csv file, seprated by , and space
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/ClusteringF1Similarity/log"), header=0, sep=",")
    except:
        return
    ground_truth = df[df.prune_algo == "original"]["F1_Similarity_mean"].values[0]
    df = df[df.prune_algo.isin(prune_algos)]

    # remove prune rate < 0
    df = df[df.prune_rate >= 0]
    df = df[df.prune_rate < 0.93]

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in text_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    grouped = df.groupby('prune_algo')

    # plot
    fig, ax = plt.subplots(figsize=(figwidth, figheight+figheight_legend_compensation*2))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="F1_Similarity_mean", yerr="F1_Similarity_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    plt.axhline(y=ground_truth, color='g', linestyle='--', linewidth=linewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("F1 Similarity", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    if len(prune_algos) > 3:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=text_map[key])
            handles.append(handle)
    else:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=key)
            handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.43), ncol=4, fontsize=legendfontsize-3)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_ClusteringF1Similarity.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

def Centrality(dataset_name, algo, prune_algos=None):
    # read csv file, seprated by , and space
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/{algo}Centrality/log"), header=0, sep=",")
    except:
        return

    # remove some algorithms
    df = df[df.prune_algo.isin(prune_algos)]

    # remove prune rate < 0
    df = df[df.prune_rate >= 0]
    df = df[df.prune_rate < 0.93]

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in text_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    grouped = df.groupby('prune_algo')

    ### precision plot
    fig, ax = plt.subplots(figsize=(figwidth, figheight+figheight_legend_compensation))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="top_100_precision_mean", yerr="top_100_precision_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Precision", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    if len(prune_algos) > 3:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=text_map[key])
            handles.append(handle)
    else:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=key)
            handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
    # if algo == "Katz":
    #     plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=legendfontsize-3)
    # else:
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=4, fontsize=legendfontsize-3)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_{algo}Centrality_precision.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

def DetectCommunity(dataset_name, prune_algos=None):
    # read csv file, seprated by , and space
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/DetectCommunity/log"), header=0, sep=",")
    except:
        return
    num_community_ground_truth = df[df.prune_algo == "original"]["num_community_mean"].values[0]
    modularity_ground_truth = df[df.prune_algo == "original"]["modularity_mean"].values[0]
    df = df[df.prune_algo.isin(prune_algos)]

    # remove prune rate < 0
    df = df[df.prune_rate >= 0]
    df = df[df.prune_rate < 0.93]

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in text_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    grouped = df.groupby('prune_algo')

    ### plot number of community
    fig, ax = plt.subplots(figsize=(figwidth, figheight+figheight_legend_compensation+1))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="num_community_mean", yerr="num_community_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    plt.axhline(y=num_community_ground_truth, color='g', linestyle='--', linewidth=linewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize-5)
    plt.ylabel("Number of Communities", fontsize=ylabelfontsize-5)
    plt.yscale("log")
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    if len(prune_algos) > 3:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=text_map[key])
            handles.append(handle)
    else:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=key)
            handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.45), ncol=4, fontsize=legendfontsize-3)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_num_community.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

def QuadraticFormSimilarity(dataset_name, prune_algos=None):
    # read csv file, seprated by , and space
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/QuadraticFormSimilarity/log"), header=0, sep=",")
    except:
        return
    df = df[df.prune_algo.isin(prune_algos)]

    # remove prune rate < 0
    df = df[df.prune_rate >= 0]
    df = df[df.prune_rate < 0.93]

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in text_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    grouped = df.groupby('prune_algo')

    # plot
    fig, ax = plt.subplots(figsize=(figwidth, figheight+figheight_legend_compensation))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="QuadraticFormSimilarity_mean", yerr="QuadraticFormSimilarity_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Quadratic Form Similarity", fontsize=ylabelfontsize-5)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # replace "ER-Max_weighted" with "ER-weighted" in prune_algos
    if "ER-Max_weighted" in prune_algos:
        prune_algos[prune_algos.index("ER-Max_weighted")] = "ER-weighted"
    # make legend without error bar
    handles = []
    if len(prune_algos) > 3:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=text_map[key])
            handles.append(handle)
    else:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=key)
            handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=legendfontsize)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_QuadraticFormSimilarity.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

def MaxFlow(dataset_name, prune_algos=None):
    # read csv file, seprated by , and space
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/MaxFlow/log"), header=0, sep=",")
    except:
        return
    original_unreachable = df[df.prune_algo == "original"]["Unreachable_mean"].values[0]
    df = df[df.prune_algo.isin(prune_algos)]

    # remove prune rate < 0
    df = df[df.prune_rate >= 0]
    df = df[df.prune_rate < 0.93]

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in text_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    grouped = df.groupby('prune_algo')

    ### plot MaxFlow stretch factor
    fig, ax = plt.subplots(figsize=(figwidth, figheight+figheight_legend_compensation))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="MaxFlow_mean", yerr="MaxFlow_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Mean Stretch Factor", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    if len(prune_algos) > 3:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=text_map[key])
            handles.append(handle)
    else:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=key)
            handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=4, fontsize=legendfontsize-5)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_MaxFlow_stretch_factor.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

    ### plot MaxFlow unreachable ratio
    fig, ax = plt.subplots(figsize=(figwidth, figheight+figheight_legend_compensation))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="Unreachable_mean", yerr="Unreachable_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    ax.fill_between(np.arange(0, 1.01, 0.1), original_unreachable+0.2, 1, facecolor='gray', alpha=0.3)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Unreachable Ratio", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    if len(prune_algos) > 3:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=text_map[key])
            handles.append(handle)
    else:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=key)
            handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=4, fontsize=legendfontsize-5)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_MaxFLow_unreachable.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

    ### plot MaxFlow stretch factor with constraint
    fig, ax = plt.subplots(figsize=(figwidth, figheight+figheight_legend_compensation))
    df = df[df.Unreachable_mean < original_unreachable+0.2]
    grouped = df.groupby('prune_algo')
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="MaxFlow_mean", yerr="MaxFlow_std", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], markersize=markersize, 
                                    linewidth=linewidth, capsize=capsize, capthick=capthick, elinewidth=elinewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Mean Stretch Factor", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    handles = []
    if len(prune_algos) > 3:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=text_map[key])
            handles.append(handle)
    else:
        for key in prune_algos:
            handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                    linewidth=linewidth, label=key)
            handles.append(handle)
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=4, fontsize=legendfontsize-5)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_MaxFlow_stretch_factor_with_constraint.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

def GCN(dataset_name, prune_algos=None):
    # read csv file, seprated by , and space
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/GCN/log"), header=0, sep=",")
    except:
        return
    full_acc = df[df.prune_algo == "original"]["test_acc"].values[0]
    empty_acc = df[df.prune_algo == "empty"]["test_acc"].values[0]
    if prune_algos is None:
        df = df[~df.prune_algo.isin(["original", "empty", "ER-Min", "ER-Min_weighted", "ER-Min_unweighted", "ER-Max_weighted"])]
    else:
        df = df[df.prune_algo.isin(prune_algos)]

    # remove prune rate < 0
    df = df[df.prune_rate >= 0]
    df = df[df.prune_rate < 0.93]

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in text_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    # sort df by prune rate
    df = df.sort_values(by=['prune_rate'])
    grouped = df.groupby('prune_algo')

    ### plot MaxFlow stretch factor
    plot_legend = True
    if plot_legend:
        fig, ax = plt.subplots(figsize=(figwidth+figwidth_legend_compensation, figheight+figheight_legend_compensation))
    else:
        fig, ax = plt.subplots(figsize=(figwidth, figheight))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="test_acc", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], 
                                    markersize=markersize, linewidth=linewidth)
    if addTitle:
        plt.title(f"GCN Test AUC-ROC ({dataset_name})", fontsize=titlefontsize)
    plt.axhline(y=full_acc, color='g', linestyle='--', linewidth=linewidth)
    plt.axhline(y=empty_acc, color='r', linestyle='--', linewidth=linewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("AUC-ROC", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    if plot_legend:
        handles = []
        if prune_algos is None:
            for key in ["Random", "KNeighbor", "RankDegree", "LocalDegree", "SpanningForest", "Spanner-3", "Spanner-5", 
                        "Spanner-7", "ForestFire", "LSpar", "GSpar", "LocalSimilarity", "SCAN", "ER-unweighted"]:
                handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                        linewidth=linewidth, label=text_map[key])
                handles.append(handle)
        else:
            for key in prune_algos:
                handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                        linewidth=linewidth, label=key)
                handles.append(handle)
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
        # plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=legendfontsize)
    else:
        plt.legend().set_visible(False)
    plt.tight_layout()

    # save plot
    if saveformat == "png":
        figpath = osp.join(PROJECT_HOME, f"output_metric_plot/{dataset_name}/GCN/{dataset_name}_GCN.{saveformat}")
    else:
        figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_GCN.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

def ClusterGCN(dataset_name, prune_algos=None):
    # read csv file, seprated by , and space
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/ClusterGCN/log"), header=0, sep=",")
    except:
        return
    full_acc = df[df.prune_algo == "original"]["test_acc"].values[0]
    empty_acc = df[df.prune_algo == "empty"]["test_acc"].values[0]
    if prune_algos is None:
        df = df[~df.prune_algo.isin(["original", "empty", "ER-Min", "ER-Min_weighted", "ER-Min_unweighted", "ER-Max_weighted"])]
    else:
        df = df[df.prune_algo.isin(prune_algos)]

    # remove prune rate < 0
    df = df[df.prune_rate >= 0]
    df = df[df.prune_rate < 0.93]

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in text_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    # sort df by prune rate
    df = df.sort_values(by=['prune_rate'])
    grouped = df.groupby('prune_algo')

    ### plot MaxFlow stretch factor
    plot_legend = True
    if plot_legend:
        fig, ax = plt.subplots(figsize=(figwidth, figheight+figheight_legend_compensation))
    else:
        fig, ax = plt.subplots(figsize=(figwidth, figheight))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="test_acc", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], 
                                    markersize=markersize, linewidth=linewidth)
    plt.axhline(y=full_acc, color='g', linestyle='--', linewidth=linewidth)
    plt.axhline(y=empty_acc, color='r', linestyle='--', linewidth=linewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("Accuracy (%)", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    if plot_legend:
        handles = []
        if prune_algos is None:
            for key in ["Random", "KNeighbor", "RankDegree", "LocalDegree", "SpanningForest", "Spanner-3", "Spanner-5", 
                        "Spanner-7", "ForestFire", "LSpar", "GSpar", "LocalSimilarity", "SCAN", "ER-unweighted"]:
                handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                        linewidth=linewidth, label=text_map[key])
                handles.append(handle)
        else:
            for key in prune_algos:
                handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                        linewidth=linewidth, label=text_map[key])
                handles.append(handle)
        # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
        plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=4, fontsize=legendfontsize-3)
    else:
        plt.legend().set_visible(False)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_ClusterGCN.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)

def SAGE(dataset_name, prune_algos=None):
    # read csv file, seprated by , and space
    try:
        df = pd.read_csv(osp.join(PROJECT_HOME, f"output_metric_parsed/{dataset_name}/SAGE/log"), header=0, sep=",")
    except:
        return
    full_acc = df[df.prune_algo == "original"]["test_acc"].values[0]
    empty_acc = df[df.prune_algo == "empty"]["test_acc"].values[0]
    if prune_algos is None:
        df = df[~df.prune_algo.isin(["original", "empty", "ER-Min", "ER-Min_weighted", "ER-Min_unweighted", "ER-Max_weighted"])]
    else:
        df = df[df.prune_algo.isin(prune_algos)]

    # remove prune rate < 0
    df = df[df.prune_rate >= 0]
    df = df[df.prune_rate < 0.93]

    # rename er_min_weighted and er_max_weighted to er_min and er_max
    for key, value in text_map.items():
        df.loc[df.prune_algo == key, "prune_algo"] = value

    # sort df by prune rate
    df = df.sort_values(by=['prune_rate'])
    grouped = df.groupby('prune_algo')

    ### plot MaxFlow stretch factor
    plot_legend = True
    if plot_legend:
        fig, ax = plt.subplots(figsize=(figwidth, figheight+figheight_legend_compensation))
    else:
        fig, ax = plt.subplots(figsize=(figwidth, figheight))
    for key in grouped.groups.keys():
        grouped.get_group(key).plot(x="prune_rate", y="test_acc", 
                                    ax=ax, marker=marker_map[key], label=key, color=color_map[key], 
                                    markersize=markersize, linewidth=linewidth)
    plt.axhline(y=full_acc, color='g', linestyle='--', linewidth=linewidth)
    plt.axhline(y=empty_acc, color='r', linestyle='--', linewidth=linewidth)
    plt.xlabel("Prune Rate", fontsize=xlabelfontsize)
    plt.ylabel("AUROC", fontsize=ylabelfontsize)
    plt.xticks(fontsize=xtickfontsize)
    plt.yticks(fontsize=ytickfontsize)
    plt.grid(gridon)

    # make legend without error bar
    if plot_legend:
        handles = []
        if prune_algos is None:
            for key in ["Random", "KNeighbor", "RankDegree", "LocalDegree", "SpanningForest", "Spanner-3", "Spanner-5", 
                        "Spanner-7", "ForestFire", "LSpar", "GSpar", "LocalSimilarity", "SCAN", "ER-unweighted"]:
                handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                        linewidth=linewidth, label=text_map[key])
                handles.append(handle)
        else:
            for key in prune_algos:
                handle = mlines.Line2D([], [], color=color_map[key], marker=marker_map[key], markersize=markersize, 
                                        linewidth=linewidth, label=text_map[key])
                handles.append(handle)
        # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legendfontsize)
        plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=4, fontsize=legendfontsize-3)
    else:
        plt.legend().set_visible(False)
    plt.tight_layout()

    # save plot
    figpath = osp.join(PROJECT_HOME, f"paper_fig/{dataset_name}_SAGE.{saveformat}")
    os.makedirs(osp.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)


if __name__ == "__main__":
    figwidth = 10
    figwidth_legend_compensation = 1
    figheight = 5.5
    figheight_legend_compensation = 1
    markersize = 20
    linewidth = 4
    capsize = 3 # error bar cap size
    capthick = 1
    elinewidth = 1
    titlefontsize = 30
    xlabelfontsize = 35
    ylabelfontsize = 35
    xtickfontsize = 30
    ytickfontsize = 30
    legendfontsize = 25
    gridon = True
    saveformat = "pdf"
    addTitle = False

    # ["Random", "KNeighbor", "LocalDegree", "RankDegree", "SpanningForest", "Spanner-3", "Spanner-5", "Spanner-7", 
    #  "ForestFire", "LSpar", "GSpar","LocalSimilarity", "SCAN", "ER-Max_weighted", "ER-Max_unweighted"]

    # SPSP_Eccentricity("ca-AstroPh")
    Centrality("ca-AstroPh", "TopCloseness", prune_algos=["Random", "LocalDegree", "RankDegree", "ForestFire", "LSpar", "GSpar", "SCAN"])
    # ClusteringF1Similarity("ca-HepPh", prune_algos=["Random", "KNeighbor", "LocalDegree", "LSpar", "GSpar", "LocalSimilarity", "SCAN", "ER-Max_weighted", "ER-Max_unweighted"])
    # MaxFlow("ca-HepPh", prune_algos=["Random", "KNeighbor", "ForestFire", "ER-Max_weighted", "ER-Max_unweighted"])
    # LocalClusteringCoefficient("com-Amazon", prune_algos=["Random", "KNeighbor", "SpanningForest", "Spanner-3", "Spanner-5", "Spanner-7", "LocalSimilarity", "GSpar", "SCAN"])
    # QuadraticFormSimilarity("com-Amazon", prune_algos=["Random", "ER-Max_weighted"])
    Centrality("com-DBLP", "EstimateBetweenness", prune_algos=["Random", "LocalDegree", "RankDegree", "ForestFire", "LSpar", "GSpar", "SCAN"])
    # DetectCommunity("com-DBLP", prune_algos=["Random", "KNeighbor", "LocalDegree", "RankDegree", "SpanningForest", "Spanner-3", "Spanner-5", "Spanner-7", "GSpar"])
    # SPSP_Eccentricity("com-DBLP")
    # Diameter("ego-Facebook", prune_algos=["Random", "LocalDegree", "RankDegree", "GSpar", "LocalSimilarity", "SCAN"])
    # Centrality("web-Google", "PageRank", prune_algos=["Random", "KNeighbor", "LocalDegree", "RankDegree", "GSpar", "SCAN", "ER-Max_weighted", "ER-Max_unweighted"])
    # Centrality("ego-Facebook", "PageRank", prune_algos=["Random", "KNeighbor", "LocalDegree", "RankDegree", "GSpar", "SCAN", "ER-Max_weighted", "ER-Max_unweighted"])
    # Centrality("ego-Twitter", "Katz", prune_algos=["Random", "KNeighbor", "LocalDegree", "RankDegree", "ForestFire", "ER-Max_unweighted"])
    # Centrality("email-Enron", "Eigenvector", prune_algos=["Random", "KNeighbor", "LocalDegree", "RankDegree", "ForestFire"])
    # GlobalClusteringCoefficient("human_gene2", prune_algos=["Random", "KNeighbor", "LocalSimilarity", "GSpar", "SCAN", "ER-Max_weighted"])
    # degreeDistribution("ogbn-proteins", prune_algos=["Random", "KNeighbor", "LocalDegree", "RankDegree", "ForestFire"])
    # GCN("ogbn-proteins")
    # SAGE("ogbn-proteins", prune_algos=["Random", "LocalDegree", "RankDegree", "GSpar", "LocalSimilarity", "SCAN"])
    # ClusterGCN("Reddit", prune_algos=["Random", "LocalDegree", "RankDegree", "ForestFire", "GSpar", "SCAN"])
    # sparsifier_time("ogbn-proteins")