
        myLogger.info(
            "pruned edge list will be post symmetrized, make sure the input edgelist is symmetrical, otherwise the result prune rate will be incorrect!"
        )

    duw_el_path = osp.join(
        osp.dirname(osp.abspath(__file__)),
        f"../data/{dataset_name}/pruned/random/{prune_rate}/duw.el",
    )

    if dataset_name in ["Reddit", "Reddit2", "ogbn_products"]:
        size = dataset.data.edge_index.shape[1]
    else:
        myLogger.info(message=f"{dataset_name} is not pyg dataset. Exiting...")
        sys.exit(1)

    edge_selection = torch.tensor(
        np.random.binomial(1, 1.0 - prune_rate, size=size)
    ).type(torch.bool)

    data = dataset.data
    original_num_edges = data.edge_index.shape[1]

    if post_symmetrize:  # pick src < dst edges
        edge_index = data.edge_index[:, edge_selection]

        # post_symmetrize
        edge_set = set()
        for i in range(edge_index.shape[1]):
            edge_set.add((edge_index[0, i], edge_index[1, i]))
            edge_set.add((edge_index[1, i], edge_index[0, i]))

        data.edge_list = torch.tensor(np.array(list(edge_set)).transpose())
    else:
        data.edge_index = data.edge_index[:, edge_selection]

    new_num_edges = data.edge_index.shape[1]

    if save_edgelist:
        os.makedirs(osp.dirname(duw_el_path), exist_ok=True)
        to_save = data.edge_index.numpy().transpose().astype(int)
        np.savetxt(duw_el_path, to_save, fmt="%i")

    dataset.data = data

    myLogger.info(
        f"dataset: {dataset_name}, target prune rate: {prune_rate}, actual prune rate: {1.0 - new_num_edges / original_num_edges}\n"
        + f"---------- Random sparsify End -----------\n"
    )

    return dataset


def el_random_sparsify(
    dataset,
    dataset_name,
    prune_rate,
    save_edgelist=True,
    post_symmetrize=False,
):
    """
    Input:
        dataset: str, path to edgelist file, must be absolute path
        dataset_name: str, name of dataset
        prune_rate: float, between 0 and 1
        save_edgelist: not used in this functin, kept for compatibility
        post_symmetrize: bool, if True, post symmetrize pruned edge list
    Output:
        None, pruned edgelist file saved
    """
    # sanity checks
    if not isinstance(dataset, str):
        raise ValueError(f"dataset must be str, got {type(dataset)}")
    if not osp.isabs(dataset):
        raise ValueError(f"dataset must be absolute path, got {dataset}")
    if not osp.exists(dataset):
        raise ValueError(f"dataset {dataset} does not exist")

    cwd = os.getcwd()
    current_file_dir = os.path.dirname(os.path.realpath(__file__))

    myLogger.info(
        f"---------- Random sparsify -----------\n"
        + f"dataset: {dataset_name}, prune_rate: {prune_rate}"
    )
    if post_symmetrize:
        myLogger.info(
            "pruned edge list will be post symmetrized, make sure the input edgelist is symmetrical, otherwise the result prune rate will be incorrect!"
        )

    duw_el_path = os.path.join(
        current_file_dir,
        f"../data/{dataset_name}/pruned/random/",
        f"{prune_rate}/duw.el",
    )

    os.makedirs(os.path.dirname(duw_el_path), exist_ok=True)

    os.chdir(current_file_dir)
    if post_symmetrize:
        os.system(
            f"./bin/prune -f {dataset} -q random -p {prune_rate} -o {duw_el_path} -b"
        )
    else:
        os.system(
            f"./bin/prune -f {dataset} -q random -p {prune_rate} -o {duw_el_path}"
        )
    os.chdir(cwd)


# This is the top level function
def random_sparsify(
    dataset,
    dataset_name,
    prune_rate,
    dataset_type="el",
    save_edgelist=True,
    post_symmetrize=False,
):
    """
    Input:
        dataset: str, path to edgelist file, must be absolute path
        dataset_name: str, name of dataset
        prune_rate: float, between 0 and 1
        dataset_type: str, [el/pyg], el for edgelist, pyg for pyg object
        save_edgelist: only used for pyg_random_sparsify, el_random_sparsify is always saved
        post_symmetrize: bool, if True, symmetrize pruned edge list
    """
    if dataset_type == "el":
        myLogger.info("---------- Random sparsify for el format input -----------")
        el_random_sparsify(
            dataset,
            dataset_name,
            prune_rate,
            save_edgelist=save_edgelist,
            post_symmetrize=post_symmetrize,
        )
    elif dataset_type == "pyg":
        myLogger.info("---------- Random sparsify for pyg format input -----------")
        pyg_random_sparsify(
            dataset,
            dataset_name,
            prune_rate,
            save_edgelist=save_edgelist,
            post_symmetrize=post_symmetrize,
        )
    else:
        raise ValueError(f"dataset_type must be el/pyg, got {dataset_type}")
