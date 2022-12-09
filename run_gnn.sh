#!/bin/bash

# for prune_method in "random" "sym_degree" "er"; do
#     for prune_rate in data/ogbn_products/pruned/${prune_method}/*; do
#         pruned_file_path=${prune_rate}/duw.el
#         log_path=${prune_rate}/gnn.log
#         
#         # echo ${pruned_file_path} ${log_path}
#         echo python gnn.py --runs 1 --pruned_file_path ${pruned_file_path} --log_path ${log_path} --device
#     done
# done


# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.1/duw.el --log_path data/ogbn_products/pruned/random/0.1/sage.log --use_sage --device 0 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.2/duw.el --log_path data/ogbn_products/pruned/random/0.2/sage.log --use_sage --device 1 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.3/duw.el --log_path data/ogbn_products/pruned/random/0.3/sage.log --use_sage --device 2 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.4/duw.el --log_path data/ogbn_products/pruned/random/0.4/sage.log --use_sage --device 3 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.5/duw.el --log_path data/ogbn_products/pruned/random/0.5/sage.log --use_sage --device 3 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.6/duw.el --log_path data/ogbn_products/pruned/random/0.6/sage.log --use_sage --device 2 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.7/duw.el --log_path data/ogbn_products/pruned/random/0.7/sage.log --use_sage --device 1 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.8/duw.el --log_path data/ogbn_products/pruned/random/0.8/sage.log --use_sage --device 0 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/sage.log --use_sage --device 0 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.95/duw.el --log_path data/ogbn_products/pruned/random/0.95/sage.log --use_sage --device 1 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.96/duw.el --log_path data/ogbn_products/pruned/random/0.96/sage.log --use_sage --device 2 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.97/duw.el --log_path data/ogbn_products/pruned/random/0.97/sage.log --use_sage --device 3 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/random/0.98/duw.el --log_path data/ogbn_products/pruned/random/0.98/sage.log --use_sage --device 0 &

# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.1/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.1/sage.log --use_sage --device 2 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.2/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.2/sage.log --use_sage --device 1 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.3/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.3/sage.log --use_sage --device 3 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.4/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.4/sage.log --use_sage --device 0 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.5/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.5/sage.log --use_sage --device 1 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.6/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.6/sage.log --use_sage --device 3 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.7/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.7/sage.log --use_sage --device 0 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.8/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.8/sage.log --use_sage --device 2 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/sage.log --use_sage --device 0 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.935/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.935/sage.log --use_sage --device 1 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.966/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.966/sage.log --use_sage --device 2 &

# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.132/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.132/sage.log --use_sage --device 3 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.17/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.17/sage.log --use_sage --device 0 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.206/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.206/sage.log --use_sage --device 1 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.246/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.246/sage.log --use_sage --device 2 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.294/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.294/sage.log --use_sage --device 3 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.356/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.356/sage.log --use_sage --device 2 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.442/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.442/sage.log --use_sage --device 1 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.581/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.581/sage.log --use_sage --device 0 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/sage.log --use_sage --device 3 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_1.3/duw.el --log_path data/ogbn_products/pruned/er/epsilon_1.3/sage.log --use_sage --device 1 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_1.47/duw.el --log_path data/ogbn_products/pruned/er/epsilon_1.47/sage.log --use_sage --device 0 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_1.71/duw.el --log_path data/ogbn_products/pruned/er/epsilon_1.71/sage.log --use_sage --device 2 &
# python gnn.py --runs 1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_2.1/duw.el --log_path data/ogbn_products/pruned/er/epsilon_2.1/sage.log --use_sage --device 3 &

# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 0.132 --log_path data/ogbn_products/pruned/er/epsilon_0.132/sage_dym_sample.log --use_sage --device 0 &
# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 0.17 --log_path data/ogbn_products/pruned/er/epsilon_0.17/sage_dym_sample.log --use_sage --device 1 &
# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 0.206 --log_path data/ogbn_products/pruned/er/epsilon_0.206/sage_dym_sample.log --use_sage --device 2 &
# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 0.246 --log_path data/ogbn_products/pruned/er/epsilon_0.246/sage_dym_sample.log --use_sage --device 1 &
# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 0.294 --log_path data/ogbn_products/pruned/er/epsilon_0.294/sage_dym_sample.log --use_sage --device 3 &
# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 0.356 --log_path data/ogbn_products/pruned/er/epsilon_0.356/sage_dym_sample.log --use_sage --device 2 &
# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 0.442 --log_path data/ogbn_products/pruned/er/epsilon_0.442/sage_dym_sample.log --use_sage --device 3 &
# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 0.581 --log_path data/ogbn_products/pruned/er/epsilon_0.581/sage_dym_sample.log --use_sage --device 3 &
# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 0.885 --log_path data/ogbn_products/pruned/er/epsilon_0.885/sage_dym_sample.log --use_sage --device 3 &
# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 1.3 --log_path data/ogbn_products/pruned/er/epsilon_1.3/sage_dym_sample.log --use_sage --device 3 &
# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 1.47 --log_path data/ogbn_products/pruned/er/epsilon_1.47/sage_dym_sample.log --use_sage --device 3 &
# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 1.71 --log_path data/ogbn_products/pruned/er/epsilon_1.71/sage_dym_sample.log --use_sage --device 3 &
# python gnn.py --runs 1 --reff_var_path data/ogbn_products/raw/stage3.npz --epsilon 2.1 --log_path data/ogbn_products/pruned/er/epsilon_2.1/sage_dym_sample.log --use_sage --device 3 &


# python gnn.py --runs 1 --x_drop_rate 0 --log_path x_drop/gcn_0.log --device 0 &
# python gnn.py --runs 1 --x_drop_rate 0.1 --log_path x_drop/gcn_0.1.log --device 0 &
# python gnn.py --runs 1 --x_drop_rate 0.2 --log_path x_drop/gcn_0.2.log --device 1 &
# python gnn.py --runs 1 --x_drop_rate 0.3 --log_path x_drop/gcn_0.3.log --device 2 &
# python gnn.py --runs 1 --x_drop_rate 0.4 --log_path x_drop/gcn_0.4.log --device 3 &
# python gnn.py --runs 1 --x_drop_rate 0.5 --log_path x_drop/gcn_0.5.log --device 1 &
# python gnn.py --runs 1 --x_drop_rate 0.6 --log_path x_drop/gcn_0.6.log --device 2 &
# python gnn.py --runs 1 --x_drop_rate 0.7 --log_path x_drop/gcn_0.7.log --device 3 &
# python gnn.py --runs 1 --x_drop_rate 0.8 --log_path x_drop/gcn_0.8.log --device 0 &
# python gnn.py --runs 1 --x_drop_rate 0.9 --log_path x_drop/gcn_0.9.log --device 1 &
# python gnn.py --runs 1 --x_drop_rate 0.95 --log_path x_drop/gcn_0.95.log --device 2 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0 --log_path x_drop/sage_0.log --device 3 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.1 --log_path x_drop/sage_0.1.log --device 0 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.2 --log_path x_drop/sage_0.2.log --device 1 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.3 --log_path x_drop/sage_0.3.log --device 2 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.4 --log_path x_drop/sage_0.4.log --device 3 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.5 --log_path x_drop/sage_0.5.log --device 0 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.6 --log_path x_drop/sage_0.6.log --device 1 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.7 --log_path x_drop/sage_0.7.log --device 2 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.8 --log_path x_drop/sage_0.8.log --device 3 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.9 --log_path x_drop/sage_0.9.log --device 2 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.95 --log_path x_drop/sage_0.95.log --device 3 &


# python gnn.py --runs 1 --x_drop_rate 0.1 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/gcn_x_drop_0.1.log --device 0 &
# python gnn.py --runs 1 --x_drop_rate 0.2 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/gcn_x_drop_0.2.log --device 1 &
# python gnn.py --runs 1 --x_drop_rate 0.3 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/gcn_x_drop_0.3.log --device 2 &
# python gnn.py --runs 1 --x_drop_rate 0.4 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/gcn_x_drop_0.4.log --device 3 &
# python gnn.py --runs 1 --x_drop_rate 0.5 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/gcn_x_drop_0.5.log --device 0 &
# python gnn.py --runs 1 --x_drop_rate 0.6 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/gcn_x_drop_0.6.log --device 1 &
# python gnn.py --runs 1 --x_drop_rate 0.7 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/gcn_x_drop_0.7.log --device 2 &
# python gnn.py --runs 1 --x_drop_rate 0.8 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/gcn_x_drop_0.8.log --device 3 &
# python gnn.py --runs 1 --x_drop_rate 0.9 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/gcn_x_drop_0.9.log --device 0 &
# python gnn.py --runs 1 --x_drop_rate 0.95 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/gcn_x_drop_0.95.log --device 1 &

# python gnn.py --runs 1 --use_sage --x_drop_rate 0.1 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/sage_x_drop_0.1.log --device 2 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.2 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/sage_x_drop_0.2.log --device 3 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.3 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/sage_x_drop_0.3.log --device 0 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.4 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/sage_x_drop_0.4.log --device 1 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.5 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/sage_x_drop_0.5.log --device 2 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.6 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/sage_x_drop_0.6.log --device 3 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.7 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/sage_x_drop_0.7.log --device 0 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.8 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/sage_x_drop_0.8.log --device 1 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.9 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/sage_x_drop_0.9.log --device 2 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.95 --pruned_file_path data/ogbn_products/pruned/random/0.9/duw.el --log_path data/ogbn_products/pruned/random/0.9/sage_x_drop_0.95.log --device 3 &

# python gnn.py --runs 1 --x_drop_rate 0.1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/gcn_x_drop_0.1.log --device 2 &
# python gnn.py --runs 1 --x_drop_rate 0.2 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/gcn_x_drop_0.2.log --device 3 &
# python gnn.py --runs 1 --x_drop_rate 0.3 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/gcn_x_drop_0.3.log --device 0 &
# python gnn.py --runs 1 --x_drop_rate 0.4 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/gcn_x_drop_0.4.log --device 1 &
# python gnn.py --runs 1 --x_drop_rate 0.5 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/gcn_x_drop_0.5.log --device 2 &
# python gnn.py --runs 1 --x_drop_rate 0.6 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/gcn_x_drop_0.6.log --device 3 &
# python gnn.py --runs 1 --x_drop_rate 0.7 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/gcn_x_drop_0.7.log --device 0 &
# python gnn.py --runs 1 --x_drop_rate 0.8 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/gcn_x_drop_0.8.log --device 1 &
# python gnn.py --runs 1 --x_drop_rate 0.9 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/gcn_x_drop_0.9.log --device 2 &
# python gnn.py --runs 1 --x_drop_rate 0.95 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/gcn_x_drop_0.95.log --device 3 &

# python gnn.py --runs 1 --use_sage --x_drop_rate 0.1 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/sage_x_drop_0.1.log --device 0 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.2 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/sage_x_drop_0.2.log --device 1 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.3 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/sage_x_drop_0.3.log --device 2 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.4 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/sage_x_drop_0.4.log --device 3 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.5 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/sage_x_drop_0.5.log --device 0 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.6 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/sage_x_drop_0.6.log --device 1 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.7 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/sage_x_drop_0.7.log --device 2 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.8 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/sage_x_drop_0.8.log --device 3 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.9 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/sage_x_drop_0.9.log --device 0 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.95 --pruned_file_path data/ogbn_products/pruned/sym_degree/0.9/duw.el --log_path data/ogbn_products/pruned/sym_degree/0.9/sage_x_drop_0.95.log --device 1 &

# python gnn.py --runs 1 --x_drop_rate 0.1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/gcn_x_drop_0.1.log --device 0 &
# python gnn.py --runs 1 --x_drop_rate 0.2 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/gcn_x_drop_0.2.log --device 1 &
# python gnn.py --runs 1 --x_drop_rate 0.3 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/gcn_x_drop_0.3.log --device 2 &
# python gnn.py --runs 1 --x_drop_rate 0.4 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/gcn_x_drop_0.4.log --device 3 &
# python gnn.py --runs 1 --x_drop_rate 0.5 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/gcn_x_drop_0.5.log --device 0 &
# python gnn.py --runs 1 --x_drop_rate 0.6 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/gcn_x_drop_0.6.log --device 1 &
# python gnn.py --runs 1 --x_drop_rate 0.7 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/gcn_x_drop_0.7.log --device 2 &
# python gnn.py --runs 1 --x_drop_rate 0.8 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/gcn_x_drop_0.8.log --device 3 &
# python gnn.py --runs 1 --x_drop_rate 0.9 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/gcn_x_drop_0.9.log --device 0 &
# python gnn.py --runs 1 --x_drop_rate 0.95 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/gcn_x_drop_0.95.log --device 1 &

# python gnn.py --runs 1 --use_sage --x_drop_rate 0.1 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/sage_x_drop_0.1.log --device 2 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.2 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/sage_x_drop_0.2.log --device 3 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.3 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/sage_x_drop_0.3.log --device 0 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.4 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/sage_x_drop_0.4.log --device 1 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.5 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/sage_x_drop_0.5.log --device 2 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.6 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/sage_x_drop_0.6.log --device 3 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.7 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/sage_x_drop_0.7.log --device 0 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.8 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/sage_x_drop_0.8.log --device 1 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.9 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/sage_x_drop_0.9.log --device 2 &
# python gnn.py --runs 1 --use_sage --x_drop_rate 0.95 --pruned_file_path data/ogbn_products/pruned/er/epsilon_0.885/duw.el --log_path data/ogbn_products/pruned/er/epsilon_0.885/sage_x_drop_0.95.log --device 3 &


# for workload in "bfs" "cc" "pr" "sssp"; do
#     # baseline
#     output_folder="./experiments/${workload}/ogbn_products/baseline"
#     mkdir -p $output_folder
#     ./workload/gapbs/${workload} -f ./data/ogbn_products/raw/duw.el -n 1 -v -a -z ${output_folder}/analysis.txt > ${output_folder}/stdout.txt & 
#     # pruned
#     for prune_method in "random" "sym_degree" "er"; do
#         for folder in ./data/ogbn_products/pruned/${prune_method}/*; do
#             # split the folder name by / and get the last part as prune_rate
#             prune_rate=$(echo $folder | cut -d'/' -f 6)
#             echo $prune_rate
#             output_folder="./experiments/${workload}/ogbn_products/${prune_method}/${prune_rate}"
#             mkdir -p $output_folder
#             ./workload/gapbs/${workload} -f ${folder}/duw.el -n 1 -v -a -z ${output_folder}/analysis.txt > ${output_folder}/stdout.txt & 
#         done
#     done
# done

# baseline
# output_folder="./experiments/tc/ogbn_products/baseline"
# mkdir -p $output_folder
# ./workload/gapbs/tc -f ./data/ogbn_products/raw/duw.el -n 1 -s -v -a -z ${output_folder}/analysis.txt > ${output_folder}/stdout.txt & 
# pruned
# for prune_method in "random" "sym_degree" "er"; do
for prune_method in "er"; do
    for folder in ./data/ogbn_products/pruned/${prune_method}/*; do
        # split the folder name by / and get the last part as prune_rate
        prune_rate=$(echo $folder | cut -d'/' -f 6)
        echo $prune_rate
        output_folder="./experiments/tc/ogbn_products/${prune_method}/${prune_rate}"
        mkdir -p $output_folder
        ./workload/gapbs/tc -f ${folder}/duw.el -n 1 -s -v -a -z ${output_folder}/analysis.txt > ${output_folder}/stdout.txt & 
    done
done
