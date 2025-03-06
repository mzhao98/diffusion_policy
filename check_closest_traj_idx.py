import numpy as np
import pickle
import pdb

# datafile = f'viz/gt_idx_0_timestep_to_divergences/gt_idx_{0}_timestep_to_divergences_{11}.pkl'
# datafile = f'viz/traj_opt/gt_idx_{0}_timestep_to_divergences_{1}.pkl'
# datafile = f'in_dist_viz_wo_traj/gt_idx_{0}_timestep_to_divergences_{11}.pkl'
datafile = f'recover/gt_idx_{0}_timestep_to_divergences_{3}.pkl'
# datafile = f'circle_ood/gt_idx_{0}_timestep_to_divergences_{48}.pkl'

with open(datafile, 'rb') as f:
    data = pickle.load(f)
# pdb.set_trace()

# traj_idx_to_total_divergence = {}
# for key_t in data:
#     print("key_t", key_t)
    
#     time_data = data[key_t]
#     for key_traj_idx in time_data:
#         traj_idx_data = time_data[key_traj_idx]
#         divergence = traj_idx_data['div']

#         if key_traj_idx not in traj_idx_to_total_divergence:
#             traj_idx_to_total_divergence[key_traj_idx] = 0
#         traj_idx_to_total_divergence[key_traj_idx] += divergence

# # get top 5 traj idx with lowest divergence
# # pdb.set_trace()
# sorted_traj_idx = sorted(traj_idx_to_total_divergence.items(), key=lambda x: x[1])
# top_5_traj_idx = sorted_traj_idx[:10]
# print("top_5_traj_idx", top_5_traj_idx)
# print("divergences", [x[1] for x in top_5_traj_idx])
# print()


for key_t in data:
    print("key_t", key_t)
    traj_idx_to_total_divergence = {}
    time_data = data[key_t]
    for key_traj_idx in time_data:
        traj_idx_data = time_data[key_traj_idx]
        divergence = traj_idx_data['div']

        if key_traj_idx not in traj_idx_to_total_divergence:
            traj_idx_to_total_divergence[key_traj_idx] = 0
        traj_idx_to_total_divergence[key_traj_idx] += divergence

    # get top 5 traj idx with lowest divergence
    # pdb.set_trace()
    sorted_traj_idx = sorted(traj_idx_to_total_divergence.items(), key=lambda x: x[1])
    top_5_traj_idx = sorted_traj_idx[:10]
    print("top_5_traj_idx", top_5_traj_idx)
    print("divergences", [x[1] for x in top_5_traj_idx])
    print()




