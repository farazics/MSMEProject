import numpy as np
q_last = np.load("q_table_ep4.npy")  # if num_episodes=5
q_last.shape  # should be (len(v_bins)-1, len(d_bins)-1, len(dv_bins)-1, len(actions))
