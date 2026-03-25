class DiscreteRLAgent:
#     def __init__(self,
#                  v_bins=(0.0, 5.0, 10.0, 15.0, 25.0),
#                  d_bins=(0.0, 5.0, 10.0, 20.0, 40.0, 80.0),
#                  dv_bins=(-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0),
#                  actions=(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
#                  alpha=0.05,
#                  gamma=0.99,
#                  eps_start=1.0,
#                  eps_end=0.05,
#                  eps_decay=1e-4,
#                  q_init_path=None):
#         # Discretization
#         self.v_bins = v_bins
#         self.d_bins = d_bins
#         self.dv_bins = dv_bins
#         self.actions = actions

#         # RL params
#         self.alpha = alpha
#         self.gamma = gamma
#         self.eps = eps_start
#         self.eps_end = eps_end
#         self.eps_decay = eps_decay

#         # Q-table shape
#         shape = (len(v_bins) - 1,
#                  len(d_bins) - 1,
#                  len(dv_bins) - 1,
#                  len(actions))

#         # Initialize or load Q-table
#         if q_init_path is not None:
#             try:
#                 q_loaded = np.load(q_init_path)
#                 if q_loaded.shape == shape:
#                     self.q = q_loaded
#                     print(f"Loaded Q-table from {q_init_path}")
#                 else:
#                     print(f"Warning: Q-table shape mismatch {q_loaded.shape} != {shape}, using zeros.")
#                     self.q = np.zeros(shape, dtype=np.float32)
#             except FileNotFoundError:
#                 print(f"No existing Q-table at {q_init_path}, starting from zeros.")
#                 self.q = np.zeros(shape, dtype=np.float32)
#         else:
#             self.q = np.zeros(shape, dtype=np.float32)

#     def save(self, path):
#         np.save(path, self.q)
#         print(f"Saved Q-table to {path}")

#     def _bin_index(self, value, edges):
#         # returns index [0, len(edges)-2]
#         for i in range(len(edges) - 1):
#             if edges[i] <= value < edges[i + 1]:
#                 return i
#         return len(edges) - 2

#     def state_to_index(self, v, d, dv):
#         v_i = self._bin_index(v, self.v_bins)
#         d_i = self._bin_index(d, self.d_bins)
#         dv_i = self._bin_index(dv, self.dv_bins)
#         return (v_i, d_i, dv_i)

#     def select_action(self, state_idx):
#         # epsilon-greedy
#         if random.random() < self.eps:
#             a_idx = random.randrange(len(self.actions))
#         else:
#             q_vals = self.q[state_idx]
#             a_idx = int(np.argmax(q_vals))
#         return a_idx, self.actions[a_idx]

#     def update(self, s_idx, a_idx, reward, s_next_idx, done):
#         q_sa = self.q[s_idx][a_idx]
#         if done:
#             target = reward
#         else:
#             target = reward + self.gamma * np.max(self.q[s_next_idx])
#         self.q[s_idx][a_idx] = q_sa + self.alpha * (target - q_sa)

#         # decay epsilon
#         self.eps = max(self.eps_end, self.eps - self.eps_decay)