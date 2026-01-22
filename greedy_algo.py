import os
import numpy as np
from sklearn.preprocessing import normalize
from dscribe.kernels import REMatchKernel

def load_soap(directory):
    features_list = []
    indices = []

    # Change 10000 to your dataset size if needed.
    for i in range(10000):
        file_path = os.path.join(directory, f'A{i}.csv')
        if os.path.exists(file_path):
            features = np.loadtxt(file_path, delimiter=',')
            features = normalize(features)
            features_list.append(features)
            indices.append(i)
    features = np.vstack(features_list)
    variance = np.var(features)
    n_features = features.shape[1]
    gamma = 1 / (n_features * variance)
    return features_list, indices, gamma


def rematch_similarity(soap_a, soap_b, re_kernel):
    K = re_kernel.create([soap_a, soap_b])
    return float(K[0, 1])


def select_representatives(features_list, n_reps, re_kernel):
    N = len(features_list)

    n_reps = min(int(n_reps), N)
    sims_with_reps = np.zeros((n_reps, N), dtype=np.float32)

    rep_indices = []
    is_rep = np.zeros(N, dtype=bool)

    rep0 = 0
    rep_indices.append(rep0)
    is_rep[rep0] = True

    for j in range(N):
        sims_with_reps[0, j] = rematch_similarity(features_list[rep0], features_list[j], re_kernel)

    sim_sum = sims_with_reps[0].copy()

    for k in range(1, n_reps):
        candidates = np.where(~is_rep)[0]
        if candidates.size == 0:
            break

        best = candidates[np.argmin(sim_sum[candidates])]
        rep_indices.append(best)
        is_rep[best] = True

        for j in range(N):
            sims_with_reps[k, j] = rematch_similarity(features_list[best], features_list[j], re_kernel)

        sim_sum += sims_with_reps[k]

    reps_used = len(rep_indices)
    sims_with_reps = sims_with_reps[:reps_used, :]
    return rep_indices, sims_with_reps

def assign_clusters(sims_with_reps):
    return np.argmax(sims_with_reps, axis=0).astype(int)


def build_rep_similarity_matrix(rep_indices, sims_with_reps):
    n_reps = len(rep_indices)
    S = np.zeros((n_reps, n_reps), dtype=np.float32)
    for a in range(n_reps):
        for b in range(n_reps):
            S[a, b] = sims_with_reps[a, rep_indices[b]]

    S = 0.5 * (S + S.T)
    return S


def build_full_similarity_matrix(rep_sim_matrix, cluster_ids):
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    return rep_sim_matrix[cluster_ids[:, None], cluster_ids[None, :]].astype(np.float32)


def process_greedy_rematch(soap_dir, output_dir):
    n_representatives = 100
    alpha = 1.0
    threshold = 1e-6

    features_list, indices, gamma = load_soap(soap_dir)
    N = len(indices)
    print(f"[INFO] Loaded structures: {N}")
    print(f"[INFO] Estimated gamma: {gamma:.6e}")

    re_kernel = REMatchKernel(metric="rbf", gamma=gamma, alpha=alpha, threshold=threshold)

    rep_indices, sims_with_reps = select_representatives(features_list, n_representatives, re_kernel)
    cluster_ids = assign_clusters(sims_with_reps)

    rep_sim = build_rep_similarity_matrix(rep_indices, sims_with_reps)
    print("[INFO] Building full approximate similarity matrix (NxN)...")
    full_sim = build_full_similarity_matrix(rep_sim, cluster_ids)

    full_dist = 1.0 - full_sim
    full_dist = np.clip(full_dist, 0.0, None)
    np.fill_diagonal(full_dist, 0.0)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "similarity_matrix_greedy.npy"), full_sim)
    np.save(os.path.join(output_dir, "distance_matrix_greedy.npy"), full_dist)

    with open(os.path.join(output_dir, "representatives_greedy.txt"), "w") as f:
        f.write("# k\tlist_index\tglobal_index\n")
        for k, li in enumerate(rep_indices):
            f.write(f"{k}\t{li}\t{indices[li]}\n")

    with open(os.path.join(output_dir, "cluster_assignment_greedy.txt"), "w") as f:
        f.write("# global_index\trep_id\trep_global_index\n")
        for li in range(N):
            rep_id = int(cluster_ids[li])
            rep_li = rep_indices[rep_id]
            f.write(f"{indices[li]}\t{rep_id}\t{indices[rep_li]}\n")

    print(f"[DONE] Saved: {output_dir}/distance_matrix_greedy.npy")
    print(f"[DONE] Saved: {output_dir}/similarity_matrix_greedy.npy")
