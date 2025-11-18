import os
import numpy as np
from sklearn.preprocessing import normalize
from dscribe.kernels import REMatchKernel


SOAP_DIR = "/work/e05/e05/uccayz3/Tom/soap"
OUTPUT_DIR = "/work/e05/e05/uccayz3/Tom"
N_REPRESENTATIVES = 100
ALPHA = 1.0
RE_THRESHOLD = 1e-6

def list_soap_files(soap_dir):
    file_infos = []
    for fname in os.listdir(soap_dir):
        if fname.startswith("X") and fname.endswith(".csv"):
            num_str = fname[1:-4]
            try:
                idx = int(num_str)
            except ValueError:
                continue
            file_infos.append((idx, fname))

    if not file_infos:
        raise RuntimeError(f"No files like X*.csv found in {soap_dir}")

    file_infos.sort(key=lambda x: x[0])

    indices = [idx for idx, _ in file_infos]
    file_names = [fname for _, fname in file_infos]
    file_paths = [os.path.join(soap_dir, fname) for fname in file_names]

    return file_paths, file_names, indices


def load_all_soap_and_gamma(file_paths):
    soaps = []
    for path in file_paths:
        arr = np.loadtxt(path, delimiter=",")
        arr = normalize(arr)
        soaps.append(arr)

    all_features = np.vstack(soaps)
    variance = np.var(all_features)
    n_features = all_features.shape[1]
    gamma = 1.0 / (n_features * variance)

    return soaps, gamma


def rematch_similarity(soap_a, soap_b, re_kernel):
    K = re_kernel.create([soap_a, soap_b])
    return float(K[0, 1])


def select_representatives(soaps, n_reps, re_kernel, file_names=None):
    N = len(soaps)
    if N == 0:
        raise RuntimeError("No SOAP data loaded.")

    n_reps = min(n_reps, N)

    sims_with_reps = np.zeros((n_reps, N), dtype=np.float32)

    rep_indices = []
    is_rep = np.zeros(N, dtype=bool)

    first_rep = 0
    rep_indices.append(first_rep)
    is_rep[first_rep] = True

    print(f"[REP 0] Use structure index {first_rep} as the first representative "
          f"({file_names[first_rep] if file_names else ''})")

    for j in range(N):
        sims_with_reps[0, j] = rematch_similarity(soaps[first_rep], soaps[j], re_kernel)
    sim_sum = sims_with_reps[0].copy()

    for k in range(1, n_reps):
        candidate_indices = np.where(~is_rep)[0]
        if candidate_indices.size == 0:
            print("[INFO] No more non-representatives left.")
            break

        best_idx = candidate_indices[np.argmin(sim_sum[candidate_indices])]

        rep_indices.append(best_idx)
        is_rep[best_idx] = True

        print(f"[REP {k}] Selected structure index {best_idx} as representative "
              f"({file_names[best_idx] if file_names else ''}), "
              f"min total similarity = {sim_sum[best_idx]:.6f}")

        for j in range(N):
            sims_with_reps[k, j] = rematch_similarity(soaps[best_idx], soaps[j], re_kernel)
        sim_sum += sims_with_reps[k]

    actual_reps = len(rep_indices)
    sims_with_reps = sims_with_reps[:actual_reps, :]

    print(f"[INFO] Total representatives used: {actual_reps}")
    return rep_indices, sims_with_reps


def assign_clusters(sims_with_reps):
    cluster_ids = np.argmax(sims_with_reps, axis=0)
    return cluster_ids


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
    full_sim = rep_sim_matrix[cluster_ids[:, None], cluster_ids[None, :]].astype(np.float32)
    return full_sim


def main():
    file_paths, file_names, indices = list_soap_files(SOAP_DIR)
    N = len(file_paths)
    print(f"[INFO] Found {N} SOAP files under {SOAP_DIR}")

    print("[INFO] Loading all SOAP descriptors and estimating gamma...")
    soaps, gamma = load_all_soap_and_gamma(file_paths)
    print(f"[INFO] Estimated gamma = {gamma}")

    re_kernel = REMatchKernel(
        metric="rbf",
        gamma=gamma,
        alpha=ALPHA,
        threshold=RE_THRESHOLD
    )

    # 4. 选择代表结构
    rep_indices, sims_with_reps = select_representatives(
        soaps,
        N_REPRESENTATIVES,
        re_kernel,
        file_names=file_names
    )

    actual_reps = len(rep_indices)

    cluster_ids = assign_clusters(sims_with_reps)

    rep_sim_matrix = build_rep_similarity_matrix(rep_indices, sims_with_reps)

    print("[INFO] Building full approximate similarity matrix...")
    full_sim = build_full_similarity_matrix(rep_sim_matrix, cluster_ids)

    print("[INFO] Converting similarity matrix to distance matrix...")
    full_dist = 1.0 - full_sim
    full_dist = np.clip(full_dist, 0.0, None)
    np.fill_diagonal(full_dist, 0.0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sim_path = os.path.join(OUTPUT_DIR, "similarity_matrix.npy")
    dist_path = os.path.join(OUTPUT_DIR, "distance_matrix.npy")
    reps_txt = os.path.join(OUTPUT_DIR, "representatives.txt")
    cluster_txt = os.path.join(OUTPUT_DIR, "cluster_assignment.txt")

    np.save(sim_path, full_sim)
    np.save(dist_path, full_dist)

    with open(reps_txt, "w") as f:
        f.write("# Representative structures (by selection order):\n")
        for k, idx in enumerate(rep_indices):
            f.write(f"{k}\tindex={indices[idx]}\tfile={file_names[idx]}\n")

    with open(cluster_txt, "w") as f:
        f.write("# Each line: global_index\tfile_name\trep_selection_id\trep_global_index\trep_file_name\n")
        for i in range(N):
            rep_id = cluster_ids[i]
            rep_global_idx = rep_indices[rep_id]
            f.write(f"{indices[i]}\t{file_names[i]}\t{rep_id}\t{indices[rep_global_idx]}\t{file_names[rep_global_idx]}\n")

    print(f"[DONE] Approximate similarity matrix saved to: {sim_path}")
    print(f"[DONE] Approximate distance matrix  saved to: {dist_path}")
    print(f"[DONE] Representatives list saved to: {reps_txt}")
    print(f"[DONE] Cluster assignment saved to: {cluster_txt}")


if __name__ == "__main__":
    main()
