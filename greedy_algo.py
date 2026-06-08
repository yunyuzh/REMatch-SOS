from pathlib import Path
import numpy as np
from sklearn.preprocessing import normalize
from dscribe.kernels import REMatchKernel


# =========================
# User-editable parameters
# =========================
soap_dir = "./soap"          # folder containing SOAP .csv files
n_representatives = 100      # number of representative structures to select


def load_soap_descriptors(soap_dir):
    soap_path = Path(soap_dir)
    csv_files = sorted(soap_path.glob("*.csv"))

    features_list = []
    file_names = []

    for csv_file in csv_files:
        features = np.loadtxt(csv_file, delimiter=",")

        # Each SOAP descriptor is normalized before REMatch comparison.
        features = normalize(features)

        features_list.append(features)
        file_names.append(csv_file.stem)

    return features_list, file_names


def estimate_gamma(features_list):
    all_features = np.vstack(features_list)
    variance = np.var(all_features)
    n_features = all_features.shape[1]

    gamma = 1.0 / (n_features * variance)
    return gamma


def calculate_rematch_similarity(soap_a, soap_b, re_kernel):
    K = re_kernel.create([soap_a, soap_b])
    return float(K[0, 1])


def select_greedy_representatives(features_list, n_representatives, re_kernel):
    n_structures = len(features_list)
    n_representatives = min(n_representatives, n_structures)

    sims_with_reps = np.zeros(
        (n_representatives, n_structures),
        dtype=np.float32
    )

    representative_indices = []
    is_representative = np.zeros(n_structures, dtype=bool)

    first_rep = 0
    representative_indices.append(first_rep)
    is_representative[first_rep] = True

    for j in range(n_structures):
        sims_with_reps[0, j] = calculate_rematch_similarity(
            features_list[first_rep],
            features_list[j],
            re_kernel
        )

    sim_sum = sims_with_reps[0].copy()

    for k in range(1, n_representatives):
        candidates = np.where(~is_representative)[0]

        new_rep = candidates[np.argmin(sim_sum[candidates])]

        representative_indices.append(new_rep)
        is_representative[new_rep] = True

        for j in range(n_structures):
            sims_with_reps[k, j] = calculate_rematch_similarity(
                features_list[new_rep],
                features_list[j],
                re_kernel
            )

        sim_sum += sims_with_reps[k]

        print(f"Selected representative {k + 1}/{n_representatives}")

    return representative_indices, sims_with_reps


def assign_structures_to_representatives(sims_with_reps):
    cluster_ids = np.argmax(sims_with_reps, axis=0)
    return cluster_ids.astype(int)


def build_representative_similarity_matrix(representative_indices, sims_with_reps):
    n_reps = len(representative_indices)
    rep_sim_matrix = np.zeros((n_reps, n_reps), dtype=np.float32)

    for i in range(n_reps):
        for j in range(n_reps):
            rep_sim_matrix[i, j] = sims_with_reps[i, representative_indices[j]]

    rep_sim_matrix = 0.5 * (rep_sim_matrix + rep_sim_matrix.T)

    return rep_sim_matrix


def build_approximate_similarity_matrix(rep_sim_matrix, cluster_ids):
    cluster_ids = np.asarray(cluster_ids, dtype=int)

    full_similarity_matrix = rep_sim_matrix[
        cluster_ids[:, None],
        cluster_ids[None, :]
    ]

    return full_similarity_matrix.astype(np.float32)


def save_outputs(
    output_dir,
    full_similarity_matrix,
    full_distance_matrix,
    representative_indices,
    cluster_ids,
    file_names
):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    np.save(output_path / "similarity_matrix_greedy.npy", full_similarity_matrix)
    np.save(output_path / "distance_matrix_greedy.npy", full_distance_matrix)

    with open(output_path / "representatives_greedy.txt", "w") as f:
        f.write("# representative_id\tlist_index\tstructure_name\n")
        for rep_id, list_index in enumerate(representative_indices):
            f.write(f"{rep_id}\t{list_index}\t{file_names[list_index]}\n")

    with open(output_path / "cluster_assignment_greedy.txt", "w") as f:
        f.write("# structure_name\trepresentative_id\trepresentative_name\n")
        for structure_id, rep_id in enumerate(cluster_ids):
            rep_index = representative_indices[rep_id]
            f.write(
                f"{file_names[structure_id]}\t"
                f"{rep_id}\t"
                f"{file_names[rep_index]}\n"
            )


def main():
    output_dir = "./greedy_rematch"

    print("[INFO] Loading SOAP descriptors...")
    features_list, file_names = load_soap_descriptors(soap_dir)

    print(f"[INFO] Loaded structures: {len(features_list)}")

    print("[INFO] Estimating gamma...")
    gamma = estimate_gamma(features_list)
    print(f"[INFO] Estimated gamma: {gamma:.6e}")

    # REMatch parameters are kept inside the script.
    alpha = 1.0
    threshold = 1e-6

    re_kernel = REMatchKernel(
        metric="rbf",
        gamma=gamma,
        alpha=alpha,
        threshold=threshold
    )

    print("[INFO] Selecting representative structures...")
    representative_indices, sims_with_reps = select_greedy_representatives(
        features_list,
        n_representatives,
        re_kernel
    )

    print("[INFO] Assigning structures to representatives...")
    cluster_ids = assign_structures_to_representatives(sims_with_reps)

    print("[INFO] Building representative similarity matrix...")
    rep_sim_matrix = build_representative_similarity_matrix(
        representative_indices,
        sims_with_reps
    )

    print("[INFO] Building approximate full similarity matrix...")
    full_similarity_matrix = build_approximate_similarity_matrix(
        rep_sim_matrix,
        cluster_ids
    )

    print("[INFO] Converting similarity matrix to distance matrix...")
    full_distance_matrix = 1.0 - full_similarity_matrix
    full_distance_matrix = np.clip(full_distance_matrix, 0.0, None)
    np.fill_diagonal(full_distance_matrix, 0.0)

    save_outputs(
        output_dir,
        full_similarity_matrix,
        full_distance_matrix,
        representative_indices,
        cluster_ids,
        file_names
    )

    print(f"[DONE] Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
