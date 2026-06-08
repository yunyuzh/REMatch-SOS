from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# User-editable parameters
# =========================
distance_matrix_file = "./greedy_rematch/distance_matrix_greedy.npy"
structure_list_file = "./greedy_rematch/cluster_assignment_greedy.txt"

sos_min = 0.5       # lower bound of SOS outlier score
sos_max = 0.7       # upper bound of SOS outlier score

perplexity = 30     # default SOS perplexity


def load_distance_matrix(distance_matrix_file):
    return np.load(distance_matrix_file)


def load_structure_names(structure_list_file):
    structure_names = []

    with open(structure_list_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.strip().split()
            structure_names.append(parts[0])

    return structure_names


def get_perplexity(D, beta):
    A = np.exp(-D * beta)
    sumA = np.sum(A)

    H = np.log(sumA) + beta * np.sum(D * A) / sumA

    return H, A


def distance_to_affinity(D, perplexity=30, eps=1e-5):
    n, _ = D.shape
    A = np.zeros((n, n))

    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for i in range(n):
        betamin = -np.inf
        betamax = np.inf

        other_indices = np.concatenate((np.r_[0:i], np.r_[i + 1:n]))
        Di = D[i, other_indices]

        H, thisA = get_perplexity(Di, beta[i])
        Hdiff = H - logU

        tries = 0
        while (np.isnan(Hdiff) or np.abs(Hdiff) > eps) and tries < 5000:
            if np.isnan(Hdiff):
                beta[i] = beta[i] / 10.0

            elif Hdiff > 0:
                betamin = beta[i].copy()

                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0

            else:
                betamax = beta[i].copy()

                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            H, thisA = get_perplexity(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        A[i, other_indices] = thisA

        if (i + 1) % 100 == 0:
            print(f"[INFO] Processed affinity row {i + 1}/{n}")

    return A


def affinity_to_binding_probability(A):
    B = A / A.sum(axis=1)[:, np.newaxis]
    return B


def binding_probability_to_outlier_score(B):
    O = np.prod(1 - B, axis=0)
    return O


def save_all_sos_scores(output_dir, structure_names, outlier_scores):
    df = pd.DataFrame({
        "structure_name": structure_names,
        "Outlier_Score": outlier_scores
    })

    output_path = Path(output_dir) / "sos_outlier_scores.csv"
    df.to_csv(output_path, index=False)


def save_selected_structures(output_dir, structure_names, outlier_scores, sos_min, sos_max):
    df = pd.DataFrame({
        "structure_name": structure_names,
        "Outlier_Score": outlier_scores
    })

    selected = df[
        (df["Outlier_Score"] >= sos_min) &
        (df["Outlier_Score"] <= sos_max)
    ]

    output_path = Path(output_dir) / "selected_structures.txt"

    selected.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    return selected


def main():
    output_dir = "./sos"
    Path(output_dir).mkdir(exist_ok=True)

    print("[INFO] Loading distance matrix...")
    D = load_distance_matrix(distance_matrix_file)

    print("[INFO] Loading structure names...")
    structure_names = load_structure_names(structure_list_file)

    print("[INFO] Calculating affinity matrix...")
    A = distance_to_affinity(D, perplexity=perplexity)

    print("[INFO] Calculating binding probability matrix...")
    B = affinity_to_binding_probability(A)

    print("[INFO] Calculating SOS outlier scores...")
    outlier_scores = binding_probability_to_outlier_score(B)

    print("[INFO] Saving all SOS scores...")
    save_all_sos_scores(output_dir, structure_names, outlier_scores)

    print("[INFO] Saving selected structures...")
    selected = save_selected_structures(
        output_dir,
        structure_names,
        outlier_scores,
        sos_min,
        sos_max
    )

    print(f"[DONE] Saved all SOS scores to: {output_dir}/sos_outlier_scores.csv")
    print(f"[DONE] Saved selected structures to: {output_dir}/selected_structures.txt")
    print(f"[DONE] Number of selected structures: {len(selected)}")


if __name__ == "__main__":
    main()
