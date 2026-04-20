from pathlib import Path
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

# =========================
# User-editable parameters
# =========================
input_dir = "/work/gin"   # folder containing .gin files
target_elements = ["Au"]                      # atoms used as SOAP centers
species = ["Au", "Ce", "O"]                  # all species appearing in the system


def parse_gin_to_atoms(gin_path):
    symbols = []
    positions = []

    with open(gin_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 11:
                element = parts[0]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                symbols.append(element)
                positions.append((x, y, z))

    return Atoms(symbols=symbols, positions=positions)


def generate_soap_descriptor(atoms, csv_path, target_elements, species):
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=10.0,
        n_max=2,
        l_max=2,
        compression={'mode': 'mu1nu1', 'species_weighting': None}
    )

    target_indices = [atom.index for atom in atoms if atom.symbol in target_elements]

    soap_descriptors = soap.create(atoms, centers=target_indices)
    np.savetxt(csv_path, soap_descriptors, delimiter=",")


def main():
    input_path = Path(input_dir)
    output_dir = Path.cwd() / "soap"
    output_dir.mkdir(exist_ok=True)

    gin_files = sorted(input_path.glob("*.gin"))

    for gin_file in gin_files:
        csv_path = output_dir / f"{gin_file.stem}.csv"
        try:
            atoms = parse_gin_to_atoms(gin_file)
            generate_soap_descriptor(atoms, csv_path, target_elements, species)
            print(f"Processed: {gin_file.name}")
        except Exception as e:
            print(f"Failed on {gin_file.name}: {e}")


if __name__ == "__main__":
    main()
