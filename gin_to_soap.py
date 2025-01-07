import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

def parse_gin_to_atoms(gin_path):
    with open(gin_path, 'r') as gin_file:
        lines = gin_file.readlines()

    symbols = []
    positions = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 11 and parts[0] != 'H' and parts[0] != 'He':
            element = parts[0]
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            symbols.append(element)
            positions.append((x, y, z))

    atoms = Atoms(symbols=symbols, positions=positions)
    return atoms

def generate_soap_descriptor(atoms, csv_file):
    target_elements = ['Cu']
    target_indices = [atom.index for atom in atoms if atom.symbol in target_elements]

    species = ["Cu", "Zn", "O"]
    r_cut = 10.0
    n_max = 2
    l_max = 2

    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        compression={'mode':'mu1nu1','species_weighting':None}
    )

    soap_descriptors = soap.create(atoms, centers=target_indices)
    np.savetxt(csv_file, soap_descriptors, delimiter=',') 

def process_gin(gin_path, output_path):
    atoms = parse_gin_to_atoms(gin_path)
    generate_soap_descriptor(atoms, output_path)