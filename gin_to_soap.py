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
        if len(parts) == 11 and parts[0] != 'H' and parts[0] != 'He': # ignore these lines when parsing *.gin
            element = parts[0]
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            symbols.append(element)
            positions.append((x, y, z))

    atoms = Atoms(symbols=symbols, positions=positions)
    return atoms

def generate_soap_descriptor(atoms, csv_file):
    target_elements = ['Cu'] # SOAP centers: only compute on these atoms
    target_indices = [atom.index for atom in atoms if atom.symbol in target_elements]

    species = ["Cu", "Zn", "O"] # all species that may appear in the structure
    r_cut = 10.0 # A cutoff for local region in angstroms
    n_max = 2 # The number of radial basis functions
    l_max = 2 # The maximum degree of spherical harmonics

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
