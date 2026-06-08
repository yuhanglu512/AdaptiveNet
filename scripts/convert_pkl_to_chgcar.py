# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pymatgen.io.vasp import Chgcar
from ase.calculators.vasp import VaspChargeDensity

from src.utils.data import load_atoms_file, load_density_file

from sklearn.metrics import r2_score

parser = argparse.ArgumentParser()
parser.add_argument("--density_file", type=str, help="path to .npy density file")
parser.add_argument("--atoms_file", type=str, help="path to .pkl atoms file")
parser.add_argument("--output_file", type=str, help="path to CHGCAR file to save out")
parser.add_argument("--aug_chgcar_file", type=str, default=None, help="path to original CHGCAR file to retrieve augmentation")


def deepdft_to_chgcar(density_file, atoms_file, aug_chgcar_file=None, density_spin_file=None) -> VaspChargeDensity:
    density = load_density_file(density_file)
    atoms = load_atoms_file(atoms_file)
    

    # retrieve augmentation, if requested
    if aug_chgcar_file is not None:
        aug = Chgcar.from_file(aug_chgcar_file).data_aug
        density_origin=Chgcar.from_file(aug_chgcar_file)
    else:
        aug = None
    
    if density_spin_file is not None:
        if density.ndim == 4:
            density = density[...,0]
        density_spin = load_density_file(density_spin_file)
        if density_spin.ndim == 4:
            density_spin = density_spin[...,1]
        density = np.stack([density, density_spin], axis=-1)
        
    # extract spin, if available
    if len(density.shape) == 4:  # implies a spin channel exists
        charge_grid = density[..., 0]
        spin_grid = density[..., 1]
    else:
        charge_grid = density
        
    # create Chgcar object
    print(density_file, np.abs(density*density_origin.structure.volume-density_origin.data['total']).sum()/density_origin.data['total'].sum())
    vcd = VaspChargeDensity(filename=None)
    vcd.atoms.append(atoms)
    vcd.chg.append(charge_grid)
    if density_spin_file is not None:
        vcd.chgdiff.append(spin_grid)
    if aug is not None:
        vcd.aug = "".join(aug["total"])
        if density_spin_file is not None:
            vcd.augdiff = "".join(aug["diff"])
        
    return vcd, density_origin.poscar

if __name__ == "__main__":
    args = parser.parse_args()
    
    chgcar = deepdft_to_chgcar(args.density_file, args.atoms_file, aug_chgcar_file=args.aug_chgcar_file)
    chgcar.write(args.output_file, format="chgcar")