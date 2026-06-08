# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
from scripts.convert_pkl_to_chgcar import deepdft_to_chgcar
import argparse
from pathlib import Path
import glob
import tqdm
from multiprocessing.pool import Pool
from pymatgen.io.vasp import Poscar
import shutil
import os


parser = argparse.ArgumentParser()
parser.add_argument('npy_dir', help='Directory with cube files in .npy format with predicted charge and/or spin densities.')
parser.add_argument('chgcar_dir', help='Directory of original CHGCARs')
parser.add_argument('out_dir', help='Directory to output CHGCAR files. Will put a CHGCAR in individual folders.')
parser.add_argument("--workers", type=int, default=1, help="Number of workers to run conversion")

def convert(npy_file, atoms_file, output_file, aug_file=None, npys_spin=None):
    '''Convert and write a single set of files'''
    if os.path.exists(atoms_file) is False:
        atom_filename=Path(atoms_file).stem
        shutil.copy(f'data/mp/{atom_filename}.pkl', atoms_file)

    chgcar, poscar = deepdft_to_chgcar(npy_file, atoms_file, aug_chgcar_file=aug_file, density_spin_file=npys_spin)
    os.makedirs(output_file, exist_ok=True)
    chgcar.write(output_file+"/CHGCAR", format="chgcar")
    poscar.write_file(output_file+"/POSCAR")
    shutil.copy(aug_file, output_file+"/CHGCAR_orig")
    #shutil.copy(aug_file.replace("CHGCAR","task_doc.json"), output_file+"/task_doc.json")

def main(npy_dir: Path, out_dir: Path, aug_chgcar_dir: Path = None, workers: int = 1):
    '''Convert and write all of the files in the specified directories'''
    npys=[]
    npys_spin=[]
    for i in npy_dir.rglob('*.npy'):
        if "spin" in str(i):
            npys_spin.append(str(Path(i)))
            # this is for only spin test, so we copy original npy here
            shutil.copy(str(aug_chgcar_dir).replace("_raw","")+"/"+Path(i).stem.replace("_spin","")+".npy",i.replace("_spin",""))
            npys.append(str(Path(i.replace("_spin",""))))
        else:
            npys.append(str(Path(i)))
            npys_spin.append(None)
    
    pkls = [p[:-4] + "_atoms.pkl" for p in npys]
    mpids = [Path(i).stem for i in npys]
    outs = [str(out_dir / f"{mpid}") for mpid in mpids]
    if aug_chgcar_dir is not None:
        augs = [str(aug_chgcar_dir / mpid / "CHGCAR") for mpid in mpids] 
    else:
        augs = [None]*len(outs)
    
    if workers <= 1:
        for f_npy, f_pkl, f_out, f_aug, f_spin in zip(npys, pkls, outs, augs, npys_spin):
            convert(f_npy, f_pkl, f_out, f_aug, f_spin)
    else:
        with Pool(workers) as p:
            p.starmap(convert, zip(npys, pkls, outs, augs, npys_spin))
    
if __name__ == "__main__":
    args = parser.parse_args()
    npy_dir = Path(args.npy_dir)
    chgcar_dir = Path(args.chgcar_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    
    main(npy_dir, out_dir, chgcar_dir, workers=args.workers)