# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import pandas as pd
from pathlib import Path
from src.utils.data import _load_pickled_atoms, load_numpy_density
from multiprocessing import Pool
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json
from tqdm import tqdm
import numpy as np

# TODO change if data is stored elsewhere
MP_DATASET_PATH = Path("data/mp/")
MP_DATASET_RAW_PATH = Path("data/mp_raw/")
N_TEST = 2000
N_VAL = 512


def get_atoms(member):
    return member, _load_pickled_atoms(MP_DATASET_PATH, member)

def _get_atoms_stats_multi(mpid_atoms):
    mpid, atoms = mpid_atoms
    atoms_stats=get_atoms_stats(atoms)
    atoms_stats.update(get_taskdoc(mpid))
    return mpid, atoms_stats

        
def get_atoms_stats(atoms_obj):
    # get info from atoms_obj
    chemical_formula = atoms_obj.get_chemical_formula()

    # get info from structure
    struc = AseAtomsAdaptor.get_structure(atoms_obj)
    
    # get info from composition
    composition = struc.composition
    num_atoms = composition.num_atoms
    chemical_system = composition.chemical_system
    
    # get info from reduced composition
    reduced_composition, repeats = composition.get_reduced_composition_and_factor()
    reduced_formula, repeats = composition.get_reduced_formula_and_factor()
    num_atoms_reduced = reduced_composition.num_atoms
    
    # get info from space group analyzer
    sga = SpacegroupAnalyzer(struc)
    sgn = sga.get_space_group_number()
    sgs = sga.get_space_group_symbol()
    
    return {
            "chemical_formula":chemical_formula, 
            "reduced_chemical_formula": reduced_formula,
            "chemical_system": chemical_system,
            "num_atoms": num_atoms,
            "reduced_num_atoms": num_atoms_reduced,
            "factor": repeats,
            "space_group_number": sgn,
            "space_group_symbol": sgs,
            "atomic_weight":composition.weight
            }

def get_taskdoc(mpid):
    with open(MP_DATASET_RAW_PATH/str(mpid)/'task_doc.json', "r") as f:
        taskdoc=json.load(f)
    task= get_POTCAR(taskdoc)
    task.update({"ISPIN":get_ISPIN(taskdoc)})
    task.update({"LDAU":get_LDAU(taskdoc)})
    return task

def get_ISPIN(taskdoc):
    try:
        ispin=taskdoc["calcs_reversed"][0]["input"]["incar"]["ISPIN"]
    except KeyError:
        ispin=0
    return ispin
    
def get_POTCAR(taskdoc):
    try:
        POTCAR_type=taskdoc["calcs_reversed"][0]["input"]['potcar_type']
    except KeyError:
        POTCAR_type=0
    
    try:
        POTCAR=taskdoc["calcs_reversed"][0]["input"]['potcar']
    except KeyError:
        POTCAR=0

    try:
        POTCAR_spec=[i['titel'] for i in taskdoc["calcs_reversed"][0]["input"]['potcar_spec']]
    except KeyError:
        POTCAR_spec=0

    return {"POTCAR":POTCAR, "POTCAR_type":POTCAR_type, "POTCAR_spec":POTCAR_spec}

def get_LDAU(taskdoc):
    try:
        LDAU=taskdoc["calcs_reversed"][0]["input"]['parameters']['LDAU']
    except KeyError:
        LDAU=0
    
    return 1 if LDAU else 0

def main():
    # Get file names / mpids
    filename = Path(MP_DATASET_PATH) / "filelist.txt"
    with open(filename, "r") as f:
        lines = f.readlines()
    member_list = [line.strip() for line in lines]
    
    # load the atoms objects using multiprocessing
    with Pool(20) as p:
        out = list(tqdm(p.imap(get_atoms, member_list),total=len(member_list)))
    mpid_to_atoms = dict(out)

    # get statistics on each atoms object
    with Pool(20) as p:
        out = list(tqdm(p.imap(_get_atoms_stats_multi, list(mpid_to_atoms.items())), total=len(mpid_to_atoms)))
        
    # Create output dataframe
    out_dicts = [dict(mpid=o[0], **o[1]) for o in out]
    df = pd.DataFrame(out_dicts)
    
    # Remove duplicates
    df_deduped = df.sort_values("mpid", ascending=False).sort_values("num_atoms").drop_duplicates(subset=["reduced_chemical_formula", "space_group_number"])

    # save to csv
    # row indices (df.loc) can be associated with the indices in the split files
    # to recover metadata for individual subsets
    df_deduped.sort_index().to_csv(MP_DATASET_PATH / "material_metadata.csv")
    #df_deduped=df_deduped.loc[df_deduped['LDAU']==0] # select LDAU=False calculation

    # stats on full dataset:
    print("Number of materials in directory: ", len(df))
    print("Number of de-duped materials: ", len(df_deduped))
    print("Number of distinct chemical formulae: ", len(df_deduped.chemical_formula.unique()))
    print("Number of distinct reduced formulae: ", len(df_deduped.reduced_chemical_formula.unique()))

    # shuffle
    df_deduped = df_deduped.sample(frac=1.0)  # shuffle

    # split dataset
    subsets = [70000, 30000, 10000, 3000, 1000, 300, 100, 30]
    test_set = df_deduped.iloc[:N_TEST]
    train_val_sets = df_deduped.iloc[N_TEST:]
    val_set = train_val_sets.iloc[:N_VAL]
    train_set = train_val_sets.iloc[N_VAL:]
    train_subsets = {n:train_set.sample(n=n, replace=False) for n in subsets}
    
    # First write json split file for entire dataset
    split_dict = dict(train=train_set.index.tolist(), 
                      test=test_set.index.tolist(), 
                      validation=val_set.index.tolist())
    
    with open(MP_DATASET_PATH / "split.json", 'w') as f:
        json.dump(split_dict, f)
        
    # Now do the same for subsets:
    for subset in subsets:
        split_dict = dict(train=train_subsets[subset].index.tolist(),
                          validation=val_set.index.tolist(),
                          test=test_set.index.tolist())
        
        with open(MP_DATASET_PATH / f"split_{subset}.json", 'w') as f:
            json.dump(split_dict, f)
            

if __name__ == "__main__":
    main()
        
        