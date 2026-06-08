# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filelist", type=str, help="path to filelist.txt for the dataset storing task_doc files")
parser.add_argument("--filelist_bak", type=str, help="path to filelist.txt for the dataset storing numpy files")
parser.add_argument("--workers", type=int, default=1, help="Number of workers to use for calculations")

def count_elements_in_numpy_file(file_path_zip):
    file_path,file_path_bak = file_path_zip
    try:
        return count_elements_in_taskdoc_file(file_path)
    except:
        # Load the numpy file into a numpy array
        arr = np.load(file_path_bak)
        # Count the number of elements in the array
        shape = arr.shape
        count = np.prod(shape[:3])  # exclude spin, if present

        file_stem = file_path_bak.stem

        # Return the file path and count as a tuple
        return (file_stem, count, shape[0], shape[1], shape[2])

def count_elements_in_taskdoc_file(file_path):
    with open(file_path / "task_doc.json", 'r') as f:
        taskdoc = json.load(f)

    # Extract the number of elements from the taskdoc
    shape_x=taskdoc["calcs_reversed"][0]["input"]["parameters"]["NGXF"]
    shape_y=taskdoc["calcs_reversed"][0]["input"]["parameters"]["NGYF"]
    shape_z=taskdoc["calcs_reversed"][0]["input"]["parameters"]["NGZF"]
    count= shape_x * shape_y * shape_z
    file_stem = file_path.stem
    # Return the file path and count as a tuple
    return (file_stem, count, shape_x, shape_y, shape_z)

def count_elements_in_numpy_files(file_list_path,file_list_path_bak, workers=10):

    # Read in the list of numpy files from the text file
    with open(file_list_path, 'r') as f:
        file_list_original = f.read().splitlines()

    file_parent = Path(file_list_path).parent
    file_parent_bak = Path(file_list_path_bak).parent

    file_list = [file_parent / f"{fil}" for fil in file_list_original]
    file_list_bak = [file_parent_bak / f"{fil}.npy" for fil in file_list_original]
    
    # Create a pool of worker processes
    with Pool(workers) as p:
        # Map the file paths across the worker processes
        results = list(tqdm(p.imap(count_elements_in_numpy_file, zip(file_list,file_list_bak)), total=len(file_list)))


    # Create a pandas dataframe from the list of results
    df = pd.DataFrame(results, columns=['id', 'Count', "shape_x", "shape_y", "shape_z"])
    df.to_csv(Path(file_list_path).parent / 'probe_counts.csv', index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    
    # Define the path to the text file containing the list of numpy files
    count_elements_in_numpy_files(args.filelist, args.filelist_bak, workers=args.workers)
