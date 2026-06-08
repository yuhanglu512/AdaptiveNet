# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import itertools
from pathlib import Path

import torch

import pandas as pd
import numpy as np
import pickle

def split_batch(batch, max_predict_batch_probes):
    """split batches into sub batches with fewer probes"""
    num_sections = torch.ceil(
        torch.div(batch["num_probes"].item(), max_predict_batch_probes)
    ).long()
    split_keys = ["probe_target", "probe_xyz", "probe_edges", "probe_edges_displacement"]
    split_dict = {}
    
    for k in split_keys[:2]:
        split_dict[k] = torch.tensor_split(batch[k], num_sections, 1)
        del batch[k]
    
    section_lens = [sec.shape[1] for sec in split_dict["probe_target"]]

    # get indices into probe_edges for each set of probes
    # ie. which probe is each edge referring to?
    start_idx = 0
    pe_idx = batch["probe_edges"][...,1].squeeze()
    pe_section_inds = []
    new_probe_edges = []
    new_probe_edges_displacement = []
    for i, length in enumerate(section_lens):
        end_idx = start_idx + length
        in_section = torch.bitwise_and(pe_idx >= start_idx, pe_idx < end_idx)
        pe_section_inds = in_section.nonzero().squeeze(dim=1)
        section_probe_edges = batch["probe_edges"][:,pe_section_inds,:]
        section_probe_edges[:,:,1] -= start_idx # need to reindex to match probe_xyz and displacement
        new_probe_edges.append(section_probe_edges)
        new_probe_edges_displacement.append(
            batch["probe_edges_displacement"][:,pe_section_inds,:]
            )
        start_idx = end_idx

    split_dict["probe_edges"] = new_probe_edges
    split_dict["probe_edges_displacement"] = new_probe_edges_displacement
    del batch["probe_edges"]
    del batch["probe_edges_displacement"]

    # run predictions with each sec
    all_loss, all_preds, all_targets = [], [], []
    for i in range(num_sections):
        this_batch = dict(batch)
        for k in split_keys:
            this_batch[k] = split_dict[k][i].clone()
        this_batch["num_probes"][0] = section_lens[i]
        this_batch["num_probe_edges"][0] = this_batch["probe_edges"].shape[1]
        yield this_batch


def compute_nmape_components(pred, target):
    pred, target = _check_cube_aligned(pred, target)

    diff = target - pred
    abs_diff = torch.abs(diff)
    return abs_diff.sum().cpu().numpy().item(), torch.abs(target).sum().cpu().numpy().item()


def compute_nmape(pred, target):
    pred, target = _check_cube_aligned(pred, target)
    
    diff = target-pred
    abs_diff = np.abs(diff)
    return abs_diff.sum() / np.abs(target).sum() * 100.

def _check_cube_aligned(pred, target):
    assert pred.shape == target.shape, "Prediction and target cubes are misaligned"
    if len(pred.shape) == 4:
        pred = pred[..., 0]
        target = target[..., 0]
    return pred, target


def save_preds(preds, save_dir):
    df = pd.DataFrame(preds)
    # compute nmape by combinine parts of saved outputs
    df_agg = df.groupby(by="filename").agg(
        diff_sum=("diff_sum", sum), 
        targ_sum=("target_sum", sum),
        time=("time", sum), # total time should be sum of partial times
    )
    
    nmape_forumla = lambda x: x["diff_sum"] / x["targ_sum"] * 100.
    df_agg["nmape"] = df_agg.apply(nmape_forumla, axis=1)

    # save times to csv
    df_time = df_agg[["time"]]
    df_time.to_csv(save_dir / "test_set_time.csv")
                
    # save nmapes to csv
    df_agg = df_agg[["nmape"]]
    df_agg.to_csv(save_dir / "test_set_nmape.csv")

    # save summary statistics for plotting (scripts/plot_nmape_dataset_size.py)
    summary_df =  pd.DataFrame(
        {
            "checkpoint":"", 
            "test/IntegralNormalizedMeanAbsoluteError": df_agg.nmape.mean(),
            "nmape_median": df_agg.nmape.median(),
            "nmape_max": df_agg.nmape.max(),
            "nmape_min": df_agg.nmape.min(),
            "num_materials": len(df_agg)
        },
        index = [0]
    )
    summary_df.to_csv(save_dir / "test_statistics.csv")

def convert_to_serializable(obj):
    """
    Recursively traverse a nested data structure (lists, tuples, dicts, sets)
    and convert all PyTorch tensors (both CPU and CUDA) to NumPy arrays.
    All other types (strings, numbers, booleans, None, existing numpy arrays)
    are left unchanged. The returned structure can be safely pickled.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return {convert_to_serializable(item) for item in obj}
    else:
        # For any other type (e.g., custom objects, functions), return as is.
        return obj

def save_results(preds, save_dir):
    """Save predictions to a pickle file, converting tensors to numpy arrays for serialization."""
    serializable_preds = convert_to_serializable(preds)
    with open(save_dir / "predictions.pkl", "wb") as f:
        pickle.dump(serializable_preds, f)


def combine_partial_cubes(partial_dir, save_dir):
    # sort by filename
    partial_files = sorted([(f.stem.split("_"), f) for f in partial_dir.iterdir() if "pred" in f.stem])
    
    # 修正：提取真实文件名和 offset
    filename_offsets = []
    for parts, filepath in partial_files:
        # 找到 'offset' 的位置
        try:
            idx = parts.index('offset')
        except ValueError:
            # 如果没有 'offset' 字段，则跳过或按原逻辑处理（兼容旧格式）
            continue
        # 文件名是 'offset' 之前的所有部分用 '_' 连接
        filename = '_'.join(parts[:idx])
        # offset 是 'offset' 后面的一个元素
        offset = int(parts[idx+1])
        filename_offsets.append((filename, offset, filepath))
    
    # group by filename
    for filename, group in itertools.groupby(filename_offsets, key=lambda fo: fo[0]):
        # sort by offset 
        sorted_group = sorted(group, key=lambda fof: fof[1])
        preds = [torch.load(f, weights_only=False) for _, _, f in sorted_group]
        grid_shape = preds[0]["grid_shape"].numpy()
        # combine densities and reshape
        densities = [p["density"] for p in preds]
        cube = np.concatenate(densities, 1).reshape(grid_shape)
        np.save(save_dir / f"{filename[5:]}.npy", cube)