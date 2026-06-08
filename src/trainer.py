# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import os
from pathlib import Path
import logging
import time
import shutil
import pynvml# Must call this first
import yaml

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from e3nn import o3

from .charge3net.models.densitymodel import PainnDensityModel
from .charge3net.models.scheduler import PowerDecayScheduler
from .utils import predictions as pred_utils

import threading
import copy
from concurrent.futures import ThreadPoolExecutor

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        log_dir: str,
        gpu_id: int,
        global_rank: int,
        load_checkpoint_path=None,
        log_steps=50,
    ):
        self.local_rank = gpu_id
        self.global_rank = global_rank
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.start_epoch = 0
        self.step = 0
        self.best_nmape = float("inf")

        if load_checkpoint_path is not None:
            assert Path(load_checkpoint_path).exists(), f"file {load_checkpoint_path} does not exist"
            print(f"Loading checkpoint {load_checkpoint_path}")
            self._load_checkpoint(load_checkpoint_path)

        self.log_dir = Path(log_dir)
        
        # add slurm job to log directory (if using slurm)
        if "SLURM_JOB_ID" in os.environ:
            job_id = os.environ["SLURM_JOB_ID"]
            if "SLURM_ARRAY_TASK_ID" in os.environ:
                job_id += '_' + os.environ['SLURM_ARRAY_TASK_ID']
            self.log_dir = self.log_dir / job_id

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_steps = log_steps
        self.checkpoint_path = self.log_dir / "checkpoint.pt"

        if self.local_rank == 0 and self.global_rank == 0:
            # setup tensorboard and print logging
            self.tensorboard = SummaryWriter(self.log_dir)
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
                handlers=[logging.FileHandler(self.log_dir / "log.txt"), logging.StreamHandler()],
            )

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{self.local_rank}")
        if "pytorch-lightning_version" in checkpoint: return self._load_checkpoint_legacy(checkpoint)
        try:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = checkpoint["epoch"]
            self.step = checkpoint["step"]
            print("load checkpoint perfectly")
        except:
            self._load_checkpoint_without_iterative_layer(checkpoint['model'])
            self._create_new_optimizer()
            self._create_new_scheduler()
            print("load checkpoint without iterative layer")
        

    def _load_checkpoint_legacy(self, checkpoint):
        # loads old lightning checkpoints
        self.model.load_state_dict({k.replace("network.", ""): v for k, v in checkpoint["state_dict"].items()})
        self.optimizer.load_state_dict(checkpoint["optimizer_states"][0])
        self.scheduler.load_state_dict(checkpoint["lr_schedulers"][0])
        self.start_epoch = checkpoint["epoch"]
        self.step = checkpoint["global_step"]
        print("load checkpoint from legacy pytorch-lightning format")

    def _load_checkpoint_without_iterative_layer(self, checkpoint):
        current_model_state = self.model.state_dict()
        current_model_state.update(checkpoint)

    def _create_new_optimizer(self):
        lr = self.optimizer.param_groups[0]['initial_lr']
        self.optimizer.param_groups.clear()

        # 添加分组参数
        self.optimizer.add_param_group({
            'params': [p for n, p in self.model.named_parameters() 
                       if ('convolutions.4.' in n or 'residual' in n) and p.requires_grad],
            'lr': lr
        })

        self.optimizer.add_param_group({
            'params': [p for n, p in self.model.named_parameters() 
                       if not ('convolutions.4.' in n or 'residual' in n) and p.requires_grad],
            'lr': lr
        })

    def _create_new_scheduler(self):
        alpha = self.scheduler.alpha if hasattr(self.scheduler, 'alpha') else 0.99
        beta = self.scheduler.beta if hasattr(self.scheduler, 'beta') else 3e3
        self.scheduler = PowerDecayScheduler(self.optimizer, alpha=alpha, beta=beta)

    def _save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "step": self.step,
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_nmape": self.best_nmape,
        }
        torch.save(checkpoint, self.checkpoint_path)

    #@torch.no_grad()
    def _train_epoch(self, spin=False, only_spin=False, mag=False, q=False, density = False):
        iter = 1
        #results = []
        for batch in self.train_dl:
            batch = self._to_device(batch)
            sign = torch.sign(torch.randn(1).to(self.local_rank))
            if q:
                pass
            elif mag:
                batch['node_vector'] = batch['magmom'] * sign
                batch['magmom'] = batch['magmom'] * sign
            elif only_spin:
                batch['node_vector'] = batch['node_vector'] * sign
                batch['probe_target'][...,1] = batch['probe_target'][...,1] * sign
            if q and not density:
                output, iter_list, residue_scale = self.model(batch, max(0.96**(self.step/2000),0.005))
                #results.append({"preds":output.detach().cpu().numpy(), "targets":batch['q'].reshape(-1).detach().cpu().numpy(), "filename":batch['filename'], "num_atoms":batch['num_nodes'].detach().cpu().numpy()})
            else:
                output = self.model(batch)
            if q and not density:
                loss = self.criterion(output, batch['q'].reshape(-1))
            elif mag:
                loss = self.criterion(output,batch['magmom'])
            elif only_spin:
                loss = self.criterion(output[...,1], batch["probe_target"][...,1])
            elif spin:
                loss = self.criterion(output, batch["probe_target"])
            else:
                loss = self.criterion(output, batch["probe_target"][...,0])
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            #if q:
            #    loss.backward()
            #    iter += 1
            #    if iter % 16 == 0:
            #        self.optimizer.step()
            #        self.scheduler.step()
            #        self.optimizer.zero_grad()
            #        self.step += 1
            #    else:
            #        continue
            #else:
            #    self.optimizer.zero_grad()
            #    loss.backward()
            #    self.optimizer.step()
            #    self.scheduler.step()
            #    self.step += 1

            if self.local_rank == 0 and self.global_rank == 0:
                # logging/checkpointing on rank 0 (otherwise would repeat on each node)
                if self.step % self.log_steps == 0:
                    last_lr = self.optimizer.param_groups[-1]['lr']
                    if q and not density:
                        logging.info(f"step: {self.step} train/loss: {loss.item():.6f} lr: {last_lr:.6f} iters: {torch.mean(iter_list.float()).item():.2f} residue_scale: {residue_scale.item():.6f}")
                        self.tensorboard.add_scalar("train/loss", loss.item(), global_step=self.step)
                        self.tensorboard.add_scalar("train/iters", torch.mean(iter_list.float()).item(), global_step=self.step)
                        self.tensorboard.add_scalar("train/residue_scale", residue_scale.item(), global_step=self.step)
                    elif only_spin:
                        logging.info(f"step: {self.step} train/spin_loss: {loss.item():.6f} lr: {last_lr:.6f}")
                        self.tensorboard.add_scalar("train/spin_loss", loss.item(), global_step=self.step)
                    else:
                        logging.info(f"step: {self.step} train/loss: {loss.item():.6f} lr: {last_lr:.6f}")
                        self.tensorboard.add_scalar("train/loss", loss.item(), global_step=self.step)
                    self.tensorboard.add_scalar("lr", last_lr, global_step=self.step)
                    self.tensorboard.flush()

        #pred_utils.save_results(results, self.log_dir)

            #torch.cuda.empty_cache()
            #if memory_info.free < 2*1024**3:
            #    # clear memory if we are using a lot of it
            #    torch.cuda.empty_cache()


    def _to_device(self, batch, key=None):
        if isinstance(batch, torch.Tensor):
            # Moves tensor to proper device
            #if key== "magmom":
                #return batch.to(self.local_rank) if torch.sum(batch) >= 0 else -batch.to(self.local_rank)
            return batch.to(self.local_rank)
        elif isinstance(batch, (list,tuple)):
            if isinstance(batch[0], torch.Tensor):
                if key == "magmom":
                    return torch.concat([self._to_device(b,key) for b in batch], dim=0)
                else:
                    return torch.concat([self._to_device(b) for b in batch], dim=0)
            else:
                return [self._to_device(b) for b in batch]
        elif isinstance(batch, dict):
            return {k: self._to_device(v,k) for k, v in batch.items()}
        else:
            return batch

    @torch.no_grad()
    def _valid_epoch(self,spin=False, only_spin=False, mag=False, q=False, density = False):
        total_nmape, total_count = 0.0, 0
        #results=[]
        if q:
            average_iter = []
            residue_list = []
        for batch in self.valid_dl:
            batch = self._to_device(batch)
            if mag:
                batch['node_vector'] = batch['magmom']
            if q and not density:
                preds, iter_list, residue_scale = self.model(batch, max(0.96**(self.step/2000),0.005))
                #results.append({"preds":preds.detach().cpu().numpy(), "targets":batch['q'].reshape(-1).detach().cpu().numpy(), "filename":batch['filename'], "num_atoms":batch['num_nodes'].detach().cpu().numpy()})
            else:
                preds = self.model(batch)
            if q and not density:
                diff = batch['q'] - preds
                nmape = torch.abs(diff) / torch.abs(batch['q']) * 100.
                num_iter = torch.mean(iter_list.float()).item()
                average_iter.append(num_iter)
                residue_list.append(residue_scale.item())
            elif mag:
                diff = batch["magmom"] - preds
                nmape = 2*torch.abs(diff) / (torch.abs(preds)+torch.abs(batch["magmom"])) * 100.
            elif only_spin:
                diff = batch["probe_target"][:,:,1] - preds[:,:,1]
                nmape = torch.abs(diff).sum(1) / torch.abs(batch["probe_target"][:,:,1]).sum(1) * 100.
            elif spin:
                diff = batch["probe_target"] - preds
                nmape = torch.abs(diff).sum(1) / torch.abs(batch["probe_target"]).sum(1) * 100.
            else:
                diff = batch["probe_target"][...,0] - preds
                nmape = torch.abs(diff).sum(1) / torch.abs(batch["probe_target"][...,0]).sum(1) * 100.
            total_count += nmape.shape[0]
            total_nmape += nmape.sum(0)

        nmape = total_nmape / total_count
        if q and not density:
            return {
                "val/IntegralNormalizedMeanAbsoluteError": nmape.item(),
                "val/average_iters": torch.mean(torch.tensor(average_iter)).item(),
                "val/average_residue": torch.mean(torch.tensor(residue_list)).item()
            }
        elif mag:
            return {"val/IntegralNormalizedMeanAbsoluteError": nmape.item()}
        elif only_spin:
            return {"val_spin/IntegralNormalizedMeanAbsoluteError": nmape.item()}
        elif spin:
            return  {
                "val/IntegralNormalizedMeanAbsoluteError": nmape[0].item(),
                "val_spin/IntegralNormalizedMeanAbsoluteError": nmape[1].item(),
            }
        else:
            return {"val/IntegralNormalizedMeanAbsoluteError": nmape.item()}
        

    def fit(self, train_dl, valid_dl, steps,spin=False, only_spin=False, mag=False, q=False, density =False):

        self.train_dl = train_dl
        #self.valid_dl = [batch for batch in valid_dl] # cache validation dataloader
        self.valid_dl = valid_dl

        epoch = self.start_epoch
        while self.step < steps:
            self.train_dl.sampler.set_epoch(epoch)
            self.model.train()
            self._train_epoch(spin=spin, only_spin=only_spin, mag=mag, q=q, density=density)
            torch.cuda.empty_cache()
            self.model.eval()
            metrics = self._valid_epoch(spin=spin, only_spin=only_spin,mag=mag , q=q, density=density)
            torch.cuda.empty_cache()

            if self.local_rank == 0 and self.global_rank == 0:
                # logging/checkpointing on rank 0 (otherwise would repeat on each node)
                logging.info(f"step: {self.step} {' '.join(f'{k}: {v:.4f}' for k, v in metrics.items())}")
                for k, v in metrics.items(): self.tensorboard.add_scalar(k, v, global_step=self.step)
                self.tensorboard.flush()
                
                if only_spin and not mag:
                    if metrics["val_spin/IntegralNormalizedMeanAbsoluteError"] <= self.best_nmape:
                        self.best_nmape = metrics["val_spin/IntegralNormalizedMeanAbsoluteError"]
                        self._save_checkpoint(epoch)
                else:
                    if metrics["val/IntegralNormalizedMeanAbsoluteError"] <= self.best_nmape:
                        self.best_nmape = metrics["val/IntegralNormalizedMeanAbsoluteError"]
                        self._save_checkpoint(epoch)



    @torch.no_grad()
    def test(self, test_dl, cube_dir=None, max_predict_batch_probes=2500, mag=False, spin=False, q=False, density=False):
        results = []
        self.model.eval()
        tmp_nmape_dir = self.log_dir / ".tmp_nmape"
        tmp_nmape_dir.mkdir(exist_ok=True)

        if cube_dir is not None:
            cube_dir = Path(cube_dir) / "cubes"
            cube_dir.mkdir(exist_ok=True, parents=True)
            tmp_cube_dir = self.log_dir / ".tmp_cubes"
            tmp_cube_dir.mkdir(exist_ok=True)

        for i, batch in enumerate(test_dl):
            if self.local_rank == 0 and self.global_rank == 0:
                logging.info(f"testing {i} / {len(test_dl)} (rank 0)")

            # write out nmape (or partial nmape)
            predictions = self._test_step(batch, max_predict_batch_probes,mag=mag, q=q, density = density)

            # TODO: handle spin density here (should only compute with preds[:, :, 0] and targets[:, :, 0]
            if predictions['preds'].dim() == 3 and spin is False:
                diff_sum, targ_sum = pred_utils.compute_nmape_components(predictions["preds"][:, :, 0], predictions["targets"][:, :, 0])
            elif predictions['preds'].dim() == 3 and spin:
                diff_sum, targ_sum = pred_utils.compute_nmape_components(predictions["preds"][:, :, 1], predictions["targets"][:, :, 1])
            elif predictions['preds'].dim() == 2 and predictions['targets'].dim() == 3:
                diff_sum, targ_sum = pred_utils.compute_nmape_components(predictions["preds"], predictions["targets"][...,0])
            else:
                diff_sum, targ_sum = pred_utils.compute_nmape_components(predictions["preds"], predictions["targets"])
            if mag:
                out_dict = {
                    "diff_sum": diff_sum,
                    "target_sum": targ_sum,
                    "num_probes": torch.prod(torch.tensor(predictions["preds"].shape)).item(),
                    "filename": predictions["filename"],
                    #"probe_offset": predictions["probe_offset"].item(),
                    #"grid_shape": predictions["grid_shape"].tolist(),
                    "time": predictions["time"]
                }
            else:
                out_dict = {
                    "diff_sum": diff_sum,
                    "target_sum": targ_sum,
                    "num_probes": torch.prod(torch.tensor(predictions["preds"].shape)).item(),
                    "filename": predictions["filename"],
                    "probe_offset": predictions["probe_offset"].item(),
                    "grid_shape": predictions["grid_shape"].tolist(),
                    "time": predictions["time"]
                }

            #torch.save(out_dict, tmp_nmape_dir / f"pred_{predictions['filename']}_offset_{predictions['probe_offset']}.pt")

            if cube_dir is None:
                results.append(predictions)
                continue
            
            # write out cubes (or partial cubes)
            preds = predictions["preds"].cpu().numpy()
            grid_shape = predictions["grid_shape"].cpu()

            if not predictions["partial"]:
                cube = preds.reshape(grid_shape.numpy())
                if spin:
                    np.save(cube_dir / f"{predictions['filename']}_spin.npy", cube)
                else:
                    np.save(cube_dir / f"{predictions['filename']}.npy", cube)
            else:
                out_dict = {
                    "density": preds,
                    "grid_shape": grid_shape,
                    "probe_offset": predictions["probe_offset"].item(),
                    "filename": predictions["filename"],
                }
                if spin:
                    torch.save(out_dict, tmp_cube_dir / f"pred_{predictions['filename']}_spin_offset_{predictions['probe_offset']}.pt")
                else:
                    torch.save(out_dict, tmp_cube_dir / f"pred_{predictions['filename']}_offset_{predictions['probe_offset']}.pt")

        
        # save results
        if self.local_rank == 0 and self.global_rank == 0 and cube_dir is None:
            pred_utils.save_results(results, self.log_dir)

        # End of testing, once all nodes have finished
        torch.distributed.barrier()
        if self.local_rank == 0 and self.global_rank == 0:
            preds = [torch.load(f, map_location=self.local_rank) for f in tmp_nmape_dir.iterdir() if "pred" in f.name]
            pred_utils.save_preds(preds, self.log_dir)
            if cube_dir is not None:
                pred_utils.combine_partial_cubes(tmp_cube_dir, cube_dir)

            # cleanup
            shutil.rmtree(tmp_nmape_dir)
            if cube_dir is not None:
                shutil.rmtree(tmp_cube_dir)
                        
    def _test_step(self, batch, max_predict_batch_probes,mag=False, q=False, density=False):
        start_time = time.time()
            
        # see NOTE below
        batch = self._to_device(batch)

        if not mag and batch["num_probes"] > max_predict_batch_probes:            
            all_loss, all_preds, all_targets, atom_repr = [], [], [], None


            for i, sub_batch in enumerate(pred_utils.split_batch(batch, max_predict_batch_probes)):
                # NOTE: much slower to move sub-batch one at a time instead of all at once above
                # sub_batch = self._to_device(sub_batch)

                # if self.local_rank == 0 and self.global_rank == 0:
                #     logging.info(f"sub-batch {i} (rank 0)")

                # atom representations only need to be calculated once
                if atom_repr is None:
                    atom_repr = self.model.module.atom_model(sub_batch)
                if isinstance(self.model.module, PainnDensityModel):
                    # PaiNN takes (scalar, vector) tuple as two args
                    outputs = self.model.module.probe_model(sub_batch, *atom_repr)
                else:
                    outputs = self.model.module.probe_model(sub_batch, atom_repr)

                loss = self.criterion(outputs, sub_batch["probe_target"])
                all_loss.append(loss)
                all_preds.append(outputs)
                all_targets.append(sub_batch["probe_target"])

            preds = torch.cat(all_preds, dim=1)
            loss = torch.mean(torch.tensor(all_loss))
            targets = torch.cat(all_targets, dim=1) 

        else:
            if q and not density:
                preds, iter, residual_scale = self.model(batch, 0.005)
                targets = batch['q'].reshape(-1)
                loss = self.criterion(preds, targets)
            else:
                preds = self.model(batch)
                targets = batch["probe_target"] if not mag else batch["magmom"]
                loss = self.criterion(preds, targets[...,0])

        if q and not density:
            return {
                "loss": loss, 
                "preds": preds, 
                "targets": targets,
                "iter": iter,
                "num_atoms": len(preds),
                "volume":torch.linalg.det(batch["cell"]).item(),
                "filename":batch["filename"][0],
                "time": time.time() - start_time + batch["load_time"][0],
            }
        elif mag:
            return {
                "loss": loss, 
                "preds": preds, 
                "targets": targets, 
                "filename":batch["filename"][0],
                #"probe_offset": batch["probe_offset"][0],
                #"grid_shape": batch["grid_shape"][0],
                #"partial": batch["partial"][0],
                "time": time.time() - start_time + batch["load_time"][0],
            }
        else:
            return {
                "loss": loss, 
                "preds": preds, 
                "targets": targets, 
                "filename":batch["filename"][0],
                "probe_offset": batch["probe_offset"][0],
                "grid_shape": batch["grid_shape"][0],
                "partial": batch["partial"][0],
                "time": time.time() - start_time + batch["load_time"][0],
            }
        
    @torch.no_grad()
    def inference(self, test_dl, cube_dir=None, model1=None, model2=None, max_predict_batch_probes=5000):
        results = []
        self.model1 = model1.eval()
        self.model2 = model2.eval()
        tmp_nmape_dir = self.log_dir / ".tmp_nmape"
        tmp_nmape_dir.mkdir(exist_ok=True)

        if cube_dir is not None:
            cube_dir = Path(cube_dir) / "cubes"
            cube_dir.mkdir(exist_ok=True, parents=True)
            tmp_cube_dir = self.log_dir / ".tmp_cubes"
            tmp_cube_dir.mkdir(exist_ok=True)

        time1=time.time()
        print(len(test_dl))

        for i, batch in enumerate(test_dl):
            if self.local_rank == 0 and self.global_rank == 0:
                logging.info(f"testing {i} / {len(test_dl)} (rank 0)")

            # write out nmape (or partial nmape)
            predictions = self._inference_step(batch, max_predict_batch_probes)
            #predictions = self._inference_step_parallel(batch, max_predict_batch_probes)

            # TODO: handle spin density here (should only compute with preds[:, :, 0] and targets[:, :, 0]
            out_dict = {
                "num_probes": torch.prod(torch.tensor(predictions["preds"].shape)).item(),
                "filename": predictions["filename"],
                "probe_offset": predictions["probe_offset"].item(),
                "grid_shape": predictions["grid_shape"].tolist(),
                "time": predictions["time"],
            }

            torch.save(out_dict, tmp_nmape_dir / f"pred_{predictions['filename']}_offset_{predictions['probe_offset']}.pt")

            if cube_dir is None:
                results.append(predictions)
                continue

            # write out cubes (or partial cubes)
            preds = predictions["preds"].cpu().numpy()
            grid_shape = predictions["grid_shape"].cpu()

            if not predictions["partial"]:
                cube = preds.reshape(grid_shape.numpy())
                np.save(cube_dir / f"{predictions['filename']}.npy", cube)
            else:
                out_dict = {
                    "density": preds,
                    "grid_shape": grid_shape,
                    "probe_offset": predictions["probe_offset"].item(),
                    "filename": predictions["filename"],
                }
                torch.save(out_dict, tmp_cube_dir / f"pred_{predictions['filename']}_offset_{predictions['probe_offset']}.pt")

        print("total time:", time.time()-time1)
        torch.distributed.barrier()
        if self.local_rank == 0:
            if cube_dir is None:
                pred_utils.save_results(results, self.log_dir)
            if cube_dir is not None:
                pred_utils.combine_partial_cubes(tmp_cube_dir, cube_dir)
            shutil.rmtree(tmp_nmape_dir)
            if cube_dir is not None:
                shutil.rmtree(tmp_cube_dir)
                
    def _inference_step(self,batch, max_predict_batch_probes):
        start_time = time.time()

        # see NOTE below
        batch = self._to_device(batch)

        if batch["num_probes"] > max_predict_batch_probes:            
            all_loss, all_preds, all_targets, atom_repr = [], [], [], None


            for i, sub_batch in enumerate(pred_utils.split_batch(batch, max_predict_batch_probes)):
                # NOTE: much slower to move sub-batch one at a time instead of all at once above
                # sub_batch = self._to_device(sub_batch)

                # if self.local_rank == 0 and self.global_rank == 0:
                #     logging.info(f"sub-batch {i} (rank 0)")

                # atom representations only need to be calculated once
                if atom_repr is None:
                    preds_q, iter, residual_scale = self.model1(batch, 0.005)
                    midtime = time.time()
                    sub_batch['q'] = preds_q
                    atom_repr = self.model2.atom_model(sub_batch)
                if isinstance(self.model2.probe_model, PainnDensityModel):
                    # PaiNN takes (scalar, vector) tuple as two args
                    outputs = self.model2.probe_model(sub_batch, *atom_repr)
                else:
                    outputs = self.model2.probe_model(sub_batch, atom_repr)

                all_preds.append(outputs)

            preds = torch.cat(all_preds, dim=1)
            print("time for q:", midtime-start_time)

            return {
                "preds": preds, 
                "filename":batch["filename"][0],
                "probe_offset": batch["probe_offset"][0],
                "grid_shape": batch["grid_shape"][0],
                "partial": batch["partial"][0],
                "time": time.time() - start_time + batch["load_time"][0],
            }


        else:
            preds_q, iter, residual_scale = self.model1(batch, 0.005)
            targets_q = batch['q'].reshape(-1)
            batch['q'] = preds_q
            preds = self.model2(batch)

        return {
            "preds": preds, 
            "preds_q": preds_q,
            "filename":batch["filename"][0],
            "probe_offset": batch["probe_offset"][0],
            "grid_shape": batch["grid_shape"][0],
            "partial": batch["partial"][0],
            "time": time.time() - start_time + batch["load_time"][0],
        }
    
        
    @torch.no_grad()
    def test_whole(self, test_dl, cube_dir=None, model1=None, model2=None, max_predict_batch_probes=2500):
        results = []
        self.model1 = model1.eval()
        self.model2 = model2.eval()
        tmp_nmape_dir = self.log_dir / ".tmp_nmape"
        tmp_nmape_dir.mkdir(exist_ok=True)

        if cube_dir is not None:
            cube_dir = Path(cube_dir) / "cubes"
            cube_dir.mkdir(exist_ok=True, parents=True)
            tmp_cube_dir = self.log_dir / ".tmp_cubes"
            tmp_cube_dir.mkdir(exist_ok=True)

        for i, batch in enumerate(test_dl):
            if self.local_rank == 0 and self.global_rank == 0:
                logging.info(f"testing {i} / {len(test_dl)} (rank 0)")

            # write out nmape (or partial nmape)
            predictions = self._test_step_whole(batch, max_predict_batch_probes)

            # TODO: handle spin density here (should only compute with preds[:, :, 0] and targets[:, :, 0]
            diff_sum, targ_sum = pred_utils.compute_nmape_components(predictions["preds"], predictions["targets"])
            out_dict = {
                "diff_sum": diff_sum,
                "target_sum": targ_sum,
                "num_probes": torch.prod(torch.tensor(predictions["preds"].shape)).item(),
                "filename": predictions["filename"],
                "probe_offset": predictions["probe_offset"].item(),
                "grid_shape": predictions["grid_shape"].tolist(),
                "time": predictions["time"],
            }

            torch.save(out_dict, tmp_nmape_dir / f"pred_{predictions['filename']}_offset_{predictions['probe_offset']}.pt")

            if cube_dir is None:
                results.append(predictions)
                continue

        # write out cubes (or partial cubes)
            preds = predictions["preds"].cpu().numpy()  
            grid_shape = predictions["grid_shape"].cpu()

            if not predictions["partial"]:
                cube = preds.reshape(grid_shape.numpy())
                np.save(cube_dir / f"{predictions['filename']}.npy", cube)
            else:
                out_dict = {
                    "density": preds,
                    "grid_shape": grid_shape,
                    "probe_offset": predictions["probe_offset"].item(),
                    "filename": predictions["filename"],
                }
                torch.save(out_dict, tmp_cube_dir / f"pred_{predictions['filename']}_offset_{predictions['probe_offset']}.pt")

        # save results
        if self.local_rank == 0 and self.global_rank == 0 and cube_dir is None:
            pred_utils.save_results(results, self.log_dir)

            # End of testing, once all nodes have finished
            torch.distributed.barrier()
            if self.local_rank == 0 and self.global_rank == 0:
                preds = [torch.load(f, map_location=self.local_rank) for f in tmp_nmape_dir.iterdir() if "pred" in f.name]
                pred_utils.save_preds(preds, self.log_dir)
                if cube_dir is not None:
                    pred_utils.combine_partial_cubes(tmp_cube_dir, cube_dir)

                # cleanup
                shutil.rmtree(tmp_nmape_dir)
                if cube_dir is not None:
                    shutil.rmtree(tmp_cube_dir)

    def _test_step_whole(self,batch, max_predict_batch_probes):
        start_time = time.time()

        # see NOTE below
        batch = self._to_device(batch)

        if batch["num_probes"] > max_predict_batch_probes:            
            all_loss, all_preds, all_targets, atom_repr = [], [], [], None


            for i, sub_batch in enumerate(pred_utils.split_batch(batch, max_predict_batch_probes)):
                # NOTE: much slower to move sub-batch one at a time instead of all at once above
                # sub_batch = self._to_device(sub_batch)

                # if self.local_rank == 0 and self.global_rank == 0:
                #     logging.info(f"sub-batch {i} (rank 0)")

                # atom representations only need to be calculated once
                if atom_repr is None:
                    preds_q, iter, residual_scale = self.model1(batch, 0.005)
                    midtime = time.time()
                    sub_batch['q'] = preds_q
                    atom_repr = self.model2.atom_model(sub_batch)
                if isinstance(self.model2.probe_model, PainnDensityModel):
                    # PaiNN takes (scalar, vector) tuple as two args
                    outputs = self.model2.probe_model(sub_batch, *atom_repr)
                else:
                    outputs = self.model2.probe_model(sub_batch, atom_repr)

                loss = self.criterion(outputs, sub_batch["probe_target"])
                all_loss.append(loss)
                all_preds.append(outputs)
                all_targets.append(sub_batch["probe_target"])

            preds = torch.cat(all_preds, dim=1)
            loss = torch.mean(torch.tensor(all_loss))
            targets = torch.cat(all_targets, dim=1) 

            print(f"time for model1: {midtime-start_time:.4f}, time for model2: {time.time()-midtime:.4f}")

            return {
                "loss": loss, 
                "preds": preds, 
                "targets": targets, 
                "filename":batch["filename"][0],
                "probe_offset": batch["probe_offset"][0],
                "grid_shape": batch["grid_shape"][0],
                "partial": batch["partial"][0],
                "time": time.time() - start_time + batch["load_time"][0],
            }


        else:
            preds_q, iter, residual_scale = self.model1(batch, 0.005)
            targets_q = batch['q'].reshape(-1)
            batch['q'] = preds_q
            preds = self.model2(batch)
            targets = batch["probe_target"][...,0]
            loss = self.criterion(preds, targets)

        return {
            "loss": loss, 
            "preds": preds, 
            "targets": targets, 
            "iter": iter,
            "preds_q": preds_q,
            "targets_q": targets_q,
            "num_atoms": len(targets_q),
            "residual_scale": residual_scale,
            "filename":batch["filename"][0],
            "probe_offset": batch["probe_offset"][0],
            "grid_shape": batch["grid_shape"][0],
            "partial": batch["partial"][0],
            "volume":torch.linalg.det(batch["cell"]).item(),
            "time": time.time() - start_time + batch["load_time"][0],
        }



def transform_log(x, eps=6, forward=False):
    """
    Transform x using log(x + eps) or exp(x) depending on inverse.
    """
    if forward:
        x = torch.exp(x)-eps
    else:
        x = torch.log(x + eps)
        
    return x


