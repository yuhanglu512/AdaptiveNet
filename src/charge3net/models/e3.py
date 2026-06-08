# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import ase
import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct, Linear
from e3nn.util.jit import compile_mode
from torch import nn
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
# from mace.modules.blocks import (
#     AtomicEnergiesBlock,
#     EquivariantProductBasisBlock,
#     InteractionBlock,
#     LinearDipoleReadoutBlock,
#     LinearNodeEmbeddingBlock,
#     LinearReadoutBlock,
#     NonLinearDipoleReadoutBlock,
#     NonLinearReadoutBlock,
#     RadialEmbeddingBlock,
#     ScaleShiftBlock,
# )
# from mace.tools.scatter import scatter_sum
# from mace.modules.utils import (
#     compute_fixed_charge_dipole,
#     get_atomic_virials_stresses,
#     get_edge_vectors_and_lengths,
#     get_outputs,
#     get_symmetric_displacement,
#     prepare_graph,
# )

import src.charge3net.data.layer as layer

magnetic_atom_numbers = torch.tensor(list(range(21,31))+list(range(39,49))+list(range(57,81))+list(range(89,113)),device='cuda')

def get_irreps(total_mul, lmax):
    """
    Get irreps up to lmax, all with roughly the same multiplicity with a total multiplicity of total_mul
    Example:
        get_irreps(500, lmax=2) = 167x0o + 167x0e + 56x1o + 56x1e + 33x2o + 33x2e
    """
    return [
        (round(total_mul / (lmax + 1) / (l * 2 + 1)), (l, p))
        for l in range(lmax + 1)
        for p in [-1, 1]
    ]

def configration_calculator(atom_number:torch.tensor,mag:torch.tensor):
    mask = torch.isin(atom_number, magnetic_atom_numbers)
    return torch.where(mask, torch.sign(mag), 0.0)

class e3_GNN(nn.Module):
    def __init__(
        self,
        num_interactions,
        num_neighbors,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
        num_neighbors_max=8,
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.num_neighbors_max = num_neighbors_max
        self.basis = RadialBasis(
            start=0.0, 
            end=cutoff,
            number=self.number_of_basis,
            basis=basis,
            cutoff=False,
            normalize=True
        )

        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        # store irreps of each output (mostly so the probe model can use)
        self.atom_irreps_sequence = []

        self.num_species = len(ase.data.atomic_numbers)

        # scalar inputs (one-hot atomic numbers) with even parity
        irreps_node_input = f"{self.num_species}x 0e" # scalar inputs (one-hot atomic numbers) with even parity
        self.irreps_node_input = irreps_node_input
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_node_attr = "0e"
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        fc_neurons = [self.number_of_basis, 100]

        # activation to use with even (1) or odd (-1) parities
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_node = irreps_node_input

        for num_i in range(num_interactions):
            # scalar irreps that exist in the tensor product between node and edge irreps
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            # Gate activation function, see https://docs.e3nn.org/en/stable/api/nn/nn_gate.html
            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps_node,
                irreps_node_attr,
                irreps_edge_attr,
                gate.irreps_in,
                fc_neurons,
                num_neighbors,
            )
            irreps_node = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)

            # store output node irreps for each layer
            self.atom_irreps_sequence.append(irreps_node)
        
        self.magnetization_readout = Linear(irreps_node, "0e")
    
    def forward(self, input_dict, delta):
        # Unpad and concatenate edges into batch (0th) dimension
        # incrementing by offset to keep graphs separate
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )

        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        edge_src = edges[:, 0]
        edge_dst = edges[:, 1]

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        #nodes_vector = configration_calculator(nodes_scalar,input_dict["node_vector"])

        # one-hot encode atoms
        nodes_scalar = F.one_hot(nodes_scalar, num_classes=self.num_species)

        nodes= nodes_scalar.float()
        #nodes = torch.cat([nodes_scalar, nodes_vector.reshape(-1,1)], dim=-1)

        # Node attributes are not used here
        node_attr = torch.ones((nodes_scalar.shape[0], 1),dtype=torch.float32, device=nodes_scalar.device)

        # Compute edge distances
        edge_vec = calc_edge_vec(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
        )

        edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.basis(edge_length)
        
        # 应用固定层
        nodes_list = []
        for conv, gate in zip(self.convolutions, self.gates):
            nodes = conv(
                nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
            )
            nodes = gate(nodes)
            nodes_list.append(nodes)

        output = self.magnetization_readout(nodes_list[-1]).reshape(-1)
            
        return output, torch.tensor(len(self.convolutions), device=output.device), torch.tensor(0.0, device=output.device)
    

class AdaptiveNet(nn.Module):
    def __init__(
        self,
        num_interactions,
        num_neighbors,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
        num_neighbors_max=8,
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.num_neighbors_max = num_neighbors_max
        self.basis = RadialBasis(
            start=0.0, 
            end=cutoff,
            number=self.number_of_basis,
            basis=basis,
            cutoff=False,
            normalize=True
        )

        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        # store irreps of each output (mostly so the probe model can use)
        self.atom_irreps_sequence = []

        self.num_species = len(ase.data.atomic_numbers)

        # scalar inputs (one-hot atomic numbers) with even parity
        irreps_node_input = f"{self.num_species}x 0e" # scalar inputs (one-hot atomic numbers) with even parity
        self.irreps_node_input = irreps_node_input
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_node_attr = "0e"
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        fc_neurons = [self.number_of_basis, 100]

        # activation to use with even (1) or odd (-1) parities
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_node = irreps_node_input

        for num_i in range(num_interactions+1):
            # scalar irreps that exist in the tensor product between node and edge irreps
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            # Gate activation function, see https://docs.e3nn.org/en/stable/api/nn/nn_gate.html
            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps_node,
                irreps_node_attr,
                irreps_edge_attr,
                gate.irreps_in,
                fc_neurons,
                num_neighbors,
            )
            irreps_node = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)

            # store output node irreps for each layer
            self.atom_irreps_sequence.append(irreps_node)
        
        self.magnetization_readout = Linear(irreps_node, "0e")
        self.residual_scale = -torch.log(torch.tensor(1/0.1 -1))
        #self.residual_scale = torch.nn.Parameter(torch.tensor(-1.0))
        #self.init_convolution_weights(self.convolutions[-1], init_scale=0.1)

    def _forward_batch(self, input_dict,delta):
        # Unpad and concatenate edges into batch (0th) dimension
        # incrementing by offset to keep graphs separate
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )

        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        edge_src = edges[:, 0]
        edge_dst = edges[:, 1]

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        #nodes_vector = configration_calculator(nodes_scalar,input_dict["node_vector"])

        # one-hot encode atoms
        nodes_scalar = F.one_hot(nodes_scalar, num_classes=self.num_species)

        nodes= nodes_scalar.float()
        #nodes = torch.cat([nodes_scalar, nodes_vector.reshape(-1,1)], dim=-1)

        # Node attributes are not used here
        node_attr = torch.ones((nodes_scalar.shape[0], 1),dtype=torch.float32, device=nodes_scalar.device)

        # Compute edge distances
        edge_vec = calc_edge_vec(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
        )

        edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.basis(edge_length)

        current_nodes = nodes  # 使用单个变量而不是列表
        for conv, gate in zip(self.convolutions[:-1], self.gates[:-1]):
            current_nodes = conv(
                current_nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
            )
            current_nodes = gate(current_nodes)

        total_nodes = current_nodes.shape[0]

        # 预计算材料信息
        batch_size = len(input_dict["num_nodes"])
        material_mask = torch.repeat_interleave(
            torch.arange(1, batch_size + 1, device=current_nodes.device),
            input_dict["num_nodes"]
        )

        # 预计算材料索引矩阵
        material_indices = (material_mask.unsqueeze(1) == torch.arange(1, batch_size + 1, device=current_nodes.device)).float()
        material_counts = material_indices.sum(dim=0)

        # 初始输出
        current_output = self.magnetization_readout(current_nodes).reshape(-1)

        # 计算初始材料输出
        material_outputs = (material_indices.t() @ current_output.unsqueeze(1)).squeeze(1) / material_counts
        prev_material_outputs = material_outputs.clone()

        # 收敛状态跟踪
        converged = torch.zeros(batch_size, dtype=torch.bool, device=current_nodes.device)
        len_iter = (len(self.convolutions)-1) * torch.ones(batch_size, dtype=torch.int32, device=current_nodes.device)

        # 预计算材料节点索引（避免在循环中重复计算）
        material_node_indices = []
        node_start = 0
        for num_nodes in input_dict["num_nodes"]:
            material_node_indices.append(torch.arange(node_start, node_start + num_nodes, device=current_nodes.device))
            node_start += num_nodes

        while torch.max(len_iter) < self.num_neighbors_max and not converged.all():
            active_materials = ~converged
            len_iter[active_materials] += 1

            # 使用预计算的索引创建活跃节点mask
            active_node_mask = torch.zeros(total_nodes, dtype=torch.bool, device=current_nodes.device)
            for i in range(batch_size):
                if active_materials[i]:
                    active_node_mask[material_node_indices[i]] = True

            # 如果没有活跃节点，提前退出
            if not active_node_mask.any():
                break
            
            # 筛选活跃节点
            active_nodes = current_nodes[active_node_mask]
            active_node_attr = node_attr[active_node_mask]

            # 创建节点索引映射
            node_idx_mapping = torch.zeros(total_nodes, dtype=torch.long, device=current_nodes.device)
            active_indices = torch.where(active_node_mask)[0]
            node_idx_mapping[active_node_mask] = torch.arange(active_nodes.shape[0], device=current_nodes.device)

            # 筛选活跃边
            active_edge_mask = active_node_mask[edge_src] & active_node_mask[edge_dst]

            # 修复边索引映射
            active_edge_src = node_idx_mapping[edge_src[active_edge_mask]]
            active_edge_dst = node_idx_mapping[edge_dst[active_edge_mask]]
            active_edge_attr = edge_attr[active_edge_mask]
            active_edge_scalars = edge_length_embedding[active_edge_mask]

            # 对活跃节点执行卷积
            nodes_tmp = self.convolutions[-1](
                active_nodes, active_node_attr, active_edge_src, active_edge_dst, 
                active_edge_attr, active_edge_scalars
            )
            new_active_nodes = active_nodes + 1/(1+torch.exp(-self.residual_scale))*self.gates[-1](nodes_tmp)

            # 更新全局节点特征（只更新活跃节点）
            current_nodes = current_nodes.clone()  # 避免原地操作
            current_nodes[active_node_mask] = new_active_nodes

            # 计算新输出
            new_output = self.magnetization_readout(current_nodes).reshape(-1)

            # 计算新材料输出
            new_material_outputs = (material_indices.t() @ new_output.unsqueeze(1)).squeeze(1) / material_counts

            # 计算每个材料的输出变化
            diff = torch.abs(new_material_outputs - prev_material_outputs)

            # 确定新收敛的材料
            newly_converged = ~converged & (diff < delta)

            # 更新前一轮输出（只更新未收敛的材料）
            update_mask = ~converged & ~newly_converged
            prev_material_outputs[update_mask] = new_material_outputs[update_mask]

            # 更新收敛状态
            converged = converged | newly_converged

            if converged.all():
                break
            
        # 最终输出
        final_output = self.magnetization_readout(current_nodes).reshape(-1)
        return final_output, len_iter, 1/(1+torch.exp(-self.residual_scale))
    
    def forward_single(self, input_dict, delta):
        # Unpad and concatenate edges into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )

        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        edge_src = edges[:, 0]
        edge_dst = edges[:, 1]

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])

        # one-hot encode atoms
        nodes_scalar_onehot = F.one_hot(nodes_scalar, num_classes=self.num_species).float()
        nodes = nodes_scalar_onehot

        # Node attributes are not used here
        node_attr = torch.ones((nodes_scalar.shape[0], 1), dtype=torch.float32, device=nodes_scalar.device)

        # Compute edge distances
        edge_vec = calc_edge_vec(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
        )

        edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.basis(edge_length)

        # 固定层（除最后一层外）
        for conv, gate in zip(self.convolutions[:-1], self.gates[:-1]):
            nodes = conv(nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)
            nodes = gate(nodes)
        
        last_conv = self.convolutions[-1]
        last_gate = self.gates[-1]
        residual_coef = 1/(1+torch.exp(-self.residual_scale))

        # 自适应迭代 - 仅保留上一个输出用于收敛判断
        current_output = self.magnetization_readout(nodes).reshape(-1)
        len_iter = len(self.convolutions) - 1

        while len_iter < self.num_neighbors_max:
            len_iter += 1

            # 执行最后一个卷积层和门
            nodes_tmp = last_conv(nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)
            nodes = nodes + residual_coef * last_gate(nodes_tmp)

            # 计算新输出
            new_output = self.magnetization_readout(nodes).reshape(-1)

            # 检查收敛（使用平均绝对差）
            if torch.abs(new_output - current_output).mean() < delta:
                current_output = new_output
                break

            current_output = new_output            
        return current_output, None, 1/(1+torch.exp(-self.residual_scale))
    
    def forward(self, input_dict, delta):
        batch_size = len(input_dict["num_nodes"])
        if batch_size == 1:
            return self.forward_single(input_dict, delta)
        else:
            return self._forward_batch(input_dict, delta)
        

    def init_convolution_weights(self, conv_layer, init_scale=0.1):
        """
        专门为 e3nn Convolution 层设计的初始化函数
        """
        # 初始化 FullyConnectedTensorProduct 层
        if hasattr(conv_layer, 'sc'):
            self._init_fully_connected_tp(conv_layer.sc, init_scale)
        if hasattr(conv_layer, 'lin1'):
            self._init_fully_connected_tp(conv_layer.lin1, init_scale)
        if hasattr(conv_layer, 'lin2'):
            self._init_fully_connected_tp(conv_layer.lin2, init_scale)
        if hasattr(conv_layer, 'lin3'):
            self._init_fully_connected_tp(conv_layer.lin3, init_scale)

        # 初始化 TensorProduct
        if hasattr(conv_layer, 'tp'):
            self._init_tensor_product(conv_layer.tp, init_scale)

        # 初始化 FullyConnectedNet
        if hasattr(conv_layer, 'fc'):
            self._init_fully_connected_net(conv_layer.fc, init_scale)

    def _init_fully_connected_tp(self, tp_layer, init_scale):
        """初始化 FullyConnectedTensorProduct"""
        if hasattr(tp_layer, 'weight'):
            if tp_layer.weight is not None:
                # 使用较小的初始化
                torch.nn.init.normal_(tp_layer.weight, mean=0, std=init_scale * 0.1)

        # 如果有偏置项
        if hasattr(tp_layer, 'bias') and tp_layer.bias is not None:
            torch.nn.init.constant_(tp_layer.bias, 0)

    def _init_tensor_product(self, tp_layer, init_scale):
        """初始化 TensorProduct"""
        if hasattr(tp_layer, 'weight'):
            if tp_layer.weight is not None:
                # TensorProduct 通常需要更小的初始化
                torch.nn.init.normal_(tp_layer.weight, mean=0, std=init_scale * 0.05)

    def _init_fully_connected_net(self, fc_net, init_scale):
        """初始化 FullyConnectedNet"""
        for layer in fc_net:
            if hasattr(layer, 'weight'):
                # 标准线性层使用 Xavier 初始化，但用较小的 gain
                torch.nn.init.xavier_uniform_(layer.weight, gain=init_scale)
            if hasattr(layer, 'bias') and layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

#    def _init_gate_layer(self, gate_layer):
#        """初始化门控层"""
#        # 假设门控层是一个简单的线性层或MLP
#        for module in gate_layer.modules():
#            if isinstance(module, torch.nn.Linear):
#                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
#                if module.bias is not None:
#                    torch.nn.init.constant_(module.bias, 0)


class q_E3DensityModel(nn.Module):
    def __init__(
        self,
        num_interactions=3,
        num_neighbors=20,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
        spin=False,
        mag=False,
        q=False,
    ):
        super().__init__()
        self.spin = spin if not mag else False
        self.mag = mag
        self.q = q

        if not self.mag:
            self.atom_model = E3AtomRepresentationModel(
                num_interactions,
                num_neighbors,
                mul=mul,
                lmax=lmax,
                cutoff=cutoff,
                basis=basis,
                num_basis=num_basis,
                spin=spin,
                q=q
            )

            self.probe_model = E3ProbeMessageModel(
                num_interactions,
                num_neighbors,
                self.atom_model.atom_irreps_sequence,
                mul=mul,
                lmax=lmax,
                cutoff=cutoff,
                basis=basis,
                num_basis=num_basis,
                spin=spin
            )
        else:
            self.mag_model = E3AtomMagnetizationModel(
                num_interactions,
                num_neighbors,
                mul=mul,
                lmax=lmax,
                cutoff=cutoff,
                basis=basis,
                num_basis=num_basis,
                spin=spin
            )


    def forward(self, input_dict):
        if self.mag:
            magnetization = self.mag_model(input_dict)
            probe_result = magnetization
        else:
            atom_representation = self.atom_model(input_dict)
            # if spin == False, (n_batch, n_probe). if spin == True, (n_batch, n_probe, 2)
            # allow it to output spin density of up/down electrons separately
            # TODO: is it better to train on spin up/down density, or charge density + spin density (like in CHGCAR)?
            probe_result = self.probe_model(input_dict, atom_representation)   
            if self.spin:
                spin_up, spin_down = probe_result[:, :, 0], probe_result[:, :, 1]
                #probe_result[:, :, 0] = spin_up + spin_down
                #probe_result[:, :, 1] = spin_up - spin_down
        return probe_result


class E3DensityModel(nn.Module):
    def __init__(
        self,
        num_interactions=3,
        num_neighbors=20,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
        spin=False,
        mag=False
    ):
        super().__init__()
        self.spin = spin if not mag else False
        self.mag = mag

        if not self.mag:
            self.atom_model = E3AtomRepresentationModel(
                num_interactions,
                num_neighbors,
                mul=mul,
                lmax=lmax,
                cutoff=cutoff,
                basis=basis,
                num_basis=num_basis,
                spin=spin
            )

            self.probe_model = E3ProbeMessageModel(
                num_interactions,
                num_neighbors,
                self.atom_model.atom_irreps_sequence,
                mul=mul,
                lmax=lmax,
                cutoff=cutoff,
                basis=basis,
                num_basis=num_basis,
                spin=spin
            )
        else:
            self.mag_model = E3AtomMagnetizationModel(
                num_interactions,
                num_neighbors,
                mul=mul,
                lmax=lmax,
                cutoff=cutoff,
                basis=basis,
                num_basis=num_basis,
                spin=spin
            )


    def forward(self, input_dict):
        if self.mag:
            magnetization = self.mag_model(input_dict)
            probe_result = magnetization
        else:
            atom_representation = self.atom_model(input_dict)
            # if spin == False, (n_batch, n_probe). if spin == True, (n_batch, n_probe, 2)
            # allow it to output spin density of up/down electrons separately
            # TODO: is it better to train on spin up/down density, or charge density + spin density (like in CHGCAR)?
            probe_result = self.probe_model(input_dict, atom_representation)   
            if self.spin:
                spin_up, spin_down = probe_result[:, :, 0], probe_result[:, :, 1]
                #probe_result[:, :, 0] = spin_up + spin_down
                #probe_result[:, :, 1] = spin_up - spin_down
        return probe_result


class E3AtomMagnetizationModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        num_neighbors,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
        spin=False
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.spin = spin
        self.basis = RadialBasis(
            start=0.0, 
            end=cutoff,
            number=self.number_of_basis,
            basis=basis,
            cutoff=False,
            normalize=True
        )

        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        # store irreps of each output (mostly so the probe model can use)
        self.atom_irreps_sequence = []

        self.num_species = len(ase.data.atomic_numbers)

        # scalar inputs (one-hot atomic numbers) with even parity
        irreps_node_input = f"{self.num_species+1}x 0e" # scalar inputs (one-hot atomic numbers) with even parity
        self.irreps_node_input = irreps_node_input
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_node_attr = "0e"
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        fc_neurons = [self.number_of_basis, 100]

        # activation to use with even (1) or odd (-1) parities
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_node = irreps_node_input

        for _ in range(num_interactions):
            # scalar irreps that exist in the tensor product between node and edge irreps
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            # Gate activation function, see https://docs.e3nn.org/en/stable/api/nn/nn_gate.html
            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps_node,
                irreps_node_attr,
                irreps_edge_attr,
                gate.irreps_in,
                fc_neurons,
                num_neighbors,
            )
            irreps_node = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)

            # store output node irreps for each layer
            self.atom_irreps_sequence.append(irreps_node)
        
        self.magnetization_readout = Linear(irreps_node, "0e")

    def forward(self, input_dict):
        # Unpad and concatenate edges into batch (0th) dimension
        # incrementing by offset to keep graphs separate
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )

        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        edge_src = edges[:, 0]
        edge_dst = edges[:, 1]

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes_vector = configration_calculator(nodes_scalar,input_dict["node_vector"])

        # one-hot encode atoms
        nodes_scalar = F.one_hot(nodes_scalar, num_classes=self.num_species)

        #nodes= nodes_scalar.float()
        nodes = torch.cat([nodes_scalar, nodes_vector.reshape(-1,1)], dim=-1)

        # Node attributes are not used here
        node_attr = torch.ones((nodes_scalar.shape[0], 1),dtype=torch.float32, device=nodes_scalar.device)

        # Compute edge distances
        edge_vec = calc_edge_vec(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
        )

        edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.basis(edge_length)

        nodes_list = []
        # Apply interaction layers
        for conv, gate in zip(self.convolutions, self.gates):
            nodes = conv(
                nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
            )
            nodes = gate(nodes)
            nodes_list.append(nodes)

        return self.magnetization_readout(nodes).squeeze(-1)


class MagneticMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float = 4.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 3,
        num_interactions: int = 3,
        num_elements: int = 100,
        hidden_irreps: str = "128x0e + 64x1o + 32x2e",
        MLP_irreps: str = "64x0e",
        avg_num_neighbors: float = 20.0,
        correlation: int = 3,
        radial_MLP: Optional[List[int]] = None,
        spin: bool = False
    ):
        super().__init__()
        
        self.spin = spin
        self.num_elements = num_elements
        
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
            
        # 转换为o3.Irreps对象
        hidden_irreps = o3.Irreps(hidden_irreps)
        MLP_irreps = o3.Irreps(MLP_irreps)
        
        # 原子序数列表
        atomic_numbers = list(range(1, num_elements + 1))
        
        # 使用MACE的组件
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        
        # 节点嵌入
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, 
            irreps_out=node_feats_irreps
        )
        
        # 球谐函数
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        
        # 构建交互层和乘积层
        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        
        for i in range(num_interactions):
            # 交互层
            inter = InteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps if i == 0 else hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=o3.Irreps(f"{self.radial_embedding.out_dim}x0e"),
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            
            # 乘积层
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=(i > 0),  # 第一层不使用self-connection
            )
            self.products.append(prod)
        
        # 磁矩读取头 - 输出标量磁矩
        self.magnetization_readout = LinearReadoutBlock(
            hidden_irreps, 
            o3.Irreps("1x0e")  # 输出标量磁矩
        )
    
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 准备图数据
        data = self.prepare_graph_data(data)
        
        # 节点嵌入
        node_feats = self.node_embedding(data["node_attrs"])
        
        # 边特征
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data.get("shifts", None),
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], None
        )
        
        # 交互层
        for interaction, product in zip(self.interactions, self.products):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
        
        # 预测磁矩
        atomic_magnetization = self.magnetization_readout(node_feats).squeeze(-1)
        
        # 如果需要总磁矩
        total_magnetization = scatter_sum(
            src=atomic_magnetization,
            index=data["batch"],
            dim=0,
            dim_size=data["ptr"].numel() - 1,
        )
        
        return {
            "atomic_magnetization": atomic_magnetization,
            "total_magnetization": total_magnetization,
        }


class E3AtomRepresentationModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        num_neighbors,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
        spin=False,
        q=False
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.spin = spin
        self.q = q
        self.basis = RadialBasis(
            start=0.0, 
            end=cutoff,
            number=self.number_of_basis,
            basis=basis,
            cutoff=False,
            normalize=True
        )

        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        # store irreps of each output (mostly so the probe model can use)
        self.atom_irreps_sequence = []

        self.num_species = len(ase.data.atomic_numbers)

        # scalar inputs (one-hot atomic numbers) with even parity
        irreps_node_input = f"{self.num_species}x 0e" if not q else f"{self.num_species+1}x 0e" # scalar inputs (one-hot atomic numbers) with even parity
        self.irreps_node_input = irreps_node_input
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_node_attr = "0e"
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        fc_neurons = [self.number_of_basis, 100]

        # activation to use with even (1) or odd (-1) parities
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_node = irreps_node_input

        for _ in range(num_interactions):
            # scalar irreps that exist in the tensor product between node and edge irreps
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            # Gate activation function, see https://docs.e3nn.org/en/stable/api/nn/nn_gate.html
            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps_node,
                irreps_node_attr,
                irreps_edge_attr,
                gate.irreps_in,
                fc_neurons,
                num_neighbors,
            )
            irreps_node = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)

            # store output node irreps for each layer
            self.atom_irreps_sequence.append(irreps_node)  

    def forward(self, input_dict):
        # Unpad and concatenate edges into batch (0th) dimension
        # incrementing by offset to keep graphs separate
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )

        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        edge_src = edges[:, 0]
        edge_dst = edges[:, 1]

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes_scalar = F.one_hot(nodes_scalar, num_classes=self.num_species).float()
        if self.spin:
            nodes_vector = layer.unpad_and_cat(input_dict["node_vector"], input_dict["num_nodes"])
            #nodes_vector = configration_calculator(nodes_scalar,nodes_vector)
            
            nodes = torch.cat([nodes_scalar, nodes_vector.reshape(-1,1)], dim=-1)
        elif self.q:
            nodes = torch.cat([nodes_scalar, input_dict['q'].reshape(-1,1)], dim=-1)
        else:
            # one-hot encode atoms
            nodes= nodes_scalar

        # Node attributes are not used here
        node_attr = torch.ones((nodes_scalar.shape[0], 1),dtype=torch.float32, device=nodes_scalar.device)

        # Compute edge distances
        edge_vec = calc_edge_vec(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
        )

        edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.basis(edge_length)

        nodes_list = []
        # Apply interaction layers
        for conv, gate in zip(self.convolutions, self.gates):
            nodes = conv(
                nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
            )
            nodes = gate(nodes)
            nodes_list.append(nodes)

        return nodes_list


class E3ProbeMessageModel(torch.nn.Module):
    def __init__(
        self,
        num_interactions,
        num_neighbors,
        atom_irreps_sequence,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
        spin=False
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.basis = RadialBasis(
            start=0.0, 
            end=cutoff,
            number=self.number_of_basis,
            basis=basis,
            cutoff=False,
            normalize=True
        )

        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        # scalar inputs with even parity (for probes its just 0s)
        irreps_node_input = "0e"
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_node_attr = "0e"
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        fc_neurons = [self.number_of_basis, 100]

        # activation to use with even (1) or odd (-1) parities
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_node = irreps_node_input

        for i in range(num_interactions):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            # Gate activation function, see https://docs.e3nn.org/en/stable/api/nn/nn_gate.html
            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )

            conv = ConvolutionOneWay(
                irreps_sender_input=atom_irreps_sequence[i],
                irreps_sender_attr=irreps_node_attr,
                irreps_receiver_input=irreps_node,
                irreps_receiver_attr=irreps_node_attr,
                irreps_edge_attr=irreps_edge_attr,
                irreps_node_output=gate.irreps_in,
                fc_neurons=fc_neurons,
                num_neighbors=num_neighbors,
            )
            irreps_node = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)

        # last layer, scalar output
        if spin:
            out = "2x0e"
        else:
            out = "0e"
        self.readout = Linear(irreps_node, out)

    def forward(self, input_dict, atom_representation):
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = layer.unpad_and_cat(
            input_dict["probe_xyz"], input_dict["num_probes"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        probe_edges_displacement = layer.unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
        edge_probe_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_probes"].device),
                    input_dict["num_probes"][:-1],
                )
            ),
            dim=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = torch.cat((edge_offset, edge_probe_offset), dim=2)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        probe_edge_vec = calc_edge_vec_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
        )
        probe_edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), probe_edge_vec, True, normalization="component"
        )
        probe_edge_length = probe_edge_vec.norm(dim=1)
        probe_edge_length_embedding = self.basis(probe_edge_length)

        probe_edge_src = probe_edges[:, 0]
        probe_edge_dst = probe_edges[:, 1]

        # initialize probes
        probes = torch.zeros(
            (torch.sum(input_dict["num_probes"]), 1),
            device=atom_representation[0].device,
        )

        # Probe attributes are not used here
        probe_attr = probes.new_ones(probes.shape[0], 1)

        # Node attributes are not used here
        atom_node_attr = probes.new_ones(atom_xyz.shape[0], 1)

        # Apply interaction layers
        for conv, gate, atom_nodes in zip(
            self.convolutions, self.gates, atom_representation
        ):
            probes = conv(
                atom_nodes,
                atom_node_attr,
                probes,
                probe_attr,
                probe_edge_src,
                probe_edge_dst,
                probe_edge_attr,
                probe_edge_length_embedding,
            )
            probes = gate(probes)

        probes = self.readout(probes).squeeze()

        # rebatch
        probes = layer.pad_and_stack(
            torch.split(
                probes,
                list(input_dict["num_probes"].detach().cpu().numpy()),
                dim=0,
            )
        )
        return probes
    
    def forward_with_precomputed_edges(
        self,
        input_dict,
        atom_representation,
        precomputed_edge_src,
        precomputed_edge_dst,
        precomputed_edge_attr,
        precomputed_edge_length_embedding,
        probe_start_idx,
        probe_end_idx,
    ):
        """
        使用预计算的边特征进行推理，避免重复计算球谐函数和径向基。
        
        Args:
            input_dict: 包含 probe_target, num_probes 等信息的字典（子批次）
            atom_representation: atom_model 输出的节点特征列表
            precomputed_edge_src: 全量边源节点索引 (num_edges_total,)
            precomputed_edge_dst: 全量边目标节点索引 (num_edges_total,)  (全局 probe 索引)
            precomputed_edge_attr: 全量边球谐特征 (num_edges_total, ...)
            precomputed_edge_length_embedding: 全量边径向基特征 (num_edges_total, num_basis)
            probe_start_idx: 当前子批次在全局 probe 中的起始索引
            probe_end_idx: 结束索引（不包含）
        """
        # 筛选属于当前子批次的边
        mask = (precomputed_edge_dst >= probe_start_idx) & (precomputed_edge_dst < probe_end_idx)
        edge_src = precomputed_edge_src[mask]
        edge_dst = precomputed_edge_dst[mask] - probe_start_idx   # 映射到子批次局部索引
        edge_attr = precomputed_edge_attr[mask]
        edge_length_embedding = precomputed_edge_length_embedding[mask]

        num_probes_local = probe_end_idx - probe_start_idx
        probes = torch.zeros((num_probes_local, 1), device=edge_src.device)
        probe_attr = torch.ones_like(probes)
        atom_node_attr = torch.ones(atom_representation[0].shape[0], 1, device=probes.device)

        # 执行卷积层
        for conv, gate, atom_nodes in zip(self.convolutions, self.gates, atom_representation):
            probes = conv(
                atom_nodes, atom_node_attr,
                probes, probe_attr,
                edge_src, edge_dst,
                edge_attr, edge_length_embedding
            )
            probes = gate(probes)

        # 读出
        probes = self.readout(probes).squeeze()
        # 重新 batch 化（假设 input_dict 中包含 num_probes 信息，形状为 (batch, num_probes)）
        # 这里 probes 形状为 (num_probes_local,)，需要 reshape 为 (1, num_probes_local)
        probes = probes.view(1, -1)
        return probes


class RadialBasis(nn.Module):
    r"""
    Wrapper for e3nn.math.soft_one_hot_linspace, with option for normalization
    Args:
        start (float): mininum value of basis
        end (float): maximum value of basis
        number (int): number of basis functions
        basis ({'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}): basis family
        cutoff (bool): all x outside interval \approx 0
        normalize (bool): normalize function to have a mean of 0, std of 1
        samples (int): number of samples to use to find mean/std
    """
    def __init__(
        self,
        start,
        end,
        number,
        basis="gaussian",
        cutoff=False,
        normalize=True,
        samples=4000
    ):
        super().__init__()
        self.start = start
        self.end = end
        self.number = number
        self.basis = basis
        self.cutoff = cutoff
        self.normalize = normalize

        if normalize:
            with torch.no_grad():
                rs = torch.linspace(start, end, samples+1)[1:]
                bs = soft_one_hot_linspace(rs, start, end, number, basis, cutoff)
                assert bs.ndim == 2 and len(bs) == samples
                std, mean = torch.std_mean(bs, dim=0)
            self.register_buffer("mean", mean)
            self.register_buffer("inv_std", torch.reciprocal(std))
        
    def forward(self, x):
        x = soft_one_hot_linspace(x, self.start, self.end, self.number, self.basis, self.cutoff)
        if self.normalize:
            x = (x - self.mean) * self.inv_std
        return x


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)


def calc_edge_vec(
    positions: torch.Tensor,
    cells: torch.Tensor,
    edges: torch.Tensor,
    edges_displacement: torch.Tensor,
    splits: torch.Tensor,
):
    """
    Calculate vectors of edges
    (modified from src.data.layer.calc_distance)

    Args:
        positions: Tensor of shape (num_nodes, 3) with xyz coordinates inside cell
        cells: Tensor of shape (num_splits, 3, 3) with one unit cell for each split
        edges: Tensor of shape (num_edges, 2)
        edges_displacement: Tensor of shape (num_edges, 3) with the offset (in number of cell vectors) of the sending node
        splits: 1-dimensional tensor with the number of edges for each separate graph
    """
    unitcell_repeat = torch.repeat_interleave(cells, splits, dim=0)  # num_edges, 3, 3
    displacement = torch.matmul(
        torch.unsqueeze(edges_displacement, 1), unitcell_repeat
    )  # num_edges, 1, 3
    displacement = torch.squeeze(displacement, dim=1)
    neigh_pos = positions[edges[:, 0]]  # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions[edges[:, 1]]  # num_edges, 3
    vec = this_pos - neigh_abs_pos  # num_edges, 3
    return vec


def calc_edge_vec_to_probe(
    positions: torch.Tensor,
    positions_probe: torch.Tensor,
    cells: torch.Tensor,
    edges: torch.Tensor,
    edges_displacement: torch.Tensor,
    splits: torch.Tensor,
    return_diff=False,
):
    """
    Calculate vectors of edges from atoms to probes
    (modified from src.data.layer.calc_distance)

    Args:
        positions: Tensor of shape (num_nodes, 3) with xyz coordinates inside cell
        positions_probe: Tensor of shape (num_probes, 3) with xyz coordinates of probes inside cell
        cells: Tensor of shape (num_splits, 3, 3) with one unit cell for each split
        edges: Tensor of shape (num_edges, 2)
        edges_displacement: Tensor of shape (num_edges, 3) with the offset (in number of cell vectors) of the sending node
        splits: 1-dimensional tensor with the number of edges for each separate graph
    """
    unitcell_repeat = torch.repeat_interleave(cells, splits, dim=0)  # num_edges, 3, 3
    displacement = torch.matmul(
        torch.unsqueeze(edges_displacement, 1), unitcell_repeat
    )  # num_edges, 1, 3
    displacement = torch.squeeze(displacement, dim=1)
    neigh_pos = positions[edges[:, 0]]  # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions_probe[edges[:, 1]]  # num_edges, 3
    vec = this_pos - neigh_abs_pos  # num_edges, 3
    return vec


# Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
# University of California, through Lawrence Berkeley National Laboratory
# (subject to receipt of any required approvals from the U.S. Dept. of Energy), 
# Ecole Polytechnique Federale de Lausanne (EPFL), Free University of Berlin 
# and Kostiantyn Lapchevskyi. All rights reserved.
# Modified from https://github.com/e3nn/e3nn/blob/05b386177ed039156526f9c67d0d87b6c21ff5d3/e3nn/nn/models/v2103/points_convolution.py
#  - Remove torch_scatter dependency
#  - Add support for differently indexed sending/receiver nodes.
#  - Sender and receiver nodes can have different irreps.
@compile_mode("script")
class Convolution(torch.nn.Module):
    """
    Equivariant Convolution
    Args:
        irreps_node_input (e3nn.o3.Irreps): representation of the input node features
        irreps_node_attr (e3nn.o3.Irreps): representation of the node attributes
        irreps_edge_attr (e3nn.o3.Irreps): representation of the edge attributes
        irreps_node_output (e3nn.o3.Irreps or None): representation of the output node features
        fc_neurons (list[int]): number of neurons per layers in the fully connected network
            first layer and hidden layers but not the output layer
        num_neighbors (float): typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output
        )

        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            fc_neurons + [tp.weight_numel], torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_node_attr, self.irreps_node_output
        )
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

    def forward(
        self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars
    ) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        node_self_connection = self.sc(node_input, node_attr)
        node_features = self.lin1(node_input, node_attr)

        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(
            edge_features, edge_dst, dim_size=node_input.shape[0]
        ).div(self.num_neighbors**0.5)

        node_conv_out = self.lin2(node_features, node_attr)
        node_angle = 0.1 * self.lin3(node_features, node_attr)
        #            ^^^------ start small, favor self-connection

        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * node_self_connection + sin * node_conv_out


@compile_mode("script")
class OptimizedConvolution(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        num_neighbors,
        # 新增优化参数
        use_simplified_irreps=True,
        reduce_mid_irreps=True,
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors
        
        # 优化1: 简化不可约表示
        if use_simplified_irreps:
            self.irreps_node_input = self._simplify_irreps(self.irreps_node_input)
            self.irreps_node_output = self._simplify_irreps(self.irreps_node_output)

        self.sc = FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output
        )

        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input
        )

        # 优化2: 减少中间表示的复杂性
        irreps_mid = []
        instructions = []
        
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    # 优化3: 更严格的过滤条件
                    if ir_out in self.irreps_node_output:
                        # 优化4: 限制多重度
                        if reduce_mid_irreps and mul > 1:
                            mul = max(1, mul // 2)  # 减少多重度
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, len(irreps_mid)-1, "uvu", True))
        
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        # 优化5: 使用更简单的全连接网络
        if len(fc_neurons) > 2:
            fc_neurons = fc_neurons[:2]  # 减少层数
            
        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        
        self.fc = FullyConnectedNet(
            fc_neurons + [tp.weight_numel], torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_node_attr, self.irreps_node_output
        )
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")
        
        # 缓存一些计算
        self._cached_output_mask = None

    def _simplify_irreps(self, irreps, max_l=2):
        """简化不可约表示，限制最大角动量"""
        simplified = []
        for mul, ir in irreps:
            if ir.l <= max_l:  # 限制角动量
                simplified.append((mul, ir))
        return o3.Irreps(simplified)
    
    @property
    def output_mask(self):
        if self._cached_output_mask is None:
            self._cached_output_mask = self.sc.output_mask
        return self._cached_output_mask

    def forward(
        self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars
    ) -> torch.Tensor:
        # 优化6: 预先计算常用值
        num_nodes = node_input.shape[0]
        normalization_factor = self.num_neighbors**0.5
        
        weight = self.fc(edge_scalars)

        # 优化7: 合并一些计算
        node_self_connection = self.sc(node_input, node_attr)
        node_features = self.lin1(node_input, node_attr)

        # 优化8: 使用更高效的索引操作
        src_features = node_features[edge_src]
        edge_features = self.tp(src_features, edge_attr, weight)
        
        # 优化9: 使用更高效的scatter操作
        node_features = scatter(
            edge_features, edge_dst, dim=0, dim_size=num_nodes
        ).div(normalization_factor)

        node_conv_out = self.lin2(node_features, node_attr)
        node_angle = 0.1 * self.lin3(node_features, node_attr)

        # 优化10: 预先计算三角函数
        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.output_mask  # 使用缓存
        
        # 优化11: 简化门控计算
        sin_gated = (1 - m) + sin * m
        
        return cos * node_self_connection + sin_gated * node_conv_out


@compile_mode("script")
class ConvolutionOneWay(torch.nn.Module):
    """
    Equivariant Convolution, but receiving nodes are differently indexed from sending nodes.
    Additionally, sender and receiver nodes can have different irreps.

    Args:
        irreps_sender_input (e3nn.o3.Irreps): representation of the input sender nodes
        irreps_sender_attr (e3nn.o3.Irreps): representation of the sender attributes
        irreps_receiver_input(e3nn.o3.Irreps): representation of the input receiver nodes
        irreps_receiver_attr (e3nn.o3.Irreps): representation of the receiver attributes
        irreps_edge_attr (e3nn.o3.Irreps): representation of the edge attributes
        irreps_node_output (e3nn.o3.Irreps or None): representation of the output node features
        fc_neurons (list[int]): number of neurons per layers in the fully connected network
            first layer and hidden layers but not the output layer
        num_neighbors (float): typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_sender_input,
        irreps_sender_attr,
        irreps_receiver_input,
        irreps_receiver_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.irreps_sender_input = o3.Irreps(irreps_sender_input)
        self.irreps_sender_attr = o3.Irreps(irreps_sender_attr)
        self.irreps_receiver_input = o3.Irreps(irreps_receiver_input)
        self.irreps_receiver_attr = o3.Irreps(irreps_receiver_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(
            self.irreps_receiver_input,
            self.irreps_receiver_attr,
            self.irreps_node_output,
        )

        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_sender_input, self.irreps_sender_attr, self.irreps_sender_input
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_sender_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_sender_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            fc_neurons + [tp.weight_numel], torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_receiver_attr, self.irreps_node_output
        )
        self.lin3 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_receiver_attr, "0e"
        )

    def forward(
        self,
        sender_input,
        sender_attr,
        receiver_input,
        receiver_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
    ) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        receiver_self_connection = self.sc(receiver_input, receiver_attr)

        sender_features = self.lin1(sender_input, sender_attr)

        edge_features = self.tp(sender_features[edge_src], edge_attr, weight)

        # scatter edge features from sender (atoms) to receiver (probes)
        receiver_features = scatter(
            edge_features, edge_dst, dim_size=receiver_input.shape[0]
        ).div(self.num_neighbors**0.5)

        receiver_conv_out = self.lin2(receiver_features, receiver_attr)
        receiver_angle = 0.1 * self.lin3(receiver_features, receiver_attr)
        #            ^^^------ start small, favor self-connection

        cos, sin = receiver_angle.cos(), receiver_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * receiver_self_connection + sin * receiver_conv_out
