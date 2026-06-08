import re
import sys
import numpy as np
import shutil
from ase import Atoms
from ase.optimize import BFGS, FIRE
from ase.constraints import ExpCellFilter
from ase.calculators.calculator import Calculator, all_changes
import os
import pickle

class VASPOutcarCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    def __init__(self, outcar='OUTCAR', **kwargs):
        super().__init__(**kwargs)
        self.outcar = outcar
        self.energy = None
        self.forces = None
        self.stress = None
        self._read_results()

    def _read_results(self):
        with open(self.outcar, 'r') as f:
            lines = f.readlines()

        energy = None
        for line in reversed(lines):
            # 匹配类似 "free  energy   TOTEN  =      -10.84987656 eV" 的行
            if re.search(r'free\s+energy\s+TOTEN', line):
                parts = line.split()
                # 找到等号后的第一个数字
                for i, p in enumerate(parts):
                    if p == '=':
                        if i+1 < len(parts):
                            energy = float(parts[i+1])
                            break
                if energy is not None:
                    break
        if energy is None:
            # 尝试另一种格式：直接找 "TOTEN  ="
            for line in reversed(lines):
                if 'TOTEN' in line and '=' in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == '=':
                            if i+1 < len(parts):
                                energy = float(parts[i+1])
                                break
                    if energy is not None:
                        break
        if energy is None:
            raise RuntimeError("Cannot find energy in OUTCAR")
        self.energy = energy

        # 读取原子受力 (TOTAL-FORCE)
        forces = []
        last_force_start = None
        for i in range(len(lines)-1, -1, -1):
            if 'TOTAL-FORCE' in lines[i] and 'eV/Angst' in lines[i]:
                last_force_start = i + 2   # 数据从该行的下两行开始
                break

        if last_force_start is None:
            raise RuntimeError("Cannot find TOTAL-FORCE in OUTCAR")

        # 从 last_force_start 开始向后读，直到遇到 "---"
        for j in range(last_force_start, len(lines)):
            if '---' in lines[j]:
                break
            parts = lines[j].split()
            if len(parts) >= 6:
                forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
        self.forces = np.array(forces)

        stress = None
        # 从后往前找最后一个 "FORCE on cell =-STRESS" 行
        last_stress_block_start = None
        for i in range(len(lines)-1, -1, -1):
            if 'FORCE on cell =-STRESS' in lines[i]:
                last_stress_block_start = i
                break

        if last_stress_block_start is not None:
            # 在该块内（从该行向后）寻找 "Total" 行，取最后一个（一般只有一个）
            for j in range(last_stress_block_start, len(lines)):
                if 'Total' in lines[j]:
                    parts = lines[j].split()
                    if len(parts) >= 7:
                        stress = np.array([float(parts[1]), float(parts[2]), float(parts[3]),
                                           float(parts[4]), float(parts[5]), float(parts[6])])

        if stress is None:
            self.stress = np.zeros((6))
        else:
            self.stress = stress

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=all_changes):
        # 获取当前晶胞体积
        volume = atoms.get_volume() if atoms is not None else 1.0
        # 将原始应力转换为单位 eV/Å³
        stress = self.stress / volume
        self.results = {'energy': self.energy,
                        'forces': self.forces,
                        'stress': stress}
        

def read_incar_isif(incar_file='INCAR'):
    """从 INCAR 中读取 ISIF 值，默认返回 2（如果找不到）"""
    isif = 2
    try:
        with open(incar_file, 'r') as f:
            for line in f:
                if re.search(r'^[ \t]*ISIF\s*=', line, re.I):
                    parts = line.split('=')
                    if len(parts) > 1:
                        isif = int(parts[1].strip().split()[0])
                        break
    except FileNotFoundError:
        pass
    return isif

def read_poscar(filename):
    """读取 POSCAR，返回晶胞矩阵、原子种类、数量、坐标类型和原子坐标（笛卡尔）。"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    scale = float(lines[1].strip())
    cell = np.array([list(map(float, lines[i].split())) for i in range(2,5)]) * scale
    species = lines[5].split()
    natoms_per_species = list(map(int, lines[6].split()))
    natoms = sum(natoms_per_species)
    coord_type = lines[7].strip().lower()
    coords = []
    for i in range(8, 8+natoms):
        parts = lines[i].split()
        coords.append(list(map(float, parts[:3])))
    coords = np.array(coords)
    if coord_type.startswith('direct'):
        coords = np.dot(coords, cell)
    return cell, species, natoms_per_species, coord_type, coords, lines

def write_poscar(filename, cell, species, natoms_per_species, coord_type, coords_cart, original_lines):
    """将笛卡尔坐标写回 POSCAR，保持原有格式（Direct 或 Cartesian）。"""
    if coord_type.startswith('direct'):
        inv_cell = np.linalg.inv(cell)
        coords_out = np.dot(coords_cart, inv_cell)
    else:
        coords_out = coords_cart
    with open(filename, 'w') as f:
        for i in range(2):
            f.write(original_lines[i])
        f.write(f"{cell[0][0]:.12f} {cell[0][1]:.12f} {cell[0][2]:.12f}\n")
        f.write(f"{cell[1][0]:.12f} {cell[1][1]:.12f} {cell[1][2]:.12f}\n")
        f.write(f"{cell[2][0]:.12f} {cell[2][1]:.12f} {cell[2][2]:.12f}\n")
        f.write(original_lines[5])
        f.write(original_lines[6])
        f.write(original_lines[7])
        for coord in coords_out:
            f.write(f"{coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}\n")
        natoms = sum(natoms_per_species)
        if len(original_lines) > 8+natoms:
            for extra in original_lines[8+natoms:]:
                f.write(extra)

def parse_forces(outcar):
    """从 OUTCAR 中提取每个原子的总受力（笛卡尔分量，单位 eV/Å）。"""
    forces = []
    with open(outcar, 'r') as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if 'TOTAL-FORCE' in line and 'eV/Angst' in line:
            start = i + 2
            break
    if start is None:
        raise RuntimeError("Cannot find TOTAL-FORCE in OUTCAR")
    for i in range(start, len(lines)):
        if '---' in lines[i]:
            break
        parts = lines[i].split()
        if len(parts) >= 6:
            forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
    return np.array(forces)

def parse_max_force(outcar):
    """从 OUTCAR 中提取最大力（FORCES: max atom 行）。"""
    with open(outcar, 'r') as f:
        for line in f:
            if 'FORCES: max atom' in line:
                parts = line.split()
                # 格式: FORCES: max atom, RMS    0.000000    0.000000
                # 第4个字段是 max force
                return float(parts[3])
    raise RuntimeError("Cannot find FORCES: max atom in OUTCAR")

def parse_stress(outcar):
    """从 OUTCAR 中提取应力张量（Total 行）和外部压力（kB）。"""
    with open(outcar, 'r') as f:
        lines = f.readlines()
    stress_tensor = None
    external_pressure = None
    for i, line in enumerate(lines):
        if 'FORCE on cell =-STRESS' in line:
            for j in range(i, len(lines)):
                if 'in kB' in lines[j]:
                    parts = lines[j].split()
                    if len(parts) >= 7:
                        stress_tensor = np.array([float(parts[2]), float(parts[3]), float(parts[4]),
                                                   float(parts[5]), float(parts[6]), float(parts[7])])
                    break
            break
    if stress_tensor is None:
        stress_tensor = np.zeros((6))
    for line in lines:
        if 'external pressure' in line:
            parts = line.split()
            external_pressure = float(parts[3])
            break
        else:
            external_pressure = 0
    return stress_tensor, external_pressure

def parse_last_positions_and_cell(outcar):
    """
    从 OUTCAR 中提取最后一次出现的原子位置（笛卡尔坐标）和晶胞基矢。
    返回 (cell, positions) ，其中 positions 为 (N,3) numpy数组，cell 为 (3,3)。
    """
    with open(outcar, 'r') as f:
        lines = f.readlines()

    # 提取晶胞（最后一个 "direct lattice vectors" 块）
    cell = None
    for i in range(len(lines)-1, -1, -1):
        if 'direct lattice vectors' in lines[i]:
            # 接下来的三行是三个晶胞矢量
            if i + 3 < len(lines):
                vecs = []
                for j in range(1, 4):
                    parts = lines[i+j].split()
                    # 通常格式: A1 = (   xx   yy   zz )
                    # 取最后三个数值
                    vals = [float(x) for x in parts[-3:]]
                    vecs.append(vals)
                cell = np.array(vecs)
                break

    # 提取原子位置（最后一个 "POSITION" 块）
    positions = []
    pos_start = None
    for i in range(len(lines)-1, -1, -1):
        if 'POSITION' in lines[i] and 'TOTAL-FORCE' in lines[i]:
            pos_start = i + 2   # 通常表头下一行是分隔线，再下一行开始数据
            # 需要确认格式，通常是 "POSITION                                       TOTAL-FORCE" 之后两行开始
            # 稳妥起见向后搜索，直到下一个空行或特定标记
            break

    if pos_start is None:
        raise RuntimeError("Cannot find POSITION in OUTCAR")

    # 从 pos_start 开始读，直到遇到 "---" 或空行，或者行首为空格且无数字
    for i in range(pos_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('---'):
            break
        parts = line.split()
        # 一行至少包含三个坐标 + 三个力，取前三个数字
        if len(parts) >= 3:
            try:
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                break

    positions = np.array(positions)
    if cell is None:
        # 如果没找到晶胞，返回单位矩阵（不合理，但可容错）
        cell = np.eye(3)
        print("Warning: No cell found in OUTCAR, using identity.")
    return cell, positions

def update_atoms(coords, forces, step_size=0.05):
    """仅更新原子位置"""
    delta=step_size*forces
    delta = np.clip(delta,-0.001,0.001)
    return coords + delta

def update_cell(cell, stress_tensor, cell_step=0.001):
    """更新晶胞矩阵"""
    S = np.array([[stress_tensor[0], stress_tensor[3], stress_tensor[5]],
                  [stress_tensor[3], stress_tensor[1], stress_tensor[4]],
                  [stress_tensor[5], stress_tensor[4], stress_tensor[2]]])
    epsilon = cell_step * S
    new_cell = cell @ (np.eye(3) + epsilon)
    return new_cell

def update_chgcar_structure(chgcar_file, new_poscar):
    """
    直接更新 CHGCAR 头部（晶胞和原子坐标），保留网格数据和缀加电荷。
    使用文件流复制，避免将整个大文件读入内存。
    """
    # 读取新结构
    new_cell, new_species, new_natoms_per_species, new_coord_type, new_coords_cart, _ = read_poscar(new_poscar)
    # 转换为 Direct 坐标
    inv_cell = np.linalg.inv(new_cell)
    new_coords_direct = np.dot(new_coords_cart, inv_cell)

    temp_file = chgcar_file + ".tmp"
    with open(chgcar_file, 'r') as f_old, open(temp_file, 'w') as f_new:
        # 1. 复制前两行（注释和缩放因子）
        for _ in range(2):
            f_new.write(f_old.readline())
        # 2. 写入新晶胞三行
        for i in range(3):
            f_new.write(f"{new_cell[i][0]:.12f} {new_cell[i][1]:.12f} {new_cell[i][2]:.12f}\n")
            f_old.readline()  # 跳过原晶胞行
        # 3. 跳过原原子种类行，写入新的
        f_old.readline()
        f_new.write(" ".join(new_species) + "\n")
        # 4. 跳过原原子数量行，写入新的
        f_old.readline()
        f_new.write(" ".join(map(str, new_natoms_per_species)) + "\n")
        # 5. 跳过原坐标类型行，写入新的（保持 Direct）
        f_old.readline()
        f_new.write("Direct\n")
        # 6. 写入新原子坐标
        for coord in new_coords_direct:
            f_new.write(f"{coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}\n")
            f_old.readline()  # 跳过原坐标行
        # 7. 复制剩余所有内容（网格尺寸、网格数据、缀加电荷）
        shutil.copyfileobj(f_old, f_new)

    shutil.move(temp_file, chgcar_file)
    print(f"CHGCAR 头部已更新为新结构: {chgcar_file}")

def compute_adaptive_delta(
    forces_curr, positions_curr,
    forces_prev, positions_prev,
    max_step=0.05,          # 每个方向分量的最大位移绝对值 (Å)
    fallback_coeff=0.1, # 回退时沿力方向移动的系数 (Å per eV/Å)
    min_force_deriv=1e-4,
    min_atom_ratio=0.3     # 至少需要这么多比例的原子有效才外推
):
    """
    利用前一步的力和位置，为 x,y,z 三个方向分别计算共享步长，
    得到每个原子的位移：delta = step_d * force_d ，d∈{x,y,z}。
    若无法外推或出现异常，则回退到保守位移 delta = fallback_coeff * forces。
    返回 (N,3) 位移数组。
    """
    N = forces_curr.shape[0]
    # 初始化为回退位移
    delta = fallback_coeff * forces_curr

    if forces_prev is None or positions_prev is None:
        # 没有历史信息，直接返回保守位移
        delta = np.clip(delta, -max_step, max_step)
        return delta

    # 三个方向的共享步长
    step_coeffs = np.full(3, fallback_coeff)  # 默认用回退系数
    for d in range(3):
        f_curr = forces_curr[:, d]
        f_prev = forces_prev[:, d]
        p_curr = positions_curr[:, d]
        p_prev = positions_prev[:, d]

        dx = p_curr - p_prev
        df = f_curr - f_prev

        # 有效原子：dx 和 df 都不太小
        valid = (np.abs(dx) > 1e-8) & (np.abs(df) > min_force_deriv)
        n_valid = np.sum(valid)
        if n_valid < min_atom_ratio * N:
            continue   # 保留 fallback

        # 计算每个有效原子的斜率，取中位数（更稳健）
        k_vals = df[valid] / dx[valid]
        k = np.median(k_vals)

        if np.abs(k) < min_force_deriv:
            continue

        # 外推系数：希望力变为0，即 f_curr + k * delta = 0 → delta = -f_curr / k
        # 由于 delta = step_coeff * f_curr，所以 step_coeff = -1/k （对所有原子近似成立）
        # 但我们用平均力来反推步长系数：step = -mean(f_curr) / (k * mean(f_curr)) ？ 不，直接算 step = -1/k
        # 实际上我们只需要 step_coeff，所以 step = -1/k
        step_candidate = -1.0 / k

        # 方向正确性检查：
        # 正常的 step 应该 >0，使得 delta 与 force 同向。
        # 如果 step_candidate 为负，那么 delta 会与力方向相反，这是异常的。
        if step_candidate < 0:
            continue   # 保留 fallback

        # 限制步长，使最大位移不超过 max_step
        max_abs_f = np.max(np.abs(f_curr))
        if max_abs_f > 0:
            max_allowed_step = max_step / max_abs_f
            step_candidate = min(step_candidate, max_allowed_step)  # 上限
        else:
            continue

        step_coeffs[d] = step_candidate

    # 根据计算的步长系数生成位移
    # 注意 forces_curr 是 (N,3)，step_coeffs 是 (3,) 会自动广播
    delta_new = forces_curr * step_coeffs  # (N,3)
    print(f"Adaptive step coefficients: {step_coeffs}, fallback_coeff: {fallback_coeff}")

    # 再次硬裁剪单个位移分量
    delta_new = np.clip(delta_new, -max_step, max_step)

    return delta_new

def compute_adaptive_displacements(
    forces_curr, positions_curr,
    forces_prev=None, positions_prev=None,
    max_step=0.05,      # 单步最大位移 (Å)
    fallback_factor=0.1,  # 无法外推时的缩放因子
    min_force_deriv=1e-4  # 力差绝对值下限，避免除以零
):
    """
    根据当前力和可选的上一步力/位置，计算每个原子的位移向量。
    - 如果有上一步信息：线性外推 f=0 的位移 delta = -f_curr / ((f_curr-f_prev)/(x_curr-x_prev))
    - 如果上一步缺失或导数过小/方向异常，则使用保守的位移 = fallback_factor * forces_curr
    最后对每个分量的位移裁剪到 [-max_step, max_step]。
    返回 (N,3) 位移数组。
    """
    N = forces_curr.shape[0]
    if forces_prev is not None and positions_prev is not None:
        dx = positions_curr - positions_prev           # (N,3)
        df = forces_curr - forces_prev                 # (N,3)
        # 避免除以零或极小值
        safe = np.abs(df) > min_force_deriv
        # 初始化位移为 fallback
        delta = fallback_factor * forces_curr          # (N,3)

        # 对每个方向独立外推
        # 我们希望解 f_curr + k * d = 0  => d = -f_curr / k
        # 其中 k = df / dx 。如果 dx 也为零则无法计算。
        # 仅当 df 和 dx 都非零时进行外推
        valid = safe & (np.abs(dx) > 1e-8)
        # 计算斜率 k = df / dx
        with np.errstate(divide='ignore', invalid='ignore'):
            k = np.divide(df, dx, where=valid)
            # 外推位移
            d_extrap = np.divide(-forces_curr, k, where=valid)
        # 只对 valid 采用外推值
        delta = np.where(valid, d_extrap, delta)

        # 额外安全检查：外推方向应与力的方向一致（即 delta 应与 force 同号）
        # 如果符号相反，说明斜率符号异常（力反方向变大），回退到 fallback
        sign_mismatch = np.sign(delta) != np.sign(forces_curr)
        delta = np.where(sign_mismatch, fallback_factor * forces_curr, delta)
    else:
        # 第一次迭代，无历史信息，直接用 fallback
        delta = fallback_factor * forces_curr

    # 最终裁剪位移
    delta = np.clip(delta, -max_step, max_step)
    return delta


#def main():
#    # ---------- 参数解析 ----------
#    cell_step = 0.00001
#    force_threshold = 0.01
#    pressure_threshold = 0.01
#    iteration = 1
#    intermediate_dir = "../intermediate_dir"   # 默认值
#
#    if len(sys.argv) > 1:
#        try:
#            force_threshold = float(sys.argv[1])
#        except ValueError:
#            pass
#    if len(sys.argv) > 2:
#        try:
#            iteration = int(sys.argv[2])
#        except ValueError:
#            pass
#    if len(sys.argv) > 3:
#        intermediate_dir = sys.argv[3]
#
#    # 读取 ISIF 和当前结构
#    isif = read_incar_isif('INCAR')
#    print(f"ISIF = {isif}")
#
#    # 读取 POSCAR，创建 ASE Atoms 对象
#    cell, species, natoms_per_species, coord_type, coords_cart, poscar_lines = read_poscar('POSCAR')
#    inv_cell = np.linalg.inv(cell)
#    coords_frac = np.dot(coords_cart, inv_cell)
#    symbols = []
#    for s, n in zip(species, natoms_per_species):
#        symbols.extend([s] * n)
#    atoms = Atoms(symbols=symbols, cell=cell, pbc=True)
#    atoms.set_positions(coords_cart)
#
#    # 创建自定义计算器，从 OUTCAR 中读取能量、力、应力
#    calc = VASPOutcarCalculator('OUTCAR')
#    atoms.set_calculator(calc)
#
#    # 获取最大力和压力
#    forces = atoms.get_forces()
#    max_force = np.max(np.linalg.norm(forces, axis=1))
#    stress_tensor, pressure = parse_stress('OUTCAR')
#    print(f"Max force = {max_force:.6f} eV/Å")
#    print(f"External pressure = {pressure:.2f} kB")
#    print(f"stress_tensor = {np.max(stress_tensor):.2f} kB")
#
#    # 收敛判断
#    if isif==2 or isif==0:
#        if max_force < force_threshold:
#            print(f"Converged: max force {max_force:.6f} < {force_threshold}")
#            sys.exit(0)
#    elif isif==3:
#        if max_force < force_threshold and abs(pressure) < pressure_threshold:
#            print(f"Converged: max force {max_force:.6f} < {force_threshold} and pressure {pressure:.2f} kB < {pressure_threshold:.2f} kB")
#            sys.exit(0)
#
#    forces_prev = None
#    positions_prev = None
#    if iteration > 1:
#        prev_outcar = os.path.join(intermediate_dir, f"OUTCAR_{iteration-1}")
#        if os.path.isfile(prev_outcar):
#            print(f"Loading previous OUTCAR: {prev_outcar}")
#            try:
#                forces_prev = parse_forces(prev_outcar)
#                _, positions_prev = parse_last_positions_and_cell(prev_outcar)
#            except Exception as e:
#                print(f"Warning: Failed to read previous OUTCAR: {e}")
#
#    # 计算自适应位移
#    delta = compute_adaptive_delta(
#        forces, coords_cart,
#        forces_prev=forces_prev,
#        positions_prev=positions_prev,
#        max_step=0.01,           # 可根据需要调整
#        fallback_coeff=0.05
#    )
#
#    # 更新原子坐标
#    new_coords_cart = coords_cart + delta
#
#    # 晶胞更新（保留原有简单方法，不做自适应）
#    if isif == 3:
#        new_cell = update_cell(cell, stress_tensor, cell_step=cell_step)
#        # 坐标转换
#        coords_frac_new = np.dot(coords_cart, inv_cell)
#        temp_coords_cart = np.dot(coords_frac_new, new_cell)
#        new_coords_cart = temp_coords_cart + delta
#    else:
#        new_cell = cell    
#
##    if isif==3:
##        # 同时优化原子和晶胞
##        #ecf = ExpCellFilter(atoms, mask=[1,1,1,1,1,1])
##        #opt = BFGS(ecf, trajectory='opt.traj', logfile='bfgs.log')
##        new_cell = update_cell(cell, stress_tensor, cell_step=cell_step)
##        # 2. 保持分数坐标不变，得到新晶胞下的笛卡尔坐标
##        temp_coords_cart = np.dot(coords_frac, new_cell)
##        # 3. 在临时笛卡尔坐标上，根据力更新原子位置
##        new_coords_cart = update_atoms(temp_coords_cart, forces, step_size=step_size)
##    else:
##        #opt = BFGS(atoms, trajectory='opt.traj', logfile='bfgs.log')
##        #opt = FIRE(atoms, maxstep=0.5, trajectory='opt.traj', logfile='fire.log')
##        new_coords_cart = update_atoms(coords_cart, forces, step_size=step_size)
##        new_cell = cell
#
##    # 只运行一步优化
##    opt.run(fmax=force_threshold, steps=1)
##
##    # 获取优化后的结构（笛卡尔坐标和晶胞）
##    new_cell = atoms.get_cell()
##    new_coords_cart = atoms.get_positions()
#
#    # 写回 CONTCAR（保持原坐标类型）
#    write_poscar('CONTCAR', new_cell, species, natoms_per_species, coord_type, new_coords_cart, poscar_lines)
#    print("新结构已写入 CONTCAR")
#
#    # 更新 CHGCAR 头部
#    update_chgcar_structure("CHGCAR", "CONTCAR")
#
#    sys.exit(1)   # 未收敛，继续循环

def main():
    # ---------- 参数解析 ----------
    force_threshold = 0.01
    iteration = 1
    intermediate_dir = "../intermediate_dir"
    optimizer_name = "FIRE"   # 可选 "BFGS" 或 "FIRE"

    if len(sys.argv) > 1:
        try:
            force_threshold = float(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        try:
            iteration = int(sys.argv[2])
        except ValueError:
            pass
    if len(sys.argv) > 3:
        intermediate_dir = sys.argv[3]

    print(f"Iteration: {iteration}, force threshold: {force_threshold}")

    # 读取 INCAR 中的 ISIF
    isif = read_incar_isif('INCAR')
    print(f"ISIF = {isif}")

    # 读取当前结构
    cell, species, natoms_per_species, coord_type, coords_cart, poscar_lines = read_poscar('POSCAR')
    symbols = []
    for s, n in zip(species, natoms_per_species):
        symbols.extend([s] * n)
    atoms = Atoms(symbols=symbols, cell=cell, pbc=True)
    atoms.set_positions(coords_cart)

    # 创建计算器，读取当前 OUTCAR 的力、能量
    calc = VASPOutcarCalculator('OUTCAR')
    atoms.set_calculator(calc)

    forces = atoms.get_forces()
    max_force = np.max(np.linalg.norm(forces, axis=1))
    print(f"Current max force = {max_force:.6f} eV/Å")

    # 收敛判断
    stress_tensor, pressure = parse_stress('OUTCAR')
    print(f"External pressure = {pressure:.2f} kB")
    if isif == 2 or isif == 0:
        if max_force < force_threshold:
            print(f"Converged: max force {max_force:.6f} < {force_threshold}")
            sys.exit(0)
    elif isif == 3:
        if max_force < force_threshold and abs(pressure) < 0.1:
            print(f"Converged: max force {max_force:.6f} < {force_threshold} and pressure < 0.1 kB")
            sys.exit(0)

    # ---------- ASE 优化器状态文件路径 ----------
    opt_state_file = os.path.join(intermediate_dir, "ase_opt_state.pkl")

    # ---------- 初始化或恢复优化器 ----------
    # 我们使用一个抽象的 Dynamics 类，但实际上可以直接用 BFGS/FIRE 并保存其内部 dict
    if iteration == 1 or not os.path.isfile(opt_state_file):
        # 新建优化器
        if optimizer_name == "BFGS":
            opt = BFGS(atoms)
        else:
            # FIRE 对于初始步长有默认值，可设置 maxstep
            opt = FIRE(atoms, maxstep=0.1)   # 限制最大步长 0.1 Å
        print(f"Created new {optimizer_name} optimizer.")
    else:
        # 从文件恢复优化器状态
        if optimizer_name == "BFGS":
            opt = BFGS(atoms)
        else:
            opt = FIRE(atoms, maxstep=0.1)
        with open(opt_state_file, 'rb') as f:
            state = pickle.load(f)
        opt.vel = state.get('vel', None)
        opt.dt = state['dt']
        opt.a = state.get('a', 0.1)          # 兼容旧存档
        opt.Nsteps = state.get('Nsteps', 0)
        print(f"Restored optimizer state from {opt_state_file}")

    # ---------- 执行一步优化 ----------
    try:
        opt.step()   # 根据当前力更新原子位置
    except Exception as e:
        print(f"Optimizer step failed: {e}")
        # 如果出错，回退到简单的沿力方向移动
        fallback_step = 0.001
        new_coords_cart = coords_cart + fallback_step * forces
        atoms.set_positions(new_coords_cart)
        # 重置优化器状态（下次重新开始）
        if os.path.exists(opt_state_file):
            os.remove(opt_state_file)

    # 保存优化器状态
    vel = getattr(opt, 'vel', None)
    state = {
        'vel': vel,
        'dt': opt.dt,
        'a': opt.a,
        'Nsteps': opt.Nsteps,
    }
    with open(opt_state_file, 'wb') as f:
        pickle.dump(state, f)

    # 获取最终坐标并写回 CONTCAR
    new_coords_cart = atoms.get_positions()
    new_cell = atoms.get_cell()  # 目前未处理晶胞变化，保持原胞
    if isif == 3:
        # 简单应力驱动晶胞更新（保留原逻辑，但通常 ASE 的 ExpCellFilter 才完全整合）
        new_cell = update_cell(cell, stress_tensor, cell_step=0.00001)
        # 坐标变换：保持分数坐标不变
        coords_frac = np.dot(new_coords_cart, np.linalg.inv(cell))
        new_coords_cart = np.dot(coords_frac, new_cell)
        atoms.set_cell(new_cell)
        atoms.set_positions(new_coords_cart)

    write_poscar('CONTCAR', new_cell, species, natoms_per_species, coord_type, new_coords_cart, poscar_lines)
    print("New structure written to CONTCAR")

    # 更新 CHGCAR 头部
    update_chgcar_structure("CHGCAR", "CONTCAR")

    # 未收敛，返回 1 使 bash 继续循环
    sys.exit(1)

if __name__ == '__main__':
    main()
