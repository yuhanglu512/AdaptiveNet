#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 110
#SBATCH -o %j.o
#SBATCH -e %j.e
#SBATCH --gres=gpu:4

source ~/yhlu/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
source ~/yhlu/intel/mkl/bin/mklvars.sh intel64
source ~/yhlu/intel/impi/2019.0.117/intel64/bin/mpivars.sh
source ~/yhlu/intel/compilers_and_libraries_2019.0.117/linux/bin/compilervars.sh intel64
source ~/yhlu/intel/compilers_and_libraries_2019.0.117/linux/bin/iccvars.sh intel64
source ~/yhlu/intel/compilers_and_libraries_2019.0.117/linux/bin/ifortvars.sh intel64
source ~/yhlu/intel/compilers_and_libraries_2019.0.117/linux/mkl/bin/mklvars.sh intel64
source ~/miniconda3/etc/profile.d/conda.sh
export PATH=$HOME/yhlu/vasp.5.4.4.pl2/bin:$PATH

export OMP_NUM_THREADS=1
conda activate py311_torch

# ======================= 用户参数 =======================
MAX_ITER=40
FORCE_THRESHOLD=0.01
VASP_CMD="mpirun -np 110 vasp_std"
PYTHON_CMD="python3.11"

CONFIG_DIR="configs/charge3net/"
CONFIG_NAME="test_chgcar_inputs_new.yaml"

# 三个核心目录
ML_INPUT_BASE="./inference"        # 存放转换后的 pkl 输入（按迭代子目录）
ML_OUTPUT_BASE="./test_file"       # ML 预测原始输出，以及 convert 后的完整 CHGCAR
INTERMEDIARY_DIR="./intermediate_dir"  # 存放模板、自洽结果、备份的 CHGCAR
DFT_DIR="relax_DFT"                # VASP 运行目录

INIT_STRUCTURE="POSCAR"
# ========================================================

mkdir -p "$ML_INPUT_BASE" "$ML_OUTPUT_BASE" "$INTERMEDIARY_DIR" "$DFT_DIR"
#rm "$INTERMEDIARY_DIR"/* "$DFT_DIR"/* -r

# ---------------------------------------------------------------
# 步骤 0：初始自洽计算（生成含正确缀加电荷的模板）
# ---------------------------------------------------------------
echo "Step 0: Initial self-consistent calculation"
cp INCAR INCAR.scf "$DFT_DIR/"
cd "$DFT_DIR" || exit 1

if [ ! -f "INCAR.scf" ]; then
    echo "Error: INCAR.scf not found in $DFT_DIR for initial SCF calculation"
    exit 1
fi
cp INCAR.scf INCAR
cp ./POSCAR ../"$INIT_STRUCTURE"
cp ../"$INIT_STRUCTURE" ./POSCAR

# 生成 KPOINTS（示例）
#echo -e '102\n2\n0.02\n' | vaspkit > /dev/null 2>&1
#$VASP_CMD >> log.txt
if [ ! -f "CHGCAR" ]; then
    echo "Initial SCF failed"
    exit 1
fi

# 将初始模板存入 intermediary_dir
cp CHGCAR "../$INTERMEDIARY_DIR/CHGCAR_template_0"
cp CONTCAR "../$INTERMEDIARY_DIR/POSCAR_initial"
cd - || exit 1

# ---------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------
ITER=1
CURRENT_TEMPLATE="$INTERMEDIARY_DIR/CHGCAR_template_0"
PREV_ENERGY=""   # 用于存储上一步的总能量
cp "scripts/optimizer_structure.py" "$DFT_DIR"

while [ $ITER -le $MAX_ITER ]; do
    echo "=========================================="
    echo "Iteration $ITER"
    echo "=========================================="

    # ---- 1. 将当前模板 CHGCAR 拷贝到 inference 目录（作为转换输入） ----
    mkdir -p "$ML_INPUT_BASE/test_loop"
    cp "$CURRENT_TEMPLATE" "$ML_INPUT_BASE/test_loop/CHGCAR"

    # ---- 2. 在 inference 目录中运行 convert，生成 pkl 输入 ----
    rm "$ML_INPUT_BASE"/probe_counts.csv "$ML_INPUT_BASE"/filelist.txt "$ML_INPUT_BASE"/split.json
    $PYTHON_CMD convert_chgcar_dir_to_pkl_dir.py \
        --input="$ML_INPUT_BASE" \
        --output="$ML_INPUT_BASE" \
        --workers=1

    # ---- 3. 运行 ML 预测，输出到 test_file 目录 ----
    $PYTHON_CMD src/inference_from_config.py \
        -cd "$CONFIG_DIR" -cn "$CONFIG_NAME" \
        hydra/launcher=submitit_local

    # ---- 4. 在 test_loop 目录中运行 convert_pkl_to_chgcar，生成完整 CHGCAR ----
    cp "$ML_INPUT_BASE"/test_loop_atoms.pkl "$ML_OUTPUT_BASE"/cubes/
    $PYTHON_CMD convert_pkl_dir_to_chgcar_dir.py \
        "$ML_OUTPUT_BASE" \
        "$ML_INPUT_BASE" \
        "$ML_OUTPUT_BASE" \
        --workers=1

    # 检查生成的文件
    if [ ! -f "$ML_OUTPUT_BASE/test_loop/CHGCAR" ]; then
        echo "Error: convert_pkl_dir_to_chgcar_dir.py did not produce CHGCAR"
        exit 1
    fi

    # ---- 5. 将生成的完整 CHGCAR 拷贝到 intermediary_dir 备份 ----
    mkdir -p "$INTERMEDIARY_DIR/CHGCAR_ml_$ITER"
    cp "$ML_OUTPUT_BASE/test_loop/CHGCAR" "$INTERMEDIARY_DIR/CHGCAR_ml_$ITER"

    # ---- 6. 同时拷贝到 DFT_DIR 用于 VASP 计算 ----
    cp "$ML_OUTPUT_BASE/test_loop/CHGCAR" "$DFT_DIR/CHGCAR"
    cp INCAR "$DFT_DIR/"
    [ -f "WAVECAR" ] && cp WAVECAR "$DFT_DIR/WAVECAR"

    # ---- 7. 运行 VASP 单步离子弛豫 ----
    cd "$DFT_DIR" || exit 1
    $VASP_CMD >> log.txt

    if [ ! -f "OUTCAR" ] || [ ! -f "CHGCAR" ]; then
        echo "VASP failed at iteration $ITER"
        cd -; exit 1
    fi

    # 提取当前步的总能量（从 OSZICAR 中获取最后一步的 free energy）
    if [ ! -f "OSZICAR" ]; then
        echo "Error: OSZICAR not found, cannot extract energy"
        cd -; exit 1
    fi

	# 更新结构和CHGCAR
    $PYTHON_CMD optimizer_structure.py $FORCE_THRESHOLD $ITER "../$INTERMEDIARY_DIR"
    exit_code=$?
	
	# 将 VASP 运行结果（新的 CHGCAR 和 CONTCAR）存入 intermediary_dir
	cp CHGCAR "../$INTERMEDIARY_DIR/CHGCAR_self_$ITER"
    cp CONTCAR "../$INTERMEDIARY_DIR/CONTCAR_$ITER"
    cp OUTCAR "../$INTERMEDIARY_DIR/OUTCAR_$ITER"

    if [ $exit_code -eq 0 ]; then
        echo "Converged at iteration $ITER (max force < $FORCE_THRESHOLD)"
        cp CONTCAR POSCAR
        break
    else
        echo "Not converged, updating structure for next iteration."
        # 将新生成的 CONTCAR 作为下一轮的 POSCAR
        cp CONTCAR POSCAR
    fi
	
    cd - || exit 1
    # ---- 8. 准备下一轮：将本次 VASP 自洽输出的 CHGCAR 作为下一轮的模板 ----
    CURRENT_TEMPLATE="$INTERMEDIARY_DIR/CHGCAR_self_$ITER"
    # 更新 POSCAR（下一轮的结构）
    cp "$INTERMEDIARY_DIR/CONTCAR_$ITER" "$DFT_DIR"/POSCAR

    ITER=$((ITER+1))
done

echo "Finished after $((ITER-1)) iterations. Final structure in CONTCAR"
