#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --partition=gpuv100,gpu2080,gputitanrtx,vis-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --mem=22G
#SBATCH --export=NONE

 
 
echo "Starte Job auf Partition:"
echo $SLURM_JOB_PARTITION
echo `hostname`


# NEUE TF Version
#module load palma/2020b
#module load GCC/10.2.0
#module load CUDA/11.1.1
#module load OpenMPI/4.0.5
#module load TensorFlow/2.4.1


# ALTE TF Version
#module load palma/2019a
#module load GCC/8.2.0-2.31.1
#module load CUDA/10.1.105
#module load OpenMPI/3.1.3
#module load TensorFlow/1.13.1-Python-3.7.2

# Torch
module load palma/2019a
module load GCCcore/8.2.0
module load GCC/8.2.0-2.31.1
module load CUDA/10.1.105
module load Python/3.7.2



# Python 3.8.6 bzw 3.7.2 venv laden abhaengig von der Architektur des ausgewaehlten Nodes

# Fuer gpuV100, normal und vis-gpu (gputitanxp)
# vis-gpu unterstuetzt nur CUDA 10.2, nicht CUDA 11.1.1  -> Nicht verwenden fuer neue TF Version

if [ "$SLURM_JOB_PARTITION" == "vis-gpu" ] || [ "$SLURM_JOB_PARTITION" == "gpuv100" ]
then
	echo "Lade Python3-Venv fuer GPUV100 oder vis-gpu (gpuTitanXP) -- Skylake"
	#source /scratch/tmp/m_wolf37/Bachelorarbeit/venv_skylake_3-8-6/bin/activate
	source /scratch/tmp/m_wolf37/Bachelorarbeit/venv_skylake_3-7-2/bin/activate


# Fuer gpuTitanRTX und gpu2080
elif [ "$SLURM_JOB_PARTITION" == "gputitanrtx" ] || [ "$SLURM_JOB_PARTITION" == "gpu2080" ]
then
	echo "Lade Python3-Venv fuer GPUTitanRTX oder GPU2080 -- Ivy-/Sandybridge"
	#source /scratch/tmp/m_wolf37/Bachelorarbeit/venv_ivysandybridge_3-8-6/bin/activate
	source /scratch/tmp/m_wolf37/Bachelorarbeit/venv_ivysandybridge_3-7-2/bin/activate

# Falls Partition keine von den oberen ist
else
	echo "Falsche Partition!"
	echo $SLURM_JOB_PARTITION

fi


echo "uebergebene Parameter"
echo "Datensatz: " ${1}
echo "Fold" ${2}
echo "is_ovo" ${3}
echo "net_type" ${4}
echo "is_finetune" ${5}
echo "train_percent" ${6}
echo "epochs" ${7}
echo "initiale learning_rate" ${8}
echo "img_size" ${9}
echo "END ARGS 9"


#python3 /scratch/tmp/m_wolf37/Bachelorarbeit/train.py --dataset ${1} --fold ${2} --is_ovo ${3} --net_type ${4} --is_finetune ${5} --train_percent ${6} --epochs ${7} --learning_rate ${8} --img_size ${9} --extra_info TF1-13-1-detTS
#python3 /scratch/tmp/m_wolf37/Bachelorarbeit/train.py --dataset ${1} --fold ${2} --is_ovo ${3} --net_type ${4} --is_finetune ${5} --train_percent ${6} --epochs ${7} --learning_rate ${8} --img_size ${9} --extra_info TF2-4-1-detTS

python3 /scratch/tmp/m_wolf37/Bachelorarbeit/train_torch.py --dataset ${1} --fold ${2} --is_ovo ${3} --net_type ${4} --is_finetune ${5} --train_percent ${6} --epochs ${7} --learning_rate ${8} --img_size ${9} --extra_info torch

echo `hostname`
