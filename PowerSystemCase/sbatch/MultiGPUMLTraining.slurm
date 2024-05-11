#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G        # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=05:05:00           # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=xudong.fan@princeton.edu
#SBATCH --constraint=gpu80

module purge
module load anaconda3/2023.3

conda activate MLSERN

python /home/xf3376/PROJ-2023-MLSERN/PowerSystemCase/scripts/predictGraph_FSDP.py
# python /home/xf3376/PROJ-2023-MLSERN/scripts/validateModel.pyr