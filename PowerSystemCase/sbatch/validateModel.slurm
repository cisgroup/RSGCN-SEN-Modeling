#!/bin/bash
#SBATCH --job-name=validateModel        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
###SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=xudong.fan@princeton.edu
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --array=0-1

module purge
module load anaconda3/2023.3

conda activate MLSERN

python /home/xf3376/PROJ-2023-MLSERN/PowerSystemCase/scripts/validateModel.py
# python /home/xf3376/PROJ-2023-MLSERN/scripts/validateModel.py