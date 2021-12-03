#!/bin/bash
#SBATCH -c 8
#SBATCH --output logs/pregen_embs_%A.log
#SBATCH --mem 85gb
#SBATCH --job-name=pregen
#SBATCH --ntasks-per-node=4
#SBATCH --mem=50G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00

set -e
#source activate wmlce-ea

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

bash pregen_embs.sh

