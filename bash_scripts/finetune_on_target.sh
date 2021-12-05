#!/bin/bash
#SBATCH -c 2
#SBATCH --job-name=PF
#SBATCH --mem=10GB
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu
#SBATCH --partition=normal
#SBATCH --array=0-27


#SBATCH -D ./log/

# $1 - target type {inhosp_mort, phenotype_first, phenotype_all}
# $2 - BERT model name {baseline_clinical_BERT_1_epoch_512, adv_clinical_BERT_1_epoch_512}
# $3 - target column name within the dataframe, ex: "Shock", "any_acute"

set -e 
#source activate wmlce-ea

BASE_DIR="/om/user/shobhita/src/HurtfulWords"
OUTPUT_DIR="/om/user/shobhita/data/6.864"

cd "$BASE_DIR/scripts"

singularity exec -B /om:/om --nv /om/user/shobhita/singularity/hurtfulwords_latest.sif \
python finetune_on_target.py \
  --idx ${SLURM_ARRAY_TASK_ID} \
  --task_name phenotype_first \
	--fold_id 9 10\
	--freeze_bert \
	--train_batch_size 32 \
	--task_type binary \
	--other_fields age sofa sapsii_prob sapsii_prob oasis oasis_prob \
        --gridsearch_classifier \
        --gridsearch_c \
        --emb_method cat4
