#!/bin/bash
#SBATCH -c 2
#SBATCH --exclude=node021,node037
#SBATCH --job-name=PA_race_true
#SBATCH --mem=50GB
#SBATCH -t 15:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu
#SBATCH --partition=cbmm
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
/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /om/user/shobhita/singularity/hurtfulwords_latest.sif \
python finetune_on_target.py \
  --idx ${SLURM_ARRAY_TASK_ID} \
  --task_name phenotype_all \
	--fold_id 9 10\
	--max_num_epochs 20 \
	--train_batch_size 32 \
	--task_type binary \
	--other_fields age sofa sapsii_prob sapsii_prob oasis oasis_prob \
  --gridsearch_c \
        --emb_method cat4 \
        --protected_group "ethnicity_to_use" \
  --overwrite \
  	--gridsearch_classifier \
  	--freeze_bert \
  	  --use_dro "True"
#  --test_script "True" \

