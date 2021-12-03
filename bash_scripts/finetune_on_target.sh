#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --output=logs/finetune_%A.log
#SBATCH --mem 60gb

# $1 - target type {inhosp_mort, phenotype_first, phenotype_all}
# $2 - BERT model name {baseline_clinical_BERT_1_epoch_512, adv_clinical_BERT_1_epoch_512}
# $3 - target column name within the dataframe, ex: "Shock", "any_acute"

set -e 
#source activate wmlce-ea

BASE_DIR="/home/nhulkund/6.864/HurtfulWords"
OUTPUT_DIR="/home/nhulkund/6.864/HurtfulWords/outputs"

cd "$BASE_DIR/scripts"

python finetune_on_target.py \
	--df_path "/nobackup/users/nhulkund/6.864/${1}" \
	--model_path "${OUTPUT_DIR}/models/$2" \
	--fold_id 9 10\
	--target_col_name "$3" \
	--output_dir "${OUTPUT_DIR}/models/finetuned/${1}_${2}_${3}/" \
	--freeze_bert \
	--train_batch_size 32 \
	--pregen_emb_path "/nobackup/users/nhulkund/6.864/pregen_embs/pregen_${2}_cat4_${1}" \
	--task_type binary \
	--other_fields age sofa sapsii_prob sapsii_prob oasis oasis_prob \
        --gridsearch_classifier \
        --gridsearch_c \
        --emb_method cat4 \
  --use_dro True
