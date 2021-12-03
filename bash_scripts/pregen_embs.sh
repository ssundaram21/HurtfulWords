#!/bin/sh
#BSUB -n 1
#BSUB -gpu "num=4"
#BSUB -gpus-per-node=4
#BSUB -time=24:00:00
#BSUB -o logs/output.log

BASE_DIR="/home/nhulkund/6.864/HurtfulWords/"
OUTPUT_DIR="/home/nhulkund/6.864/HurtfulWords/outputs/"
cd "$BASE_DIR/scripts"
mkdir -p "$OUTPUT_DIR/pregen_embs/"
emb_method='cat4'

echo "starting forlooP!"

for target in inhosp_mort phenotype_first phenotype_all; do
	for model in baseline_clinical_BERT_1_epoch_512; do
	#adv_clinical_BERT_1_epoch_512; do
	  echo "target ${target}, model ${model}"
		python pregen_embeddings.py \
		    --df_path "$OUTPUT_DIR/finetuning/$target"\
		    --model "$OUTPUT_DIR/models/$model" \
		    --output_path "${OUTPUT_DIR}/pregen_embs/pregen_${model}_${emb_method}_${target}" \
		    --emb_method $emb_method
	done
done

