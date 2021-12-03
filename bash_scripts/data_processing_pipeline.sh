#!/bin/bash
#SBATCH --job-name=data_processing
#SBATCH --partition cpu
#SBATCH -c 30 
#SBATCH --output=data_processing_%A.out
#SBATCH --mem 300gb
#SBATCH -t 120:00:00

set -e
#source activate hurtfulwords

BASE_DIR="/home/nhulkund/6.864/HurtfulWords/"
OUTPUT_DIR="/nobackup/users/nhulkund/6.864/"
mkdir -p "$OUTPUT_DIR/finetuning/"
SCIBERT_DIR="/nobackup/users/nhulkund/6.864/scibert_scivocab_uncased/"
MIMIC_BENCHMARK_DIR="/home/nhulkund/6.864/mimic3-benchmarks"

cd "$BASE_DIR/scripts/"

#echo "Processing MIMIC data..."
#python get_data.py $OUTPUT_DIR

echo "Tokenizing sentences..."
python sentence_tokenization.py "$OUTPUT_DIR/df_raw.pkl" "$OUTPUT_DIR/df_extract.pkl" "$SCIBERT_DIR"
rm "$OUTPUT_DIR/df_raw.pkl" 

echo "Grouping short sentences..."
python group_sents.py "$OUTPUT_DIR/df_extract.pkl" "$OUTPUT_DIR/df_grouped.pkl" "$SCIBERT_DIR"

echo "Pregenerating training data..."
python pregenerate_training_data.py \
	--train_df "$OUTPUT_DIR/df_grouped.pkl" \
	--col_name "BERT_sents20" \
	--output_dir "$OUTPUT_DIR/pregen_epochs/128/" \
	--bert_model "$SCIBERT_DIR" \
	--epochs_to_generate 1 \
	--max_seq_len 128 

python pregenerate_training_data.py \
	--train_df "$OUTPUT_DIR/df_grouped.pkl" \
	--col_name "BERT_sents20" \
	--output_dir "$OUTPUT_DIR/pregen_epochs/512/" \
	--bert_model "$SCIBERT_DIR" \
	--epochs_to_generate 1 \
	--max_seq_len 512 

echo "Generating finetuning targets..."
python make_targets.py \
	--processed_df "$OUTPUT_DIR/df_extract.pkl" \
	--mimic_benchmark_dir "$MIMIC_BENCHMARK_DIR" \
	--output_dir "$OUTPUT_DIR/finetuning/"

# rm "$OUTPUT_DIR/df_extract.pkl" 
# rm "$OUTPUT_DIR/df_grouped.pkl" 
