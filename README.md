# extract FASTA information from GFF files
python gff_to_fasta.py --input_folder gff_files_generated/ --output_folder extracted_fasta

# process FASTA files with prediction tools
python run_minced.py --input_folder extracted_fasta --output_folder minced_gff --update_gff gff_files_with_crispr_pred

python run_crt.py --input_folder extracted_fasta --output_folder crt_gff --update_gff gff_files_with_crispr_pred


python run_crt.py --input_folder extracted_fasta --output_folder crt_gff --update_gff gff_files_generated --output_gff gff_files_with_crispr_pred





gff_files_generated --output_gff gff_files_with_crispr_pred


python run_crt.py --input_folder extracted_fasta --output_folder crt_gff


export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"

java -cp CRT1.2-CLI.jar crt extracted_fasta/GCA_000013345.1_ASM1334v1_genomic.fasta test123



# run crisprcasfinder on AIME, transfer json to output_json/
python reformat_cplus.py --input_folder output_json --output_folder gff_files_cplus



# combine


python combine_gff.py \
    --gff_folders gff_files_generated \
    --extra_gff_folders crt_gff \
    --output_folder merged_gff \
    --log DEBUG


  python summarize_crisprs.py --input_folder output_json --output_file summary_table.csv


python predict_crispr.py \
    --summary_table summary_table.csv \
    --metadata_table limited.csv \
    --model random_forest \
    --test_size 0.2 \
    --output_folder model_output



python predict_crispr.py \
    --summary_table summary_table.csv \
    --metadata_table limited.csv \
    --model linear \
    --test_size 0.2 \
    --output_folder model_output


python predict_crispr.py \
--summary_table summary_table.csv \
--metadata_table limited.csv \
--taxa_table taxa.csv \
--model linear \
--test_size 0.2 \
--target_variable average_crispr_length \
--output_folder model_output_rf