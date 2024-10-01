import os
import json
import argparse
import csv
from statistics import mean
from collections import defaultdict

def process_json_file(json_file):
    """
    Process a single JSON file to extract CRISPR information.
    
    Args:
        json_file (str): Path to the JSON file.
    
    Returns:
        dict: A dictionary containing summarized CRISPR information for the sample.
    """
    with open(json_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {json_file}: {e}")
            return None
    
    sequences = data.get('Sequences', [])
    if not sequences:
        print(f"No sequences found in {json_file}.")
        return None
    
    total_crispr_count = 0
    crispr_lengths = []
    at_weighted_sum = 0.0
    total_sequence_length = 0
    orientation_counts = defaultdict(int)
    
    for sequence in sequences:
        seq_length = sequence.get('Length', 0)
        at_percentage = sequence.get('AT', 0.0)
        total_sequence_length += seq_length
        at_weighted_sum += at_percentage * seq_length
        
        crisprs = sequence.get('Crisprs', [])
        num_crisprs = len(crisprs)
        total_crispr_count += num_crisprs
        
        for crispr in crisprs:
            start = crispr.get('Start')
            end = crispr.get('End')
            if start is None or end is None:
                print(f"Missing 'Start' or 'End' in CRISPR entry in {json_file}. Skipping this CRISPR.")
                continue
            length = end - start
            crispr_lengths.append(length)
            
            orientation = crispr.get('Potential_Orientation', 'unknown')
            if orientation in ['+', '-']:
                orientation_counts[orientation] += 1
            else:
                orientation_counts['unknown'] += 1
    
    if total_sequence_length > 0:
        average_at = at_weighted_sum / total_sequence_length
    else:
        average_at = 0.0
    
    if crispr_lengths:
        average_crispr_length = mean(crispr_lengths)
    else:
        average_crispr_length = 0.0
    
    total_orientations = orientation_counts['+'] + orientation_counts['-']
    if total_orientations > 0:
        percent_plus = (orientation_counts['+'] / total_orientations) * 100
        percent_minus = (orientation_counts['-'] / total_orientations) * 100
    else:
        percent_plus = 0.0
        percent_minus = 0.0
    
    # Generate sample name by replacing .json with .fasta
    sample_name = os.path.basename(json_file).replace('.json', '.fasta')
    
    summary = {
        'sample': sample_name,
        'num_crispr_arrays': total_crispr_count,
        'average_crispr_length': round(average_crispr_length, 2),
        'average_AT': round(average_at, 2),
        'percent_plus_orientation': round(percent_plus, 2),
        'percent_minus_orientation': round(percent_minus, 2)
    }
    
    return summary

def summarize_crisprs(input_folder, output_file):
    """
    Summarize CRISPR information from all JSON files in the input folder.
    
    Args:
        input_folder (str): Directory containing CRISPRCasFinder JSON files.
        output_file (str): Path to the output CSV file.
    """
    json_files = [
        os.path.join(input_folder, fname)
        for fname in os.listdir(input_folder)
        if fname.endswith('.json')
    ]
    
    if not json_files:
        print(f"No JSON files found in {input_folder}.")
        return
    
    summaries = []
    for json_file in json_files:
        summary = process_json_file(json_file)
        if summary:
            summaries.append(summary)
    
    # Define CSV headers
    headers = [
        'sample',
        'num_crispr_arrays',
        'average_crispr_length',
        'average_AT',
        'percent_plus_orientation',
        'percent_minus_orientation'
    ]
    
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=';')
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)
    
    print(f"Summary table created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Summarize CRISPR information from CRISPRCasFinder JSON outputs into a table.")
    parser.add_argument(
        "--input_folder",
        required=True,
        help="Input folder containing CRISPRCasFinder JSON files."
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output file path for the summary table (e.g., summary_table.csv)."
    )
    
    args = parser.parse_args()
    
    summarize_crisprs(args.input_folder, args.output_file)

if __name__ == "__main__":
    main()