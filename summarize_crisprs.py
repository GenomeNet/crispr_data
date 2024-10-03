import os
import json
import argparse
import csv
from statistics import mean
from collections import defaultdict, Counter
from Bio.Seq import Seq
import numpy as np

def get_kmer_frequencies(sequences, k):
    """Compute k-mer frequencies for a list of sequences."""
    kmer_counter = Counter()
    for seq in sequences:
        seq = seq.upper().replace('N', '')  # Remove ambiguous bases
        seq_obj = Seq(seq)
        for i in range(len(seq_obj) - k + 1):
            kmer = str(seq_obj[i:i+k])
            if len(kmer) == k and all(base in 'ACGT' for base in kmer):
                kmer_counter[kmer] += 1
    total_kmers = sum(kmer_counter.values())
    if total_kmers == 0:
        return {}
    # Calculate frequency as percentage
    kmer_freq = {kmer: round((count / total_kmers) * 100, 2) for kmer, count in kmer_counter.items()}
    return kmer_freq

def get_nucleotide_distribution(sequences):
    """Compute nucleotide frequencies for a list of sequences."""
    total_length = sum(len(seq) for seq in sequences)
    nucleotide_counts = Counter()
    for seq in sequences:
        nucleotide_counts.update(seq.upper())
    if total_length == 0:
        return {}
    nucleotide_freq = {nuc: round((count / total_length) * 100, 2)
                       for nuc, count in nucleotide_counts.items()}
    return nucleotide_freq

def process_json_file(json_file, k_list=[3]):
    """
    Process a single JSON file to extract CRISPR information and calculate summary statistics, including k-mer profiles.

    Args:
        json_file (str): Path to the JSON file.
        k_list (list): Sizes of the k-mers. Default is [3].

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

    # New data collections
    dr_lengths = []
    spacer_counts = []
    array_lengths = []
    conservation_drs = []
    conservation_spacers = []
    dr_consensuses = []
    leftflank_sequences = []
    rightflank_sequences = []
    spacer_sequences = []
    
    # Collection for Leader FLANKs
    leader_flank_sequences = []

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
            array_lengths.append(length)

            orientation = crispr.get('Potential_Orientation', 'unknown')
            if orientation in ['+', '-']:
                orientation_counts[orientation] += 1
            else:
                orientation_counts['unknown'] += 1

            # Collect new stats
            dr_length = crispr.get('DR_Length', 0)
            dr_lengths.append(dr_length)

            num_spacers = crispr.get('Spacers', 0)
            spacer_counts.append(num_spacers)

            conservation_dr = crispr.get('Conservation_DRs', 0.0)
            conservation_drs.append(conservation_dr)

            conservation_spacer = crispr.get('Conservation_Spacers', 0.0)
            conservation_spacers.append(conservation_spacer)

            dr_consensus = crispr.get('DR_Consensus', '')
            if dr_consensus:
                dr_consensuses.append(dr_consensus)

            # Collect k-mer sequences of LeftFLANK and RightFLANK
            regions = crispr.get('Regions', [])
            for region in regions:
                region_type = region.get('Type')
                sequence_seq = region.get('Sequence', '')
                if not sequence_seq:
                    continue

                if region_type == 'LeftFLANK':
                    leftflank_sequences.append(sequence_seq)
                elif region_type == 'RightFLANK':
                    rightflank_sequences.append(sequence_seq)
                elif region_type == 'Spacer':
                    spacer_sequences.append(sequence_seq)

                # Check if Leader is 1 at the region level
                if region.get('Leader', 0) == 1:
                    leader_flank_sequences.append(sequence_seq)

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

    # New summary statistics with checks to prevent NumPy warnings
    if len(dr_lengths) > 1:
        avg_dr_length = mean(dr_lengths)
        sd_dr_length = np.std(dr_lengths, ddof=1)
    elif len(dr_lengths) == 1:
        avg_dr_length = dr_lengths[0]
        sd_dr_length = 0.0
    else:
        avg_dr_length = 0.0
        sd_dr_length = 0.0

    if len(spacer_counts) > 1:
        avg_spacers = mean(spacer_counts)
        sd_spacers = np.std(spacer_counts, ddof=1)
    elif len(spacer_counts) == 1:
        avg_spacers = spacer_counts[0]
        sd_spacers = 0.0
    else:
        avg_spacers = 0.0
        sd_spacers = 0.0

    if len(array_lengths) > 1:
        avg_array_length = mean(array_lengths)
        sd_array_length = np.std(array_lengths, ddof=1)
    elif len(array_lengths) == 1:
        avg_array_length = array_lengths[0]
        sd_array_length = 0.0
    else:
        avg_array_length = 0.0
        sd_array_length = 0.0

    if len(conservation_drs) > 1:
        avg_conservation_drs = mean(conservation_drs)
        sd_conservation_drs = np.std(conservation_drs, ddof=1)
    elif len(conservation_drs) == 1:
        avg_conservation_drs = conservation_drs[0]
        sd_conservation_drs = 0.0
    else:
        avg_conservation_drs = 0.0
        sd_conservation_drs = 0.0

    if len(conservation_spacers) > 1:
        avg_conservation_spacers = mean(conservation_spacers)
        sd_conservation_spacers = np.std(conservation_spacers, ddof=1)
    elif len(conservation_spacers) == 1:
        avg_conservation_spacers = conservation_spacers[0]
        sd_conservation_spacers = 0.0
    else:
        avg_conservation_spacers = 0.0
        sd_conservation_spacers = 0.0

    # DR Consensus k-mer profile
    dr_consensus_counts = Counter(dr_consensuses)
    if dr_consensus_counts:
        most_common_dr = dr_consensus_counts.most_common(1)[0][0]
    else:
        most_common_dr = ''

    # Compute nucleotide distributions using the global function
    dr_nt_distribution = get_nucleotide_distribution(dr_consensuses)
    dr_nt_distribution_json = json.dumps(dr_nt_distribution)

    spacer_nt_distribution = get_nucleotide_distribution(spacer_sequences)
    spacer_nt_distribution_json = json.dumps(spacer_nt_distribution)

    # Compute k-mer frequencies for spacers using user-specified k_list
    spacer_kmer_freqs = {}
    for k in k_list:
        spacer_kmer_freqs[f'{k}-mer'] = get_kmer_frequencies(spacer_sequences, k)
    spacer_kmer_json = json.dumps(spacer_kmer_freqs)

    # Collect k-mer frequencies for LeftFLANK and RightFLANK
    leftflank_kmer_freqs = {}
    rightflank_kmer_freqs = {}
    for k in k_list:
        leftflank_kmer_freqs[f'{k}-mer'] = get_kmer_frequencies(leftflank_sequences, k)
        rightflank_kmer_freqs[f'{k}-mer'] = get_kmer_frequencies(rightflank_sequences, k)
    leftflank_kmer_json = json.dumps(leftflank_kmer_freqs)
    rightflank_kmer_json = json.dumps(rightflank_kmer_freqs)

    # Compute k-mer frequencies for Leader FLANKs
    leader_flank_kmer_freqs = {}
    for k in k_list:
        leader_flank_kmer_freqs[f'{k}-mer'] = get_kmer_frequencies(leader_flank_sequences, k)
    leader_flank_kmer_json = json.dumps(leader_flank_kmer_freqs)

    # Generate sample name by replacing .json with .fasta
    sample_name = os.path.basename(json_file).replace('.json', '.fasta')

    summary = {
        'sample': sample_name,
        'num_crispr_arrays': total_crispr_count,
        'average_crispr_length': round(average_crispr_length, 2),
        'average_AT': round(average_at, 2),
        'percent_plus_orientation': round(percent_plus, 2),
        'percent_minus_orientation': round(percent_minus, 2),
        'average_DR_length': round(avg_dr_length, 2),
        'sd_DR_length': round(sd_dr_length, 2) if not np.isnan(sd_dr_length) else 0.0,
        'average_spacers': round(avg_spacers, 2),
        'sd_spacers': round(sd_spacers, 2) if not np.isnan(sd_spacers) else 0.0,
        'average_array_length': round(avg_array_length, 2),
        'sd_array_length': round(sd_array_length, 2) if not np.isnan(sd_array_length) else 0.0,
        'average_conservation_DRs': round(avg_conservation_drs, 2),
        'sd_conservation_DRs': round(sd_conservation_drs, 2) if not np.isnan(sd_conservation_drs) else 0.0,
        'average_conservation_spacers': round(avg_conservation_spacers, 2),
        'sd_conservation_spacers': round(sd_conservation_spacers, 2) if not np.isnan(sd_conservation_spacers) else 0.0,
        'most_common_DR_consensus': most_common_dr,
        'dr_nucleotide_distribution': dr_nt_distribution_json,
        'spacer_nucleotide_distribution': spacer_nt_distribution_json,
        'spacer_kmer_frequencies': spacer_kmer_json,
        'leftflank_kmer_freq': leftflank_kmer_json,
        'rightflank_kmer_freq': rightflank_kmer_json,
        'leader_flank_kmer_freq': leader_flank_kmer_json,  # New Column
    }

    return summary

def summarize_crisprs(input_folder, output_file, k_list=[3]):
    """
    Summarize CRISPR information from all JSON files in the input folder, including k-mer profiles.

    Args:
        input_folder (str): Directory containing CRISPRCasFinder JSON files.
        output_file (str): Path to the output CSV file.
        k_list (list): Sizes of the k-mers. Default is [3].
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
        summary = process_json_file(json_file, k_list)
        if summary:
            summaries.append(summary)
    
    if not summaries:
        print("No summaries to write.")
        return
    
    # Dynamically generate headers from the first summary dict
    headers = list(summaries[0].keys())
    
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=';')
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)
    
    print(f"Summary table created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Summarize CRISPR information from CRISPRCasFinder JSON outputs into a table, including k-mer profiles.")
    parser.add_argument(
        "--input_folder",
        required=True,
        help="Input folder containing CRISPRCasFinder JSON files."
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output file path for the summary table (e.g., summary_table2.csv)."
    )
    parser.add_argument(
        "--kmer_sizes",
        type=int,
        nargs='+',
        default=[3],
        help="Sizes of the k-mers to compute frequencies. Default is 3."
    )
    
    args = parser.parse_args()
    
    summarize_crisprs(args.input_folder, args.output_file, args.kmer_sizes)

if __name__ == "__main__":
    main()