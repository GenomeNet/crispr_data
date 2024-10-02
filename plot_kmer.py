import pandas as pd
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from collections import defaultdict

def get_kmer_frequencies(kmer_dict, k_sizes):
    """
    Extract k-mer frequencies for specified k sizes.

    Args:
        kmer_dict (dict): Dictionary containing k-mer frequencies.
        k_sizes (list): List of desired k-mer sizes.

    Returns:
        dict: Combined k-mer frequencies for the specified sizes.
    """
    combined_freqs = {}
    for k in k_sizes:
        current_kmer = f"{k}-mer"
        if current_kmer in kmer_dict:
            for kmer, freq in kmer_dict[current_kmer].items():
                combined_freqs[kmer] = combined_freqs.get(kmer, 0.0) + freq
    return combined_freqs

def flatten_kmer_dict(kmer_dict, k_sizes):
    """
    Flatten a nested kmer_dict containing multiple k-mer sizes into a single dict.
    Each key is the k-mer itself if its size matches the desired k sizes.

    Args:
        kmer_dict (dict): Nested kmer dict e.g., {'2-mer': {'CG': 4.1, ...}, '3-mer': {...}}
        k_sizes (list): List of k-mer sizes to include.

    Returns:
        dict: Flattened k-mer frequencies for the specified sizes.
    """
    flat_kmer_freqs = {}
    for k in k_sizes:
        key = f"{k}-mer"
        if key in kmer_dict:
            for kmer, freq in kmer_dict[key].items():
                flat_kmer_freqs[kmer] = flat_kmer_freqs.get(kmer, 0.0) + freq
    return flat_kmer_freqs

def kmer_dict_to_vector(kmer_freqs, all_kmers):
    """
    Convert a k-mer frequency dictionary to a vector based on all_kmers.

    Args:
        kmer_freqs (dict): Dictionary of k-mer frequencies.
        all_kmers (list): List of all unique k-mers.

    Returns:
        list: List of frequencies ordered according to all_kmers.
    """
    return [kmer_freqs.get(kmer, 0.0) for kmer in all_kmers]

def parse_json_column(df, column_name):
    """
    Parse a JSON column into dictionaries.

    Args:
        df (pd.DataFrame): The DataFrame containing the column.
        column_name (str): The name of the column to parse.

    Returns:
        pd.Series: A pandas Series of dictionaries.
    """
    try:
        return df[column_name].apply(json.loads)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in column '{column_name}': {e}")
        sys.exit(1)

def generate_umap_plot(feature_matrix, metadata, hue, output_path):
    """
    Perform UMAP dimensionality reduction and generate a scatter plot.

    Args:
        feature_matrix (np.ndarray): The feature matrix for UMAP.
        metadata (pd.Series): Metadata column for coloring.
        hue (str): The name of the metadata column.
        output_path (str): Path to save the plot.
    """
    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(feature_matrix_scaled)
    print(f"UMAP embedding shape: {embedding.shape}")

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Hue': metadata
    })

    # Check if hue has more than a certain number of categories to adjust palette
    unique_values = metadata.unique()
    if len(unique_values) > 10:
        palette = 'viridis'
    else:
        palette = 'tab10'

    # Plotting with dynamic palette
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='UMAP1', y='UMAP2',
        hue='Hue',
        data=plot_df,
        palette=palette,
        s=80,
        edgecolor='k',
        alpha=0.7
    )
    plt.title(f"UMAP of {hue} Colored by {hue}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot various frequency distributions using UMAP.")
    parser.add_argument("--summary_table", required=True, help="Path to the summary table CSV (e.g., summary_table3.csv).")
    parser.add_argument("--metadata_table", required=True, help="Path to the metadata CSV file (e.g., limited.csv).")
    parser.add_argument("--taxa_table", required=True, help="Path to the taxa CSV file (e.g., taxa.csv).")
    parser.add_argument("--output_folder", default="kmer_plots2", help="Folder to save plots and results.")
    parser.add_argument("--kmer_sizes", type=int, nargs='+', default=[3, 5], help="Sizes of the k-mers to plot. Default is [3, 5].")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Read the summary table
    try:
        summary_df = pd.read_csv(args.summary_table, sep=';')
        print(f"Summary table loaded with {summary_df.shape[0]} samples and {summary_df.shape[1]} columns.")
    except Exception as e:
        print(f"Error reading summary table: {e}")
        sys.exit(1)

    # Read metadata and taxa tables
    try:
        metadata_df = pd.read_csv(args.metadata_table)
        print(f"Metadata table loaded with {metadata_df.shape[0]} samples and {metadata_df.shape[1]} columns.")
    except Exception as e:
        print(f"Error reading metadata table: {e}")
        sys.exit(1)

    try:
        taxa_df = pd.read_csv(args.taxa_table)
        print(f"Taxa table loaded with {taxa_df.shape[0]} samples and {taxa_df.shape[1]} columns.")
    except Exception as e:
        print(f"Error reading taxa table: {e}")
        sys.exit(1)

    # Verify presence of 'sample' column
    if 'sample' not in summary_df.columns:
        print("Error: 'sample' column not found in the summary table.")
        sys.exit(1)
    if 'sample' not in metadata_df.columns:
        print("Error: 'sample' column not found in the metadata table.")
        sys.exit(1)
    if 'sample' not in taxa_df.columns:
        print("Error: 'sample' column not found in the taxa table.")
        sys.exit(1)

    # Merge the DataFrames
    merged_df = summary_df.merge(metadata_df, on='sample', how='left')
    merged_df = merged_df.merge(taxa_df, on='sample', how='left')
    print(f"Merged DataFrame has {merged_df.shape[0]} samples and {merged_df.shape[1]} columns.")

    # Verify that expected columns exist after merging
    expected_columns = ['Genus', 'Spore formation']
    missing_columns = [col for col in expected_columns if col not in merged_df.columns]
    if missing_columns:
        print(f"Warning: The following expected columns are missing after merging: {', '.join(missing_columns)}")

    # Define the data columns to plot
    data_columns = {
        'rightflank_kmer_freq': 'Right Flank k-mer Frequencies',
        'leftflank_kmer_freq': 'Left Flank k-mer Frequencies',
        'spacer_kmer_frequencies': 'Spacer k-mer Frequencies',
        'spacer_nucleotide_distribution': 'Spacer Nucleotide Distribution',
        'dr_nucleotide_distribution': 'DR Nucleotide Distribution'
    }

    # Dynamically retrieve hue attributes from merged DataFrame
    available_hues = ['Genus', 'Spore formation', 'Motility']  # Adjust based on your data
    hue_attributes = [col for col in available_hues if col in merged_df.columns]

    # Process each data column
    for data_col, description in data_columns.items():
        print(f"\nProcessing '{data_col}': {description}")

        if data_col not in merged_df.columns:
            print(f"Warning: Column '{data_col}' not found in the merged DataFrame.")
            continue

        if 'nucleotide_distribution' in data_col:
            # Handle nucleotide distribution columns
            try:
                freq_dict = merged_df[data_col].apply(json.loads)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in column '{data_col}': {e}")
                continue

            # Convert dictionaries to DataFrame
            freq_df = pd.json_normalize(freq_dict)
            freq_df.fillna(0, inplace=True)

            # Feature matrix and k-mers list
            feature_matrix = freq_df.values
            all_kmers = freq_df.columns.tolist()

        else:
            # Handle k-mer frequency columns
            kmer_dicts = merged_df[data_col].apply(json.loads)

            # Flatten the nested k-mer dictionaries
            kmer_dicts_flat = kmer_dicts.apply(lambda x: flatten_kmer_dict(x, args.kmer_sizes))

            # Extract all unique k-mers from flattened dicts
            all_kmers = set()
            for d in kmer_dicts_flat:
                all_kmers.update(d.keys())
            all_kmers = sorted(all_kmers)
            print(f"Total unique k-mers in '{data_col}': {len(all_kmers)}")

            # Convert flattened dictionaries to vectors
            feature_matrix = kmer_dicts_flat.apply(lambda x: kmer_dict_to_vector(x, all_kmers)).tolist()
            feature_matrix = np.array(feature_matrix)

        print(f"Feature matrix shape for '{data_col}': {feature_matrix.shape}")

        # Perform UMAP and plot for each hue attribute
        for hue in hue_attributes:
            print(f"  - Generating UMAP plot colored by '{hue}'.")
            metadata = merged_df[hue]

            output_filename = f"umap_{data_col}_{hue}.png"
            output_path = os.path.join(args.output_folder, output_filename)

            generate_umap_plot(feature_matrix, metadata, hue, output_path)

    print("\nAll plots have been generated.")

if __name__ == "__main__":
    main()