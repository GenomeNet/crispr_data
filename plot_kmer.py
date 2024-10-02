import pandas as pd
import numpy as np
import argparse
import os
import json
import sys  # Ensure sys is imported
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import umap.umap_ as umap
from collections import defaultdict
import logging

def bin_numerical_attribute(df, column, bins=5):
    """
    Bin a numerical column into quantile-based categories with range labels.

    Args:
        df (pd.DataFrame): The DataFrame containing the column.
        column (str): The name of the numerical column to bin.
        bins (int): Number of quantile-based bins.

    Returns:
        pd.Series: A categorical series with binned categories labeled as "start-end".
    """
    try:
        bin_details = pd.qcut(df[column], q=bins, duplicates='drop', retbins=True)
        binned = bin_details[0]
        bin_edges = bin_details[1]

        # Create custom labels based on bin edges
        labels = [f"{round(bin_edges[i], 2)}-{round(bin_edges[i+1], 2)}" for i in range(len(bin_edges)-1)]
        binned = pd.cut(df[column], bins=bin_edges, labels=labels, include_lowest=True)

        return binned
    except ValueError as e:
        logging.error(f"Error binning column '{column}': {e}")
        # Fallback to generic labels if binning fails
        labels = [f"Bin{i}" for i in range(1, bins + 1)]
        return pd.qcut(df[column], q=bins, labels=labels, duplicates='drop')

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
    Flatten nested k-mer dictionaries into a single dictionary.

    Args:
        kmer_dict (dict): Nested dictionary containing k-mer frequencies.
        k_sizes (list): List of desired k-mer sizes.

    Returns:
        dict: Flattened dictionary with k-mer frequencies.
    """
    return get_kmer_frequencies(kmer_dict, k_sizes)

def kmer_dict_to_vector(kmer_dict, all_kmers):
    """
    Convert a k-mer frequency dictionary to a vector based on a list of all possible k-mers.

    Args:
        kmer_dict (dict): Dictionary containing k-mer frequencies.
        all_kmers (list): List of all possible k-mers.

    Returns:
        list: Frequency vector corresponding to all_kmers.
    """
    return [kmer_dict.get(kmer, 0.0) for kmer in all_kmers]

def safe_json_loads(x):
    """
    Safely parse a JSON string or return the dict if already parsed.

    Args:
        x: The object to parse.

    Returns:
        dict or original object if parsing fails.
    """
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            logging.error(f"JSONDecodeError for value: {x}")
            return {}
    elif isinstance(x, dict):
        return x
    else:
        logging.error(f"Unexpected type {type(x)} for value: {x}")
        return {}

def generate_umap_plot(feature_matrix, metadata, hue, output_path, data_col, hue_type='categorical', log_scale=False, palette=None, all_kmers=None, output_folder=None):
    """
    Perform UMAP dimensionality reduction, train a model to predict the metadata,
    generate a scatter plot annotated with model performance metric,
    and save feature importances to a text file.

    Args:
        feature_matrix (np.ndarray): The feature matrix for UMAP and model training.
        metadata (pd.Series): Metadata column for coloring and prediction.
        hue (str): The name of the metadata column.
        output_path (str): Path to save the plot.
        data_col (str): The name of the current data column (e.g., 'left_kmer').
        hue_type (str): Type of hue ('categorical' or 'numerical').
        log_scale (bool): Whether to apply log scaling to numerical hues.
        palette (list or seaborn palette, optional): Color palette for categorical hues.
        all_kmers (list, optional): List of all k-mers corresponding to the feature matrix columns.
        output_folder (str, optional): Folder to save the feature importance files.
    """
    logging.info(f"Generating UMAP plot for hue '{hue}' with type '{hue_type}' and log_scale={log_scale}.")

    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(feature_matrix_scaled)
    logging.info(f"UMAP embedding shape for '{hue}': {embedding.shape}")

    # Prepare the metadata for model training and plotting
    metadata_model = metadata.copy()

    # Handle log scaling for numerical hues
    if hue_type == 'numerical' and log_scale:
        metadata_model = metadata_model.apply(lambda x: np.log1p(x))  # Apply log1p to handle zero values

    # Handle missing values in metadata
    valid_indices = ~metadata_model.isna()
    if not valid_indices.all():
        logging.warning(f"Missing values found in metadata '{hue}'. Excluding {(~valid_indices).sum()} samples.")
        feature_matrix_scaled = feature_matrix_scaled[valid_indices]
        embedding = embedding[valid_indices]
        metadata_model = metadata_model[valid_indices]

    # Train a model to predict the metadata from feature_matrix_scaled
    if hue_type == 'categorical':
        # Encode categories
        if not pd.api.types.is_categorical_dtype(metadata_model):
            metadata_model = metadata_model.astype('category')
        y = metadata_model.cat.codes
        num_classes = len(metadata_model.cat.categories)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix_scaled, y, test_size=0.3, random_state=42, stratify=y)

        # Train a classifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate the classifier
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Classification accuracy for '{hue}': {accuracy:.4f}")

        # **Save feature importances**
        if all_kmers is not None and output_folder is not None:
            importance = clf.feature_importances_
            feature_importances = sorted(zip(all_kmers, importance), key=lambda x: x[1], reverse=True)
            importance_file = os.path.join(output_folder, f"feature_importance_{data_col}_{hue}.txt")
            with open(importance_file, 'w') as f:
                for kmer, score in feature_importances:
                    f.write(f"{kmer}\t{score}\n")
            logging.info(f"Feature importances saved to {importance_file}")

        # Prepare text annotation
        performance_text = f"Accuracy: {accuracy:.2f}"

    elif hue_type == 'numerical':
        y = metadata_model.values

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix_scaled, y, test_size=0.3, random_state=42)

        # Train a regressor
        reg = RandomForestRegressor(random_state=42)
        reg.fit(X_train, y_train)

        # Evaluate the regressor
        y_pred = reg.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"R² score for '{hue}': {r2:.4f}")

        # **Save feature importances**
        if all_kmers is not None and output_folder is not None:
            importance = reg.feature_importances_
            feature_importances = sorted(zip(all_kmers, importance), key=lambda x: x[1], reverse=True)
            importance_file = os.path.join(output_folder, f"feature_importance_{data_col}_{hue}.txt")
            with open(importance_file, 'w') as f:
                for kmer, score in feature_importances:
                    f.write(f"{kmer}\t{score}\n")
            logging.info(f"Feature importances saved to {importance_file}")

        # Prepare text annotation
        performance_text = f"R²: {r2:.2f}"

    else:
        logging.error("hue_type must be 'categorical' or 'numerical'.")
        raise ValueError("hue_type must be 'categorical' or 'numerical'.")

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Hue': metadata_model.values
    })

    plt.figure(figsize=(12, 10))

    if hue_type == 'categorical':
        plt.title(f"UMAP Colored by {hue}\n{performance_text}")

        sns.scatterplot(
            x='UMAP1', y='UMAP2',
            hue='Hue',
            data=plot_df,
            palette=palette,
            s=80,
            edgecolor='k',
            alpha=0.7
        )

        plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')

    elif hue_type == 'numerical':
        plt.title(f"UMAP Colored by {'Log-scaled ' if log_scale else ''}{hue}\n{performance_text}")

        scatter = plt.scatter(
            plot_df['UMAP1'],
            plot_df['UMAP2'],
            c=plot_df['Hue'],
            cmap='viridis',
            s=80,
            edgecolor='k',
            alpha=0.7
        )

        cbar = plt.colorbar(scatter)
        if log_scale:
            cbar.set_label(f'Log of {hue}')
        else:
            cbar.set_label(hue)
    else:
        logging.error("hue_type must be 'categorical' or 'numerical'.")
        raise ValueError("hue_type must be 'categorical' or 'numerical'.")

    # Annotate the plot with performance metric
    plt.text(0.05, 0.95, performance_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Plot saved to {output_path}")

    
def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    parser = argparse.ArgumentParser(description="Plot k-mer content using UMAP and perform classification.")
    parser.add_argument("--summary_table", required=True, help="Path to the summary table CSV (e.g., summary_table3.csv).")
    parser.add_argument("--metadata_table", required=True, help="Path to the metadata CSV file (e.g., limited.csv).")
    parser.add_argument("--taxa_table", required=True, help="Path to the taxa CSV file (e.g., taxa.csv).")
    parser.add_argument("--output_folder", default="kmer_plots2", help="Folder to save plots and results.")
    parser.add_argument("--kmer_sizes", type=int, nargs='+', default=[3, 5], help="Sizes of the k-mers to plot. Default is [3, 5].")
    parser.add_argument("--num_bins", type=int, default=5, help="Number of bins for quantile-based binning of numerical attributes. Default is 5.")
    parser.add_argument("--top_taxa", type=int, default=6, help="Number of top categories to display for each taxonomic level. Default is 6.")
    parser.add_argument("--top_taxa_count", type=int, nargs='*', default=[], help="Number of top categories for each taxonomic level in the order: Phylum, Class, Order, Family, Genus. If not specified, --top_taxa is used for all.")
    args = parser.parse_args()

    logging.info("Starting the plot_kmer.py script...")

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    logging.info(f"Output will be saved to: {args.output_folder}")

    # Read the summary table
    try:
        summary_df = pd.read_csv(args.summary_table, sep=';')
        logging.info(f"Summary table loaded with {summary_df.shape[0]} samples and {summary_df.shape[1]} columns.")
    except Exception as e:
        logging.error(f"Error reading summary table: {e}")
        sys.exit(1)

    # Read metadata and taxa tables
    try:
        metadata_df = pd.read_csv(args.metadata_table)
        logging.info(f"Metadata table loaded with {metadata_df.shape[0]} samples and {metadata_df.shape[1]} columns.")
    except Exception as e:
        logging.error(f"Error reading metadata table: {e}")
        sys.exit(1)

    try:
        taxa_df = pd.read_csv(args.taxa_table)
        logging.info(f"Taxa table loaded with {taxa_df.shape[0]} samples and {taxa_df.shape[1]} columns.")
    except Exception as e:
        logging.error(f"Error reading taxa table: {e}")
        sys.exit(1)

    # Parse the JSON strings in k-mer frequency and nucleotide distribution columns
    try:
        summary_df['left_kmer_dict'] = summary_df['leftflank_kmer_freq'].apply(safe_json_loads)
        summary_df['right_kmer_dict'] = summary_df['rightflank_kmer_freq'].apply(safe_json_loads)
        summary_df['spacer_kmer_dict'] = summary_df['spacer_kmer_frequencies'].apply(safe_json_loads)
        summary_df['spacer_nuc_dist'] = summary_df['spacer_nucleotide_distribution'].apply(safe_json_loads)
        summary_df['dr_nuc_dist'] = summary_df['dr_nucleotide_distribution'].apply(safe_json_loads)
        logging.info("JSON columns parsed successfully.")
    except Exception as e:
        logging.error(f"Error parsing JSON columns: {e}")
        sys.exit(1)

    # Merge the DataFrames on 'sample'
    if 'sample' not in summary_df.columns or 'sample' not in metadata_df.columns or 'sample' not in taxa_df.columns:
        logging.error("One of the DataFrames does not contain the 'sample' column for merging.")
        sys.exit(1)

    merged_df = summary_df.merge(metadata_df, on='sample', how='left')
    merged_df = merged_df.merge(taxa_df, on='sample', how='left')
    logging.info(f"Merged DataFrame has {merged_df.shape[0]} samples and {merged_df.shape[1]} columns.")

    # Filter out samples where num_crispr_arrays is zero
    if 'num_crispr_arrays' in merged_df.columns:
        initial_count = merged_df.shape[0]
        merged_df = merged_df[merged_df['num_crispr_arrays'] > 0].reset_index(drop=True)
        filtered_count = merged_df.shape[0]
        logging.info(f"Filtered out {initial_count - filtered_count} samples with num_crispr_arrays <= 0.")
    else:
        logging.warning("Column 'num_crispr_arrays' not found in merged DataFrame.")

    # Define taxonomic levels to include
    taxonomic_levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus']

    # Initialize the list of categorical hues with other categorical attributes
    categorical_hues = ['Spore formation', 'Motility']  # Exclude 'Genus' for separate handling

    # Ensure other categorical_hues exist and are of 'category' dtype
    for hue in ['Spore formation', 'Motility']:
        if hue in merged_df.columns:
            merged_df[hue].fillna('Unknown', inplace=True)
            merged_df[hue] = merged_df[hue].astype('category')
            logging.info(f"Distribution of '{hue}':")
            logging.info(merged_df[hue].value_counts())
        else:
            logging.warning(f"Categorical hue '{hue}' not found in merged DataFrame.")

    # Initialize list for numerical hues
    numerical_hues = ['num_crispr_arrays', 'average_crispr_length']  # Adjust based on your data

    # Limit each taxonomic level to top N categories and create new limited columns
    for idx, taxon in enumerate(taxonomic_levels):
        if taxon in merged_df.columns:
            # Replace missing values with 'Unknown'
            merged_df[taxon].fillna('Unknown', inplace=True)

            # Determine top N categories
            if idx < len(args.top_taxa_count):
                top_n = args.top_taxa_count[idx]
            else:
                top_n = args.top_taxa

            top_categories = merged_df[taxon].value_counts().nlargest(top_n).index.tolist()

            # Create new column with limited categories
            limited_col = f"{taxon}_limited"
            merged_df[limited_col] = merged_df[taxon].apply(lambda x: x if x in top_categories else 'Other')

            # Convert to 'category' dtype to enable .cat accessor
            merged_df[limited_col] = merged_df[limited_col].astype('category')

            logging.info(f"Top {top_n} categories for '{taxon}': {top_categories}")
            logging.info(f"Distribution of '{limited_col}':")
            logging.info(merged_df[limited_col].value_counts())

            # Add to categorical_hues
            categorical_hues.append(limited_col)
        else:
            logging.warning(f"Taxonomic level '{taxon}' not found in merged DataFrame.")

    logging.info(f"\nCategorical hues to be plotted: {categorical_hues}")

    # Choose a color palette that can accommodate up to 10 distinct categories
    categorical_palette = sns.color_palette("tab10", n_colors=10)

    # Define a dictionary for specific palettes per taxonomic level (optional)
    # Uncomment and customize if distinct palettes are desired for each taxonomic level
    """
    palette_dict = {
        'Phylum_limited': sns.color_palette("Set1", n_colors=10),
        'Class_limited': sns.color_palette("Set2", n_colors=10),
        'Order_limited': sns.color_palette("Set3", n_colors=10),
        'Family_limited': sns.color_palette("Pastel1", n_colors=10),
        'Genus_limited': sns.color_palette("tab10", n_colors=10)
    }
    """

    # Read data columns for processing
    # Assuming that k-mer frequency columns are named appropriately
    data_columns = {
        'left_kmer_dict': 'Left Flank k-mer Frequencies',
        'right_kmer_dict': 'Right Flank k-mer Frequencies',
        'spacer_kmer_dict': 'Spacer k-mer Frequencies',
        'spacer_nuc_dist': 'Spacer Nucleotide Distribution',
        'dr_nuc_dist': 'DR Nucleotide Distribution'
    }

    for data_col, description in data_columns.items():
        logging.info(f"\nProcessing '{data_col}': {description}")

        if data_col not in merged_df.columns:
            logging.warning(f"Data column '{data_col}' not found in merged DataFrame.")
            continue

        # Handle nucleotide distribution columns separately if needed
        if 'nuc_dist' in data_col:
            # Placeholder: Implement nucleotide distribution handling if necessary
            logging.info(f"Skipping plotting for nucleotide distribution column '{data_col}'. Implement if needed.")
            continue

        # Handle k-mer frequency columns
        if 'kmer_dict' in data_col:
            # Extract k-mer dictionaries
            kmer_dicts = merged_df[data_col].apply(lambda x: flatten_kmer_dict(x, args.kmer_sizes))

            # Get all unique k-mers across all samples
            all_kmers = sorted(set(kmer for d in kmer_dicts for kmer in d.keys()))
            logging.info(f"Total unique k-mers in '{data_col}': {len(all_kmers)}")

            # Convert k-mer dictionaries to vectors
            feature_matrix = kmer_dicts.apply(lambda x: kmer_dict_to_vector(x, all_kmers)).tolist()
            feature_matrix = np.array(feature_matrix)
            logging.info(f"Feature matrix shape for '{data_col}': {feature_matrix.shape}")

            # Proceed only if feature_matrix is not empty
            if feature_matrix.size == 0:
                logging.warning(f"Feature matrix for '{data_col}' is empty. Skipping.")
                continue

            # Perform UMAP and plot for each categorical hue attribute (including limited taxonomic levels)
            for hue in categorical_hues:
                logging.info(f"  - Generating UMAP plot colored by categorical hue '{hue}'.")
                metadata = merged_df[hue]

                # Check for missing values in metadata
                if metadata.isnull().all():
                    logging.warning(f"    Warning: All values in '{hue}' are missing. Skipping.")
                    continue

                output_filename = f"umap_{data_col}_{hue}.png"
                output_path = os.path.join(args.output_folder, output_filename)

                # Optional: Use specific palette per hue
                # palette = palette_dict.get(hue, categorical_palette)  # Uncomment if palette_dict is defined
                palette = categorical_palette  # Using general palette

                generate_umap_plot(
                    feature_matrix=feature_matrix,
                    metadata=metadata,
                    hue=hue,
                    output_path=output_path,
                    data_col=data_col,          # Added data_col argument
                    hue_type='categorical',
                    palette=palette,
                    all_kmers=all_kmers,
                    output_folder=args.output_folder
                )

            # Optionally, perform UMAP and plot for each numerical hue attribute with log scaling
            for hue in numerical_hues:
                logging.info(f"  - Generating UMAP plot with log-scaled numerical hue '{hue}'.")
                metadata = merged_df[hue]

                # Check for missing values in metadata
                if metadata.isnull().all():
                    logging.warning(f"    Warning: All values in '{hue}' are missing. Skipping.")
                    continue

                output_filename = f"umap_{data_col}_{hue}_log.png"
                output_path = os.path.join(args.output_folder, output_filename)

                generate_umap_plot(
                    feature_matrix=feature_matrix,
                    metadata=metadata,
                    hue=hue,
                    output_path=output_path,
                    data_col=data_col,          # Added data_col argument
                    hue_type='numerical',
                    log_scale=True,
                    all_kmers=all_kmers,
                    output_folder=args.output_folder
                )

    logging.info("\nAll plots have been generated.")

if __name__ == "__main__":
    main()