# plot_kmer3.py

import pandas as pd
import numpy as np
import argparse
import os
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import umap.umap_ as umap
from collections import defaultdict
import logging
from sklearn.manifold import TSNE
from collections import Counter


def safe_json_loads(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def flatten_kmer_dict(kmer_dict, kmer_sizes):
    """
    Flatten k-mer dictionaries for specified k-mer sizes.
    """
    flattened = {}
    for k in kmer_sizes:
        current_dict = kmer_dict.get(f'{k}-mer', {})
        flattened.update(current_dict)
    return flattened


def kmer_dict_to_vector(kmer_dict, all_kmers):
    """
    Convert a k-mer dictionary to a vector based on all_kmers.
    """
    return [kmer_dict.get(kmer, 0) for kmer in all_kmers]


def bin_numerical_attribute(df, column, bins=5):
    """
    Bin a numerical attribute into quantile-based bins.
    """
    return pd.qcut(df[column], q=bins, duplicates='drop').astype(str)


def limit_categories(series, top_n):
    """
    Limit a series to top_n categories, grouping the rest into 'Other'.
    """
    counts = series.value_counts()
    top_categories = counts.nlargest(top_n).index
    return series.apply(lambda x: x if x in top_categories else 'Other')


def generate_embedding_plots(
    embedding, feature_matrix_scaled, merged_df, hues, data_col, embedding_name,
    output_folder, all_kmers=None, categorical_hues=[], numerical_hues=[],
    model_params={}, num_bins=5, feature_importances_folder=None, top_taxa=6, top_taxa_hues=[],
    exclude_other_na=False, axis_limits=None):
    """
    Generate plots with different hues on the same embedding.

    Args:
        embedding: np.array, the embedding (e.g., UMAP or t-SNE embedding)
        feature_matrix_scaled: np.array, the scaled feature matrix used to compute the embedding
        merged_df: pd.DataFrame, the merged dataframe containing metadata
        hues: list of strings, the hues to plot
        data_col: str, the data column being processed
        embedding_name: str, e.g., 'UMAP' or 't-SNE'
        output_folder: str, the folder to save output plots
        all_kmers: list, optional, needed for feature importance
        categorical_hues: list of categorical hues
        numerical_hues: list of numerical hues
        model_params: dict, optional, parameters for the models
        num_bins: int, number of bins for binning numerical attributes
        feature_importances_folder: str, path to save feature importance files
        top_taxa: int, number of top categories to keep for taxonomic levels
        top_taxa_hues: list of hues that correspond to taxonomic levels
        exclude_other_na: bool, whether to exclude 'Other' and 'NA' groups
        axis_limits: tuple, (x_min, x_max, y_min, y_max) to set consistent axes
    """
    # Use the full merged_df for reference
    full_embedding = embedding  # Original embedding

    # Update the number of hues
    num_hues = len(hues)
    nrows = int(np.ceil(num_hues / 3))  # Up to 3 plots per row
    ncols = min(num_hues, 3)

    # Initialize figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols, 6*nrows), sharex=True, sharey=True)
    if num_hues == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Determine axis limits if not provided
    if axis_limits is None:
        x_min, x_max = np.min(embedding[:, 0]), np.max(embedding[:, 0])
        y_min, y_max = np.min(embedding[:, 1]), np.max(embedding[:, 1])
        axis_limits = (x_min, x_max, y_min, y_max)
        logging.info(f"Axis limits set to: {axis_limits}")

    for idx, hue in enumerate(hues):
        ax = axes[idx]

        metadata = merged_df[hue]
        # Replace missing values with 'NA' for plotting
        plot_metadata = metadata.fillna('NA')
        hue_type = 'categorical' if hue in categorical_hues else ('numerical' if hue in numerical_hues else 'unknown')

        # Handle taxonomic levels by limiting categories
        if hue in top_taxa_hues:
            plot_metadata = limit_categories(plot_metadata, top_taxa)
            logging.info(f"Top {top_taxa} categories for '{hue}': {plot_metadata.unique()}")

        # Handle hue types
        if hue_type == 'numerical':
            # Optionally bin the numerical attribute
            binned_metadata = bin_numerical_attribute(merged_df, hue, bins=num_bins)
            plot_metadata = binned_metadata.fillna('NA')
            hue_type_for_plot = 'categorical'  # For plotting purposes
        else:
            hue_type_for_plot = hue_type

        # Apply filtering if exclude_other_na is True
        if exclude_other_na:
            valid_mask = ~plot_metadata.isin(['Other', 'NA'])
            plot_metadata = plot_metadata[valid_mask]
            emb = full_embedding[valid_mask]
            logging.info(f"Excluded 'Other' and 'NA' groups for hue '{hue}'.")
        else:
            emb = full_embedding

        # For model training, we need to handle missing values
        valid_indices_model = (~merged_df[hue].isna()) & (~merged_df[hue].isin(['Other', 'NA']))
        metadata_model = merged_df[hue][valid_indices_model]
        feature_matrix_valid = feature_matrix_scaled[valid_indices_model]
        embedding_valid = full_embedding[valid_indices_model]

        # Proceed with model training and performance metrics
        if valid_indices_model.sum() < 2:
            logging.warning(f"Not enough valid samples for hue '{hue}'. Skipping.")
            performance_text = "Insufficient samples"
        else:
            if hue_type == 'categorical' or hue_type == 'numerical':
                # Convert metadata to categorical codes
                metadata_model_cat = metadata_model.astype('category')
                y = metadata_model_cat.cat.codes

                # Check if there are enough samples per class
                class_counts = np.bincount(y)
                if np.min(class_counts) < 2:
                    logging.warning(f"Not enough samples per class for '{hue}'. Skipping model training and evaluation.")
                    performance_text = "Insufficient samples per class"
                else:
                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        feature_matrix_valid, y, test_size=0.3, random_state=42, stratify=y)

                    # Train a classifier
                    clf = RandomForestClassifier(random_state=42)
                    clf.fit(X_train, y_train)

                    # Evaluate the classifier
                    y_pred = clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    logging.info(f"Classification accuracy for '{hue}': {accuracy:.4f}")

                    # Save feature importances if required
                    if all_kmers is not None and feature_importances_folder is not None:
                        importance = clf.feature_importances_
                        feature_importances = sorted(zip(all_kmers, importance), key=lambda x: x[1], reverse=True)
                        importance_file = os.path.join(feature_importances_folder, f"feature_importance_{data_col}_{hue}_{embedding_name.lower()}.txt")
                        with open(importance_file, 'w') as f:
                            for kmer, score in feature_importances:
                                f.write(f"{kmer}\t{score}\n")
                        logging.info(f"Feature importances saved to {importance_file}")

                    performance_text = f"Accuracy: {accuracy:.2f}"
            else:
                logging.error(f"Unknown hue type for '{hue}'.")
                performance_text = "Unknown hue type"

        # Plotting
        plot_df = pd.DataFrame({
            f'{embedding_name}1': emb[:, 0],
            f'{embedding_name}2': emb[:, 1],
            'Hue': plot_metadata
        })

        if hue_type_for_plot == 'categorical':
            # Create a custom color palette that includes a color for NA values
            unique_categories = plot_df['Hue'].unique()
            n_categories = len(unique_categories)
            # Adjust palette based on number of categories
            if n_categories > 20:
                custom_palette = sns.color_palette('husl', n_colors=n_categories)
            elif n_categories > 9:
                custom_palette = sns.color_palette('tab20', n_colors=n_categories)
            else:
                custom_palette = sns.color_palette('Set1', n_colors=n_categories)

            color_dict = dict(zip(unique_categories, custom_palette))
            if 'NA' in unique_categories:
                color_dict['NA'] = (0.7, 0.7, 0.7)  # Grey color for NA values

            sns.scatterplot(
                x=f'{embedding_name}1', y=f'{embedding_name}2',
                hue='Hue',
                data=plot_df,
                palette=color_dict,
                s=80,
                edgecolor='k',
                alpha=0.7,
                ax=ax
            )
            ax.set_title(f"{hue}\n{performance_text}")

            # Adjust legend inside the plot
            ax.legend(title=hue, loc='best', fontsize='small', frameon=True)

        elif hue_type_for_plot == 'numerical':
            # Plotting numerical hue
            scatter = ax.scatter(
                plot_df[f'{embedding_name}1'],
                plot_df[f'{embedding_name}2'],
                c=plot_df['Hue'],
                cmap='viridis',
                s=80,
                edgecolor='k',
                alpha=0.7
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(hue)
            ax.set_title(f"{hue}\n{performance_text}")
        else:
            ax.text(0.5, 0.5, "Unknown hue type", transform=ax.transAxes, ha='center')
            ax.set_title(f"{hue}\n{performance_text}")

        # Set consistent axis limits
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])

    # Remove any empty subplots
    for idx in range(len(hues), nrows*ncols):
        fig.delaxes(axes[idx])

    fig.suptitle(f"{embedding_name} plots for {data_col} {'(Excluding Other & NA)' if exclude_other_na else ''}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust to accommodate suptitle
    suffix = "_no_other_na" if exclude_other_na else ""
    output_filename = f"{embedding_name.lower()}_{data_col}{suffix}.pdf"
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, format='pdf', dpi=300)
    plt.close()
    logging.info(f"{embedding_name} plots{' (excluding Other & NA)' if exclude_other_na else ''} saved to {output_path}")


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    parser = argparse.ArgumentParser(description="Plot k-mer content using UMAP and t-SNE, and perform classification.")
    parser.add_argument("--summary_table", required=True, help="Path to the summary table CSV.")
    parser.add_argument("--metadata_table", required=True, help="Path to the metadata CSV file.")
    parser.add_argument("--taxa_table", required=True, help="Path to the taxa CSV file.")
    parser.add_argument("--output_folder", default="kmer_plots", help="Folder to save plots and results.")
    parser.add_argument("--kmer_sizes", type=int, nargs='+', default=[3, 5], help="Sizes of the k-mers to plot. Default is [3, 5].")
    parser.add_argument("--num_bins", type=int, default=5, help="Number of bins for quantile-based binning of numerical attributes. Default is 5.")
    parser.add_argument("--top_taxa", type=int, default=6, help="Number of top categories to display for taxonomic levels. Default is 6.")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0, help="Perplexity parameter for t-SNE. Default is 30.")
    parser.add_argument("--tsne_n_iter", type=int, default=1000, help="Number of iterations for t-SNE. Default is 1000.")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="Number of neighbors for UMAP. Default is 15.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="Minimum distance for UMAP. Default is 0.1.")
    parser.add_argument("--umap_n_epochs", type=int, default=None, help="Number of training epochs for UMAP. Default is None (auto).")
    args = parser.parse_args()

    logging.info("Starting the plot_kmer3.py script...")

    # Define categorical and numerical hues
    categorical_hues = [
        'motility', 'gram_staining', 'aerophilicity', 'extreme_environment_tolerance',
        'biofilm_formation', 'animal_pathogenicity', 'biosafety_level',
        'health_association', 'host_association', 'plant_pathogenicity',
        'spore_formation', 'hemolysis', 'Superkingdom', 'cell_shape', 'Phylum', 'Class', 'Order', 'Family', 'Genus'
    ]
    numerical_hues = [
        'num_crispr_arrays', 'average_crispr_length', 'average_DR_length', 'average_spacers'
    ]

    # Taxonomic levels to be limited
    top_taxa_hues = ['Phylum', 'Class', 'Order', 'Family', 'Genus']

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    logging.info(f"Output will be saved to: {args.output_folder}")

    # Create a subfolder for feature importances
    feature_importances_folder = os.path.join(args.output_folder, "feature_importances")
    os.makedirs(feature_importances_folder, exist_ok=True)
    logging.info(f"Feature importances will be saved to: {feature_importances_folder}")

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
        summary_df['leader_flank_kmer_dict'] = summary_df['leader_flank_kmer_freq'].apply(safe_json_loads)
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
        logging.info(f"Filtered out {initial_count - filtered_count} samples with zero CRISPR arrays. Remaining samples: {filtered_count}")
    else:
        logging.warning("Column 'num_crispr_arrays' not found. No filtering applied.")

    # Define the data columns to process
    data_columns = {
        'leader_flank_kmer': 'Leader Flank k-mer',
        'left_kmer': 'Left Flank k-mer',
        'right_kmer': 'Right Flank k-mer',
        'spacer_kmer': 'Spacer k-mer'
    }

    # Define t-SNE parameters
    tsne_params = {
        'perplexity': args.tsne_perplexity,
        'n_components': 2,
        'n_iter': args.tsne_n_iter,
        'random_state': 42
    }

    # Define UMAP parameters
    umap_params = {
        'n_neighbors': args.umap_n_neighbors,
        'min_dist': args.umap_min_dist,
        'n_components': 2,
        'random_state': 42
    }
    if args.umap_n_epochs is not None:
        umap_params['n_epochs'] = args.umap_n_epochs

    # Process each data column
    for data_col, description in data_columns.items():
        logging.info(f"\nProcessing '{data_col}': {description}")

        # Prepare feature matrix
        if data_col.endswith('_kmer'):
            # For k-mer data
            kmer_dicts = merged_df[f'{data_col}_dict'].tolist()
            all_kmers = set()
            for kmer_dict in kmer_dicts:
                flattened = flatten_kmer_dict(kmer_dict, args.kmer_sizes)
                all_kmers.update(flattened.keys())
            all_kmers = sorted(all_kmers)

            logging.info(f"Number of unique k-mers found: {len(all_kmers)}")
            logging.info(f"First few k-mers: {all_kmers[:5]}")

            if len(all_kmers) == 0:
                logging.error(f"No k-mers found for '{data_col}'. Skipping this column.")
                continue

            # Generate feature matrix
            feature_matrix = np.array([kmer_dict_to_vector(flatten_kmer_dict(kd, args.kmer_sizes), all_kmers) for kd in kmer_dicts])

            logging.info(f"Feature matrix shape: {feature_matrix.shape}")

        else:
            # For nucleotide distribution data
            nuc_dist_col = f'{data_col}_nuc_dist'
            if nuc_dist_col not in merged_df.columns:
                logging.error(f"Column '{nuc_dist_col}' not found in the merged DataFrame. Skipping this column.")
                continue

            # Extract nucleotide distribution and handle missing data
            try:
                nuc_dist_list = merged_df[nuc_dist_col].tolist()
                # Ensure all nucleotide distributions have the same keys
                all_nucs = set()
                for nuc_dist in nuc_dist_list:
                    all_nucs.update(nuc_dist.keys())
                all_nucs = sorted(all_nucs)

                if len(all_nucs) == 0:
                    logging.error(f"No nucleotide distributions found for '{data_col}'. Skipping this column.")
                    continue

                feature_matrix = np.array([[nuc_dist.get(nuc, 0) for nuc in all_nucs] for nuc_dist in nuc_dist_list])
                logging.info(f"Feature matrix shape: {feature_matrix.shape}")
            except Exception as e:
                logging.error(f"Error processing nucleotide distribution for '{data_col}': {e}")
                continue

        if feature_matrix.shape[1] == 0:
            logging.error(f"No features found for '{data_col}'. Skipping this column.")
            continue

        # Standardize features
        scaler = StandardScaler()
        try:
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)
            logging.info(f"Feature matrix standardized for '{data_col}'.")
        except ValueError as ve:
            logging.error(f"StandardScaler error for '{data_col}': {ve}. Skipping this column.")
            continue

        # Compute UMAP embedding
        try:
            reducer = umap.UMAP(**umap_params)
            umap_embedding = reducer.fit_transform(feature_matrix_scaled)
            logging.info(f"UMAP embedding computed for '{data_col}'.")
        except Exception as e:
            logging.error(f"UMAP computation error for '{data_col}': {e}. Skipping UMAP plots.")
            umap_embedding = None

        # Compute t-SNE embedding
        n_samples = feature_matrix_scaled.shape[0]
        if tsne_params['perplexity'] >= n_samples:
            logging.warning(f"The perplexity ({tsne_params['perplexity']}) is greater than or equal to the number of samples ({n_samples}). Skipping t-SNE plots for '{data_col}'.")
            tsne_embedding = None
        else:
            try:
                tsne = TSNE(**tsne_params)
                tsne_embedding = tsne.fit_transform(feature_matrix_scaled)
                logging.info(f"t-SNE embedding computed for '{data_col}'.")
            except Exception as e:
                logging.error(f"t-SNE computation error for '{data_col}': {e}. Skipping t-SNE plots.")
                tsne_embedding = None

        umap_axis_limits = None
        tsne_axis_limits = None

        # Combine categorical and numerical hues
        hues = categorical_hues + numerical_hues

        # Generate UMAP plots
        if umap_embedding is not None:
            # First, generate the standard UMAP plots
            generate_embedding_plots(
                embedding=umap_embedding,
                feature_matrix_scaled=feature_matrix_scaled,
                merged_df=merged_df,
                hues=hues,
                data_col=data_col,
                embedding_name='UMAP',
                output_folder=args.output_folder,
                all_kmers=all_kmers if data_col.endswith('_kmer') else None,
                categorical_hues=categorical_hues,
                numerical_hues=numerical_hues,
                num_bins=args.num_bins,
                feature_importances_folder=feature_importances_folder,
                top_taxa=args.top_taxa,
                top_taxa_hues=top_taxa_hues
            )

            # Capture axis limits from the first plot
            if umap_axis_limits is None:
                x_min, x_max = np.min(umap_embedding[:, 0]), np.max(umap_embedding[:, 0])
                y_min, y_max = np.min(umap_embedding[:, 1]), np.max(umap_embedding[:, 1])
                umap_axis_limits = (x_min, x_max, y_min, y_max)

            # Then, generate the UMAP plots excluding 'Other' and 'NA'
            generate_embedding_plots(
                embedding=umap_embedding,
                feature_matrix_scaled=feature_matrix_scaled,
                merged_df=merged_df,
                hues=hues,
                data_col=data_col,
                embedding_name='UMAP',
                output_folder=args.output_folder,
                all_kmers=all_kmers if data_col.endswith('_kmer') else None,
                categorical_hues=categorical_hues,
                numerical_hues=numerical_hues,
                num_bins=args.num_bins,
                feature_importances_folder=feature_importances_folder,
                top_taxa=args.top_taxa,
                top_taxa_hues=top_taxa_hues,
                exclude_other_na=True,
                axis_limits=umap_axis_limits
            )

        # Generate t-SNE plots if embedding is available
        if tsne_embedding is not None:
            # First, generate the standard t-SNE plots
            generate_embedding_plots(
                embedding=tsne_embedding,
                feature_matrix_scaled=feature_matrix_scaled,
                merged_df=merged_df,
                hues=hues,
                data_col=data_col,
                embedding_name='t-SNE',
                output_folder=args.output_folder,
                all_kmers=all_kmers if data_col.endswith('_kmer') else None,
                categorical_hues=categorical_hues,
                numerical_hues=numerical_hues,
                num_bins=args.num_bins,
                feature_importances_folder=feature_importances_folder,
                top_taxa=args.top_taxa,
                top_taxa_hues=top_taxa_hues
            )

            # Capture axis limits from the first plot
            if tsne_axis_limits is None:
                x_min, x_max = np.min(tsne_embedding[:, 0]), np.max(tsne_embedding[:, 0])
                y_min, y_max = np.min(tsne_embedding[:, 1]), np.max(tsne_embedding[:, 1])
                tsne_axis_limits = (x_min, x_max, y_min, y_max)

            # Then, generate the t-SNE plots excluding 'Other' and 'NA'
            generate_embedding_plots(
                embedding=tsne_embedding,
                feature_matrix_scaled=feature_matrix_scaled,
                merged_df=merged_df,
                hues=hues,
                data_col=data_col,
                embedding_name='t-SNE',
                output_folder=args.output_folder,
                all_kmers=all_kmers if data_col.endswith('_kmer') else None,
                categorical_hues=categorical_hues,
                numerical_hues=numerical_hues,
                num_bins=args.num_bins,
                feature_importances_folder=feature_importances_folder,
                top_taxa=args.top_taxa,
                top_taxa_hues=top_taxa_hues,
                exclude_other_na=True,
                axis_limits=tsne_axis_limits
            )

    logging.info("Script execution completed.")


if __name__ == "__main__":
    main()
