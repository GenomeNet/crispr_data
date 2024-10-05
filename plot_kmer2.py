# plot_kmer2_updated.py
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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text

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

def generate_embedding_plots(embedding, feature_matrix_scaled, merged_df, hues, data_col, embedding_name, output_folder, all_kmers=None, categorical_hues=[], numerical_hues=[], model_params={}, num_bins=5):
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
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    from collections import Counter
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    import logging

    # Create output path
    output_filename = f"{embedding_name.lower()}_{data_col}.png"
    output_path = os.path.join(output_folder, output_filename)

    num_hues = len(hues)
    nrows = int(np.ceil(num_hues / 4))  # Let's have up to 4 plots per row
    ncols = min(num_hues, 4)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten in case of only one row

    for idx, hue in enumerate(hues):
        ax = axes[idx]

        metadata = merged_df[hue]
        # Replace missing values with 'NA' for plotting
        plot_metadata = metadata.fillna('NA')
        hue_type = 'categorical' if hue in categorical_hues else ('numerical' if hue in numerical_hues else 'unknown')

        # Handle hue types
        # For categorical hues, we might limit the categories to top N
        # For numerical hues, we might log-scale or bin
        if hue_type == 'numerical':
            # Optionally bin the numerical attribute
            binned_metadata = bin_numerical_attribute(merged_df, hue, bins=num_bins)
            plot_metadata = binned_metadata.fillna('NA')
            hue_type_for_plot = 'categorical'  # For plotting purposes
        else:
            hue_type_for_plot = hue_type

        # For model training, we need to handle missing values
        valid_indices = ~metadata.isna()
        # Prepare metadata for model training
        metadata_model = metadata[valid_indices]
        feature_matrix_valid = feature_matrix_scaled[valid_indices]
        embedding_valid = embedding[valid_indices]

        # Proceed with model training and performance metrics
        if valid_indices.sum() < 2:
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
                    if all_kmers is not None and output_folder is not None:
                        importance = clf.feature_importances_
                        feature_importances = sorted(zip(all_kmers, importance), key=lambda x: x[1], reverse=True)
                        importance_file = os.path.join(output_folder, f"feature_importance_{data_col}_{hue}_{embedding_name.lower()}.txt")
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
            f'{embedding_name}1': embedding[:, 0],
            f'{embedding_name}2': embedding[:, 1],
            'Hue': plot_metadata
        })

        if hue_type_for_plot == 'categorical':
            # Create a custom color palette that includes a color for NA values
            unique_categories = plot_df['Hue'].unique()
            n_categories = len(unique_categories)
            custom_palette = sns.color_palette('Set1', n_colors=n_categories)
            color_dict = dict(zip(unique_categories, custom_palette))
            color_dict['NA'] = (0.7, 0.7, 0.7)  # Grey color for NA values

            sns.scatterplot(
                x=f'{embedding_name}1', y=f'{embedding_name}2',
                hue='Hue',
                data=plot_df,
                palette=color_dict,
                s=80,
                edgecolor='k',
                alpha=0.7,
                ax=ax,
                legend=False
            )
            ax.set_title(f"{hue}\n{performance_text}")

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

    # Remove any empty subplots
    for idx in range(len(hues), nrows*ncols):
        fig.delaxes(axes[idx])

    fig.suptitle(f"{embedding_name} plots for {data_col}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust top to accommodate suptitle
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"{embedding_name} plots saved to {output_path}")

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
    parser.add_argument("--summary_table", required=True, help="Path to the summary table CSV (e.g., summary_table3.csv).")
    parser.add_argument("--metadata_table", required=True, help="Path to the metadata CSV file (e.g., limited.csv).")
    parser.add_argument("--taxa_table", required=True, help="Path to the taxa CSV file (e.g., taxa.csv).")
    parser.add_argument("--output_folder", default="kmer_plots2", help="Folder to save plots and results.")
    parser.add_argument("--kmer_sizes", type=int, nargs='+', default=[3, 5], help="Sizes of the k-mers to plot. Default is [3, 5].")
    parser.add_argument("--num_bins", type=int, default=5, help="Number of bins for quantile-based binning of numerical attributes. Default is 5.")
    parser.add_argument("--top_taxa", type=int, default=6, help="Number of top categories to display for each taxonomic level. Default is 6.")
    parser.add_argument("--top_taxa_count", type=int, nargs='*', default=[], help="Number of top categories for each taxonomic level in the order: Phylum, Class, Order, Family, Genus. If not specified, --top_taxa is used for all.")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0, help="Perplexity parameter for t-SNE. Default is 30.")
    parser.add_argument("--tsne_n_iter", type=int, default=1000, help="Number of iterations for t-SNE. Default is 1000.")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="Number of neighbors for UMAP. Default is 15.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="Minimum distance for UMAP. Default is 0.1.")
    parser.add_argument("--umap_n_epochs", type=int, default=None, help="Number of training epochs for UMAP. Default is None (auto).")
    args = parser.parse_args()

    logging.info("Starting the plot_kmer.py script...")

    # Define categorical and numerical hues
    categorical_hues = ['motility', 'gram_staining', 'aerophilicity', 'extreme_environment_tolerance',
                            'biofilm_formation', 'animal_pathogenicity', 'biosafety_level',
                            'health_association', 'host_association', 'plant_pathogenicity',
                            'spore_formation', 'hemolysis', 'Superkingdom', 'cell_shape', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    numerical_hues = ['num_crispr_arrays', 'average_crispr_length', 'average_DR_length', 'average_spacers']

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
        'n_iter': args.tsne_n_iter
    }

    # Define UMAP parameters
    umap_params = {
        'n_neighbors': args.umap_n_neighbors,
        'min_dist': args.umap_min_dist,
        'n_components': 2
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
                all_kmers.update(flatten_kmer_dict(kmer_dict, args.kmer_sizes).keys())
            all_kmers = sorted(all_kmers)

            feature_matrix = np.array([kmer_dict_to_vector(flatten_kmer_dict(kd, args.kmer_sizes), all_kmers) for kd in kmer_dicts])
        else:
            # For nucleotide distribution data
            feature_matrix = np.array([list(d.values()) for d in merged_df[f'{data_col}_nuc_dist']])

        logging.info(f"Feature matrix shape: {feature_matrix.shape}")

        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        # Compute UMAP embedding
        reducer = umap.UMAP(random_state=42, **umap_params)
        umap_embedding = reducer.fit_transform(feature_matrix_scaled)
        logging.info(f"UMAP embedding computed for '{data_col}'.")

        # Compute t-SNE embedding
        n_samples = feature_matrix_scaled.shape[0]
        if tsne_params['perplexity'] >= n_samples:
            logging.warning(f"The perplexity ({tsne_params['perplexity']}) is greater than or equal to the number of samples ({n_samples}). Skipping t-SNE plots for '{data_col}'.")
            tsne_embedding = None
        else:
            tsne = TSNE(random_state=42, **tsne_params)
            tsne_embedding = tsne.fit_transform(feature_matrix_scaled)
            logging.info(f"t-SNE embedding computed for '{data_col}'.")

        # Combine categorical and numerical hues
        hues = categorical_hues + numerical_hues

        # Generate UMAP plots
        generate_embedding_plots(
            embedding=umap_embedding,
            feature_matrix_scaled=feature_matrix_scaled,
            merged_df=merged_df,
            hues=hues,
            data_col=data_col,
            embedding_name='UMAP',
            output_folder=args.output_folder,
            all_kmers=all_kmers,
            categorical_hues=categorical_hues,
            numerical_hues=numerical_hues,
            num_bins=args.num_bins
        )

        # Generate t-SNE plots if embedding is available
        if tsne_embedding is not None:
            generate_embedding_plots(
                embedding=tsne_embedding,
                feature_matrix_scaled=feature_matrix_scaled,
                merged_df=merged_df,
                hues=hues,
                data_col=data_col,
                embedding_name='t-SNE',
                output_folder=args.output_folder,
                all_kmers=all_kmers,
                categorical_hues=categorical_hues,
                numerical_hues=numerical_hues,
                num_bins=args.num_bins
            )

    logging.info("Script execution completed.")

if __name__ == "__main__":
    main()