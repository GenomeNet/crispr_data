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

def plot_feature_importances(feature_importances, data_col, hue, output_plot_path):
    """
    Plot and save the feature importances, highlighting only the Top 5 and Bottom 5 features.

    Args:
        feature_importances (list of tuples): List of (kmer, importance) sorted by importance descending.
        data_col (str): The name of the current data column (e.g., 'left_kmer').
        hue (str): The name of the metadata column.
        output_plot_path (str): Path to save the feature importance plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import logging

    # Convert to DataFrame for easier handling
    fi_df = pd.DataFrame(feature_importances, columns=['kmer', 'importance'])

    # Sort descendingly by importance
    fi_df.sort_values('importance', ascending=False, inplace=True)

    # Define number of features per category
    top_n = 5
    bottom_n = 5

    total_features = fi_df.shape[0]

    # Adjust counts if total_features < 10
    if total_features < (top_n + bottom_n):
        logging.warning(
            f"Not enough features ({total_features}) to plot Top {top_n} and Bottom {bottom_n} features. "
            f"Adjusting the numbers accordingly."
        )
        top_n = min(top_n, total_features // 2)
        bottom_n = total_features - top_n
        if bottom_n > top_n:
            top_n = min(top_n, total_features - bottom_n)

    # Initialize 'Category' column to 'Other'
    fi_df['Category'] = 'Other'  # Features not in Top or Bottom

    # Assign 'Top 5'
    fi_df.iloc[:top_n, fi_df.columns.get_loc('Category')] = 'Top 5'

    # Assign 'Bottom 5'
    fi_df.iloc[-bottom_n:, fi_df.columns.get_loc('Category')] = 'Bottom 5'

    # Select only the top and bottom features
    selected_fi_df = fi_df[fi_df['Category'].isin(['Top 5', 'Bottom 5'])]

    # Sort ascendingly for horizontal bar plot
    selected_fi_df.sort_values('importance', inplace=True)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x='importance',
        y='kmer',
        hue='Category',
        data=selected_fi_df,
        dodge=False,
        palette={'Top 5': 'blue', 'Bottom 5': 'red'}
    )

    plt.xlabel('Feature Importance')
    plt.ylabel('K-mer')
    plt.title(f"Feature Importances for {data_col} - {hue}")
    plt.legend(title='Category')

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    logging.info(f"Feature importances plot saved to {output_plot_path}")


def generate_umap_plot(feature_matrix, metadata, hue, output_path, data_col, hue_type='categorical', log_scale=False, palette=None, all_kmers=None, output_folder=None, umap_params={}):
    """
    Perform UMAP dimensionality reduction, train a model to predict the metadata,
    generate a scatter plot annotated with model performance metric,
    and save feature importances to a text file and plot them.

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
        umap_params (dict): Parameters to pass to the UMAP algorithm.
    """
    logging.info(f"Generating UMAP plot for hue '{hue}' with type '{hue_type}' and log_scale={log_scale}.")

    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # Apply UMAP with custom parameters
    reducer = umap.UMAP(random_state=42, **umap_params)
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
        if not isinstance(metadata_model.dtype, pd.CategoricalDtype):
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

            # **Plot feature importances**
            feature_importance_plot = os.path.join(output_folder, f"feature_importance_plot_{data_col}_{hue}.png")
            plot_feature_importances(feature_importances, data_col, hue, feature_importance_plot)

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

            # **Plot feature importances**
            feature_importance_plot = os.path.join(output_folder, f"feature_importance_plot_{data_col}_{hue}.png")
            plot_feature_importances(feature_importances, data_col, hue, feature_importance_plot)

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
    logging.info(f"UMAP plot saved to {output_path}")

def generate_tsne_plot(feature_matrix, metadata, hue, output_path, data_col, hue_type='categorical', log_scale=False, palette=None, all_kmers=None, output_folder=None, tsne_params={}):
    """
    Perform t-SNE dimensionality reduction, train a model to predict the metadata,
    generate a scatter plot annotated with model performance metric,
    and save feature importances to a text file and plot them.

    Args:
        feature_matrix (np.ndarray): The feature matrix for t-SNE and model training.
        metadata (pd.Series): Metadata column for coloring and prediction.
        hue (str): The name of the metadata column.
        output_path (str): Path to save the plot.
        data_col (str): The name of the current data column (e.g., 'left_kmer').
        hue_type (str): Type of hue ('categorical' or 'numerical').
        log_scale (bool): Whether to apply log scaling to numerical hues.
        palette (list or seaborn palette, optional): Color palette for categorical hues.
        all_kmers (list, optional): List of all k-mers corresponding to the feature matrix columns.
        output_folder (str, optional): Folder to save the feature importance files.
        tsne_params (dict): Parameters to pass to the TSNE algorithm.
    """
    logging.info(f"Generating t-SNE plot for hue '{hue}' with type '{hue_type}' and log_scale={log_scale}.")

    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # Apply t-SNE
    tsne = TSNE(random_state=42, **tsne_params)
    embedding = tsne.fit_transform(feature_matrix_scaled)
    logging.info(f"t-SNE embedding shape for '{hue}': {embedding.shape}")

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
        if not isinstance(metadata_model.dtype, pd.CategoricalDtype):
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
            importance_file = os.path.join(output_folder, f"feature_importance_{data_col}_{hue}_tsne.txt")
            with open(importance_file, 'w') as f:
                for kmer, score in feature_importances:
                    f.write(f"{kmer}\t{score}\n")
            logging.info(f"Feature importances saved to {importance_file}")

            # **Plot feature importances**
            feature_importance_plot = os.path.join(output_folder, f"feature_importance_plot_{data_col}_{hue}_tsne.png")
            plot_feature_importances(feature_importances, data_col, hue, feature_importance_plot)

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
            importance_file = os.path.join(output_folder, f"feature_importance_{data_col}_{hue}_tsne.txt")
            with open(importance_file, 'w') as f:
                for kmer, score in feature_importances:
                    f.write(f"{kmer}\t{score}\n")
            logging.info(f"Feature importances saved to {importance_file}")

            # **Plot feature importances**
            feature_importance_plot = os.path.join(output_folder, f"feature_importance_plot_{data_col}_{hue}_tsne.png")
            plot_feature_importances(feature_importances, data_col, hue, feature_importance_plot)

        # Prepare text annotation
        performance_text = f"R²: {r2:.2f}"

    else:
        logging.error("hue_type must be 'categorical' or 'numerical'.")
        raise ValueError("hue_type must be 'categorical' or 'numerical'.")

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Dim1': embedding[:, 0],
        'Dim2': embedding[:, 1],
        'Hue': metadata_model.values
    })

    plt.figure(figsize=(12, 10))

    if hue_type == 'categorical':
        plt.title(f"t-SNE Colored by {hue}\n{performance_text}")

        sns.scatterplot(
            x='Dim1', y='Dim2',
            hue='Hue',
            data=plot_df,
            palette=palette,
            s=80,
            edgecolor='k',
            alpha=0.7
        )

        plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')

    elif hue_type == 'numerical':
        plt.title(f"t-SNE Colored by {'Log-scaled ' if log_scale else ''}{hue}\n{performance_text}")

        scatter = plt.scatter(
            plot_df['Dim1'],
            plot_df['Dim2'],
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
    logging.info(f"t-SNE plot saved to {output_path}")

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

        # Define categorical and numerical hues
        categorical_hues = ['Motility', 'Gram staining', 'Aerophilicity', 'Extreme environment tolerance',
                            'Biofilm formation', 'Animal pathogenicity', 'Biosafety level',
                            'Health association', 'Host association', 'Plant pathogenicity',
                            'Spore formation', 'Hemolysis', 'Cell shape', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
        numerical_hues = ['num_crispr_arrays', 'average_crispr_length', 'average_DR_length', 'average_spacers']

        # Generate UMAP plots
        for hue in categorical_hues:
            logging.info(f"  - Generating UMAP plot colored by categorical hue '{hue}'.")
            metadata = merged_df[hue]

            # Check for missing values in metadata
            if metadata.isnull().all():
                logging.warning(f"    Warning: All values in '{hue}' are missing. Skipping.")
                continue

            # Get top N categories
            top_n = args.top_taxa_count[categorical_hues.index(hue)] if args.top_taxa_count else args.top_taxa
            value_counts = metadata.value_counts()
            top_categories = value_counts.nlargest(top_n).index.tolist()

            # Create a new series with only top categories, others as 'Other'
            metadata_top = metadata.apply(lambda x: x if x in top_categories else 'Other')

            output_filename = f"umap_{data_col}_{hue}.png"
            output_path = os.path.join(args.output_folder, output_filename)

            generate_umap_plot(
                feature_matrix=feature_matrix,
                metadata=metadata_top,
                hue=hue,
                output_path=output_path,
                data_col=data_col,
                hue_type='categorical',
                palette='Set1',
                all_kmers=all_kmers,
                output_folder=args.output_folder,
                umap_params=umap_params
            )

        for hue in numerical_hues:
            logging.info(f"  - Generating UMAP plot colored by numerical hue '{hue}'.")
            metadata = merged_df[hue]

            # Check for missing values in metadata
            if metadata.isnull().all():
                logging.warning(f"    Warning: All values in '{hue}' are missing. Skipping.")
                continue

            output_filename = f"umap_{data_col}_{hue}.png"
            output_path = os.path.join(args.output_folder, output_filename)

            generate_umap_plot(
                feature_matrix=feature_matrix,
                metadata=metadata,
                hue=hue,
                output_path=output_path,
                data_col=data_col,
                hue_type='numerical',
                log_scale=True,
                all_kmers=all_kmers,
                output_folder=args.output_folder,
                umap_params=umap_params
            )

            # Generate binned plot
            binned_metadata = bin_numerical_attribute(merged_df, hue, bins=args.num_bins)
            output_filename = f"umap_{data_col}_{hue}_binned.png"
            output_path = os.path.join(args.output_folder, output_filename)

            generate_umap_plot(
                feature_matrix=feature_matrix,
                metadata=binned_metadata,
                hue=f"{hue} (Binned)",
                output_path=output_path,
                data_col=data_col,
                hue_type='categorical',
                palette='viridis',
                all_kmers=all_kmers,
                output_folder=args.output_folder,
                umap_params=umap_params
            )

        # Generate t-SNE plots
        for hue in categorical_hues:
            logging.info(f"  - Generating t-SNE plot colored by categorical hue '{hue}'.")
            metadata = merged_df[hue]

            # Check for missing values in metadata
            if metadata.isnull().all():
                logging.warning(f"    Warning: All values in '{hue}' are missing. Skipping.")
                continue

            # Get top N categories
            top_n = args.top_taxa_count[categorical_hues.index(hue)] if args.top_taxa_count else args.top_taxa
            value_counts = metadata.value_counts()
            top_categories = value_counts.nlargest(top_n).index.tolist()

            # Create a new series with only top categories, others as 'Other'
            metadata_top = metadata.apply(lambda x: x if x in top_categories else 'Other')

            output_filename = f"tsne_{data_col}_{hue}.png"
            output_path = os.path.join(args.output_folder, output_filename)

            generate_tsne_plot(
                feature_matrix=feature_matrix,
                metadata=metadata_top,
                hue=hue,
                output_path=output_path,
                data_col=data_col,
                hue_type='categorical',
                palette='Set1',
                all_kmers=all_kmers,
                output_folder=args.output_folder,
                tsne_params=tsne_params
            )

        for hue in numerical_hues:
            logging.info(f"  - Generating t-SNE plot colored by numerical hue '{hue}'.")
            metadata = merged_df[hue]

            # Check for missing values in metadata
            if metadata.isnull().all():
                logging.warning(f"    Warning: All values in '{hue}' are missing. Skipping.")
                continue

            output_filename = f"tsne_{data_col}_{hue}.png"
            output_path = os.path.join(args.output_folder, output_filename)

            generate_tsne_plot(
                feature_matrix=feature_matrix,
                metadata=metadata,
                hue=hue,
                output_path=output_path,
                data_col=data_col,
                hue_type='numerical',
                log_scale=True,
                all_kmers=all_kmers,
                output_folder=args.output_folder,
                tsne_params=tsne_params
            )

            # Generate binned plot
            binned_metadata = bin_numerical_attribute(merged_df, hue, bins=args.num_bins)
            output_filename = f"tsne_{data_col}_{hue}_binned.png"
            output_path = os.path.join(args.output_folder, output_filename)

            generate_tsne_plot(
                feature_matrix=feature_matrix,
                metadata=binned_metadata,
                hue=f"{hue} (Binned)",
                output_path=output_path,
                data_col=data_col,
                hue_type='categorical',
                palette='viridis',
                all_kmers=all_kmers,
                output_folder=args.output_folder,
                tsne_params=tsne_params
            )

    logging.info("Script execution completed.")

if __name__ == "__main__":
    main()