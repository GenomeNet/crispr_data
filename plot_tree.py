import pandas as pd
import numpy as np
import argparse
import os
import json
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, r2_score
import logging

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

def main():
    parser = argparse.ArgumentParser(description="Generate decision trees based on k-mer content.")
    parser.add_argument('--summary_table', required=True, help='Path to the summary table CSV file.')
    parser.add_argument('--metadata_table', required=True, help='Path to the metadata CSV file.')
    parser.add_argument('--taxa_table', required=True, help='Path to the taxa CSV file.')
    parser.add_argument('--output_folder', required=True, help='Folder to save the output plots and files.')
    parser.add_argument('--kmer_sizes', type=int, nargs='+', default=[3], help='List of k-mer sizes to process. Default is 3.')
    parser.add_argument('--top_taxa', type=int, default=5, help='Number of top taxa categories to include. Default is 5.')
    parser.add_argument('--num_bins', type=int, default=5, help='Number of bins for numerical attributes. Default is 5.')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Starting the plot_tree.py script...")
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    logging.info(f"Output will be saved to: {args.output_folder}")

    # Read summary table
    try:
        summary_df = pd.read_csv(args.summary_table, sep=';')
        logging.info(f"Summary table loaded with {summary_df.shape[0]} samples.")
    except Exception as e:
        logging.error(f"Error reading summary table: {e}")
        sys.exit(1)

    # Read metadata and taxa tables
    try:
        metadata_df = pd.read_csv(args.metadata_table)
        logging.info(f"Metadata table loaded with {metadata_df.shape[0]} samples.")
    except Exception as e:
        logging.error(f"Error reading metadata table: {e}")
        sys.exit(1)

    try:
        taxa_df = pd.read_csv(args.taxa_table)
        logging.info(f"Taxa table loaded with {taxa_df.shape[0]} samples.")
    except Exception as e:
        logging.error(f"Error reading taxa table: {e}")
        sys.exit(1)

    # Parse the JSON strings in k-mer frequency and nucleotide distribution columns
    try:
        summary_df['left_kmer_dict'] = summary_df['leftflank_kmer_freq'].apply(safe_json_loads)
        summary_df['right_kmer_dict'] = summary_df['rightflank_kmer_freq'].apply(safe_json_loads)
        summary_df['spacer_kmer_dict'] = summary_df['spacer_kmer_frequencies'].apply(safe_json_loads)
        # Skipping 'spacer_nuc_dist' and 'dr_nuc_dist' as per user request
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
        merged_df = merged_df[merged_df['num_crispr_arrays'] > 0]
        filtered_count = merged_df.shape[0]
        logging.info(f"Filtered out {initial_count - filtered_count} samples with zero CRISPR arrays. Remaining samples: {filtered_count}")
    else:
        logging.warning("Column 'num_crispr_arrays' not found. No filtering applied.")

    # Define the data columns to process (only k-mer data)
    data_columns = {
        'left_kmer': 'Left Flank k-mer',
        'right_kmer': 'Right Flank k-mer',
        'spacer_kmer': 'Spacer k-mer'
    }

    # Loop through each data column and generate decision trees
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
            logging.warning(f"Data column '{data_col}' does not end with '_kmer'. Skipping.")
            continue

        logging.info(f"Feature matrix shape: {feature_matrix.shape}")

        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        # Define categorical and numerical hues
        categorical_hues = ['Phylum', 'Class', 'Order', 'Family', 'Genus']
        numerical_hues = ['num_crispr_arrays', 'average_crispr_length', 'average_DR_length', 'average_spacers']

        # Generate Decision Trees for Categorical Hues
        for hue in categorical_hues:
            logging.info(f"  - Generating Decision Tree for categorical hue '{hue}'.")
            metadata = merged_df[hue]

            # Check for missing values in metadata
            if metadata.isnull().all():
                logging.warning(f"    Warning: All values in '{hue}' are missing. Skipping.")
                continue

            # Get top N categories
            value_counts = metadata.value_counts()
            top_categories = value_counts.nlargest(args.top_taxa).index.tolist()

            # Create a new series with only top categories, others as 'Other'
            metadata_top = metadata.apply(lambda x: x if x in top_categories else 'Other')

            if metadata_top.nunique() <= 1:
                logging.warning(f"    Warning: Not enough categories in '{hue}' after filtering. Skipping.")
                continue

            # Encode categories
            metadata_model = metadata_top.astype('category')
            y = metadata_model.cat.codes

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                feature_matrix_scaled, y, test_size=0.3, random_state=42, stratify=y)

            # Train a Decision Tree Classifier
            dt_clf = DecisionTreeClassifier(random_state=42, max_depth=3)
            dt_clf.fit(X_train, y_train)

            # Evaluate the classifier
            y_pred = dt_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"    Decision Tree accuracy for '{hue}': {accuracy:.4f}")

            # Plot the Decision Tree
            tree_plot_path = os.path.join(args.output_folder, f"decision_tree_{data_col}_{hue}.png")
            plt.figure(figsize=(20, 10))
            plot_tree(dt_clf, feature_names=all_kmers if all_kmers else None, 
                      class_names=metadata_model.cat.categories.astype(str) if all_kmers else None, filled=True)
            plt.title(f"Decision Tree for {description} - {hue}\nAccuracy: {accuracy:.2f}")
            plt.savefig(tree_plot_path, dpi=300)
            plt.close()
            logging.info(f"    Decision tree plot saved to {tree_plot_path}")

        # Generate Decision Trees for Numerical Hues
        for hue in numerical_hues:
            logging.info(f"  - Generating Decision Tree for numerical hue '{hue}'.")
            metadata = merged_df[hue]

            # Check for missing values in metadata
            if metadata.isnull().all():
                logging.warning(f"    Warning: All values in '{hue}' are missing. Skipping.")
                continue

            y = metadata.values

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                feature_matrix_scaled, y, test_size=0.3, random_state=42)

            # Train a Decision Tree Regressor
            dt_reg = DecisionTreeRegressor(random_state=42, max_depth=3)
            dt_reg.fit(X_train, y_train)

            # Evaluate the regressor
            y_pred = dt_reg.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            logging.info(f"    Decision Tree R² score for '{hue}': {r2:.4f}")

            # Plot the Decision Tree
            tree_plot_path = os.path.join(args.output_folder, f"decision_tree_{data_col}_{hue}.png")
            plt.figure(figsize=(20, 10))
            plot_tree(dt_reg, feature_names=all_kmers if all_kmers else None, filled=True)
            plt.title(f"Decision Tree for {description} - {hue}\nR² Score: {r2:.2f}")
            plt.savefig(tree_plot_path, dpi=300)
            plt.close()
            logging.info(f"    Decision tree plot saved to {tree_plot_path}")

    logging.info("Script execution completed.")

if __name__ == "__main__":
        main()