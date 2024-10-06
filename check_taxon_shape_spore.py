import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Analyze the relationship between num_crispr_arrays, cell_shape, spore_formation, and taxonomic classifications.")
    parser.add_argument("--summary_table", required=True, help="Path to the summary table CSV containing num_crispr_arrays.")
    parser.add_argument("--taxa_table", required=True, help="Path to the taxa CSV file containing all taxonomic ranks.")
    parser.add_argument("--metadata_table", required=True, help="Path to the metadata CSV file containing cell shape and spore formation information.")
    parser.add_argument("--output_folder", default="taxon_shape_spore_output", help="Folder to save the generated plots.")
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Read the summary table
    try:
        summary_df = pd.read_csv(args.summary_table, delimiter=';')  # Adjust delimiter if necessary
    except Exception as e:
        print(f"Error reading summary table: {e}")
        sys.exit(1)

    # Check if 'sample' and 'num_crispr_arrays' columns exist
    required_columns = ['sample', 'num_crispr_arrays']
    for col in required_columns:
        if col not in summary_df.columns:
            print(f"Error: '{col}' column not found in summary table.")
            sys.exit(1)

    # Read the taxa table with all taxonomic ranks
    try:
        taxa_df = pd.read_csv(args.taxa_table)
    except Exception as e:
        print(f"Error reading taxa table: {e}")
        sys.exit(1)

    # Check if 'sample' column exists in taxa_df
    if 'sample' not in taxa_df.columns:
        print("Error: 'sample' column not found in taxa table.")
        sys.exit(1)

    # Read the metadata table
    try:
        metadata_df = pd.read_csv(args.metadata_table)
    except Exception as e:
        print(f"Error reading metadata table: {e}")
        sys.exit(1)

    # Check if 'sample', 'cell_shape', and 'spore_formation' columns exist in metadata_df
    metadata_required_columns = ['sample', 'cell_shape', 'spore_formation']
    for col in metadata_required_columns:
        if col not in metadata_df.columns:
            print(f"Error: '{col}' column not found in metadata table.")
            sys.exit(1)

    # Clean 'sample' columns by stripping whitespace and standardizing file extensions
    summary_df['sample'] = summary_df['sample'].str.strip()
    taxa_df['sample'] = taxa_df['sample'].str.strip()
    metadata_df['sample'] = metadata_df['sample'].str.strip()

    # Merge the dataframes
    merged_df = summary_df.merge(metadata_df[['sample', 'cell_shape', 'spore_formation']], on='sample', how='left')
    merged_df = merged_df.merge(taxa_df, on='sample', how='left')

    # Handle missing values in 'cell_shape' and 'spore_formation'
    merged_df['cell_shape'] = merged_df['cell_shape'].fillna('Unknown')
    merged_df['spore_formation'] = merged_df['spore_formation'].fillna(False).astype(bool)

    # Ensure all taxonomic ranks are available
    taxonomic_ranks = ['Superkingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    for rank in taxonomic_ranks:
        if rank not in merged_df.columns:
            print(f"Warning: '{rank}' column not found in taxa table.")
            merged_df[rank] = 'Unknown'

    # Overview Statistics
    print("=== Overview Statistics ===")
    print(f"Total samples after merging: {len(merged_df)}")
    print(merged_df[['cell_shape', 'spore_formation', 'num_crispr_arrays']].describe(include='all'))
    print("\nNumber of samples with spore_formation=True:", merged_df['spore_formation'].sum())
    print("Number of samples with each cell_shape:")
    print(merged_df['cell_shape'].value_counts())

    # Plot 1: Boxplot of num_crispr_arrays by cell_shape
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='cell_shape', y='num_crispr_arrays', data=merged_df)
    plt.title('Number of CRISPR Arrays by Cell Shape')
    plt.xlabel('Cell Shape')
    plt.ylabel('Number of CRISPR Arrays')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'num_crispr_arrays_by_cell_shape.png'))
    plt.close()
    print(f"Boxplot saved to: {os.path.join(args.output_folder, 'num_crispr_arrays_by_cell_shape.png')}")

    # Plot 2: Boxplot of num_crispr_arrays by spore_formation
    plt.figure(figsize=(8,6))
    sns.boxplot(x='spore_formation', y='num_crispr_arrays', data=merged_df)
    plt.title('Number of CRISPR Arrays by Spore Formation')
    plt.xlabel('Spore Formation')
    plt.ylabel('Number of CRISPR Arrays')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'num_crispr_arrays_by_spore_formation.png'))
    plt.close()
    print(f"Boxplot saved to: {os.path.join(args.output_folder, 'num_crispr_arrays_by_spore_formation.png')}")

    # Plot 3: Boxplot of num_crispr_arrays by cell_shape and spore_formation
    plt.figure(figsize=(12,8))
    sns.boxplot(x='cell_shape', y='num_crispr_arrays', hue='spore_formation', data=merged_df)
    plt.title('Number of CRISPR Arrays by Cell Shape and Spore Formation')
    plt.xlabel('Cell Shape')
    plt.ylabel('Number of CRISPR Arrays')
    plt.xticks(rotation=45)
    plt.legend(title='Spore Formation')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'num_crispr_arrays_by_cell_shape_spore.png'))
    plt.close()
    print(f"Grouped Boxplot saved to: {os.path.join(args.output_folder, 'num_crispr_arrays_by_cell_shape_spore.png')}")

    # Plot 4: Scatter plot of num_crispr_arrays vs cell_shape colored by Taxonomic Rank
    for rank in taxonomic_ranks:
        plt.figure(figsize=(14,10))
        sns.stripplot(x='cell_shape', y='num_crispr_arrays', hue=rank, data=merged_df, jitter=True, dodge=True, alpha=0.7)
        plt.title(f'Number of CRISPR Arrays by Cell Shape Colored by {rank}')
        plt.xlabel('Cell Shape')
        plt.ylabel('Number of CRISPR Arrays')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=rank)
        plt.xticks(rotation=45)
        plt.tight_layout()
        filename = f'crispr_arrays_cell_shape_{rank.lower()}.png'
        plt.savefig(os.path.join(args.output_folder, filename))
        plt.close()
        print(f"Scatter plot saved to: {os.path.join(args.output_folder, filename)}")

    # Plot 5: Heatmap of Average num_crispr_arrays across Taxonomic Ranks
    avg_crispr_by_taxa = merged_df.groupby(taxonomic_ranks)['num_crispr_arrays'].mean().reset_index()
    pivot_table = avg_crispr_by_taxa.pivot_table(values='num_crispr_arrays', index='Phylum', columns='Class', aggfunc='mean')
    plt.figure(figsize=(12,10))
    sns.heatmap(pivot_table, cmap='viridis')
    plt.title('Average Number of CRISPR Arrays across Phylum and Class')
    plt.xlabel('Class')
    plt.ylabel('Phylum')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'avg_crispr_arrays_heatmap.png'))
    plt.close()
    print(f"Heatmap saved to: {os.path.join(args.output_folder, 'avg_crispr_arrays_heatmap.png')}")

    # Plot 6: Pair Plot of Numerical Variables Colored by Cell Shape
    numerical_vars = ['num_crispr_arrays']
    pairplot_df = merged_df[numerical_vars + ['cell_shape']]
    sns.pairplot(pairplot_df, hue='cell_shape', diag_kind='kde')
    plt.savefig(os.path.join(args.output_folder, 'pairplot_numerical_vars.png'))
    plt.close()
    print(f"Pairplot saved to: {os.path.join(args.output_folder, 'pairplot_numerical_vars.png')}")

    # Correlation Matrix including Taxonomic Ranks
    # Encode categorical variables
    categorical_vars = ['cell_shape', 'spore_formation'] + taxonomic_ranks
    encoded_df = pd.get_dummies(merged_df[categorical_vars], drop_first=True)
    correlation_df = pd.concat([merged_df[['num_crispr_arrays']], encoded_df], axis=1)
    correlation = correlation_df.corr()

    plt.figure(figsize=(20,18))
    sns.heatmap(correlation, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix Including Taxonomic Ranks')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'correlation_matrix_extended.png'))
    plt.close()
    print(f"Extended correlation matrix heatmap saved to: {os.path.join(args.output_folder, 'correlation_matrix_extended.png')}")

    # Summary Report
    with open(os.path.join(args.output_folder, 'summary_report.txt'), 'w') as report:
        report.write("=== Summary Report ===\n")
        report.write(f"Total samples after merging: {len(merged_df)}\n\n")
        report.write("Basic Statistics:\n")
        report.write(str(merged_df[['cell_shape', 'spore_formation', 'num_crispr_arrays']].describe(include='all')))
        report.write("\n\nNumber of Samples per Cell Shape:\n")
        report.write(str(merged_df['cell_shape'].value_counts()))
        report.write("\n\nNumber of Samples per Spore Formation Status:\n")
        report.write(str(merged_df['spore_formation'].value_counts()))
        report.write("\n\nNumber of Samples per Taxonomic Rank:\n")
        for rank in taxonomic_ranks:
            report.write(f"\n{rank}:\n")
            report.write(str(merged_df[rank].value_counts()))
        report.write("\n\nCorrelation Matrix:\n")
        report.write(str(correlation))
    print(f"Summary report saved to: {os.path.join(args.output_folder, 'summary_report.txt')}")

if __name__ == "__main__":
    main()