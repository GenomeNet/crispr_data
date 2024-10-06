import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def main():
    parser = argparse.ArgumentParser(description="Analyze the relationship between num_crispr_arrays, cell_shape, and taxonomic classifications.")
    parser.add_argument("--summary_table", required=True, help="Path to the summary table CSV containing num_crispr_arrays.")
    parser.add_argument("--taxa_table", required=True, help="Path to the taxa CSV file containing all taxonomic ranks.")
    parser.add_argument("--metadata_table", required=True, help="Path to the metadata CSV file containing cell shape information.")
    parser.add_argument("--output_folder", default="taxon_cell_shape_output", help="Folder to save the generated plots.")
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

    # Check if 'sample' and 'cell_shape' columns exist in metadata_df
    metadata_required_columns = ['sample', 'cell_shape']
    for col in metadata_required_columns:
        if col not in metadata_df.columns:
            print(f"Error: '{col}' column not found in metadata table.")
            sys.exit(1)

    # Clean 'sample' columns by stripping whitespace and standardizing file extensions
    summary_df['sample'] = summary_df['sample'].str.strip()
    taxa_df['sample'] = taxa_df['sample'].str.strip()
    metadata_df['sample'] = metadata_df['sample'].str.strip()

    # Merge the dataframes
    merged_df = summary_df.merge(metadata_df[['sample', 'cell_shape']], on='sample', how='left')
    merged_df = merged_df.merge(taxa_df, on='sample', how='left')

    # Handle missing values in 'cell_shape'
    merged_df['cell_shape'] = merged_df['cell_shape'].fillna('Unknown')
    print("Filled missing 'cell_shape' values with 'Unknown'.")

    # Ensure all taxonomic ranks are available
    taxonomic_ranks = ['Superkingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    for rank in taxonomic_ranks:
        if rank not in merged_df.columns:
            print(f"Warning: '{rank}' column not found in taxa table.")
            merged_df[rank] = 'Unknown'

    ### Analysis focusing on cell shape across taxonomic ranks ###

    # For each taxonomic rank, plot the average num_crispr_arrays for different cell shapes
    for rank in taxonomic_ranks:
        # Identify taxa with multiple cell shapes
        shape_counts = merged_df.groupby(rank)['cell_shape'].nunique()
        valid_taxa = shape_counts[shape_counts > 1].index  # Taxa with multiple cell shapes

        # Filter the dataframe to include only valid taxa
        filtered_df = merged_df[merged_df[rank].isin(valid_taxa)]

        if filtered_df.empty:
            print(f"No taxa with multiple cell shapes found for rank: {rank}. Skipping plot.")
            continue

        # Calculate the overall average num_crispr_arrays per taxon
        avg_crispr = filtered_df.groupby(rank)['num_crispr_arrays'].mean().sort_values(ascending=False)

        # Define the order based on average number of CRISPR arrays
        order = avg_crispr.index

        plt.figure(figsize=(12,8))
        sns.barplot(
            x=rank,
            y='num_crispr_arrays',
            hue='cell_shape',
            data=filtered_df,
            ci=None,  # Remove error bars
            palette='Set1',
            order=order
        )
        plt.title(f'Average Number of CRISPR Arrays by Cell Shape across {rank}')
        plt.xlabel(rank)
        plt.ylabel('Average Number of CRISPR Arrays')
        plt.xticks(rotation=45)
        plt.legend(title='Cell Shape', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        filename = f'avg_crispr_arrays_cell_shape_{rank.lower()}.png'
        plt.savefig(os.path.join(args.output_folder, filename))
        plt.close()
        print(f"Bar plot saved to: {os.path.join(args.output_folder, filename)}")

    # Statistical analysis to test if the difference is significant at each taxonomic rank
    with open(os.path.join(args.output_folder, 'statistical_tests_cell_shape.txt'), 'w') as report:
        report.write("Statistical Tests for Difference in num_crispr_arrays between Cell Shapes\n\n")
        for rank in taxonomic_ranks:
            # Identify taxa with multiple cell shapes
            shape_counts = merged_df.groupby(rank)['cell_shape'].nunique()
            valid_taxa = shape_counts[shape_counts > 1].index  # Taxa with multiple cell shapes

            # Filter the dataframe to include only valid taxa
            filtered_df = merged_df[merged_df[rank].isin(valid_taxa)]

            if filtered_df.empty:
                continue

            report.write(f'=== {rank} Level ===\n')
            for taxa in filtered_df[rank].unique():
                sub_df = filtered_df[filtered_df[rank] == taxa]
                cell_shapes = sub_df['cell_shape'].unique()
                if len(cell_shapes) < 2:
                    continue
                for i in range(len(cell_shapes)):
                    for j in range(i+1, len(cell_shapes)):
                        shape1 = cell_shapes[i]
                        shape2 = cell_shapes[j]
                        group1 = sub_df[sub_df['cell_shape'] == shape1]['num_crispr_arrays']
                        group2 = sub_df[sub_df['cell_shape'] == shape2]['num_crispr_arrays']
                        if len(group1) > 0 and len(group2) > 0:
                            t_stat, p_value = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
                            report.write(f'{taxa} - {shape1} vs {shape2}:\n')
                            report.write(f'  {shape1} mean: {group1.mean():.2f} (n={len(group1)})\n')
                            report.write(f'  {shape2} mean: {group2.mean():.2f} (n={len(group2)})\n')
                            report.write(f'  t-statistic: {t_stat:.3f}, p-value: {p_value:.3e}\n\n')
                        else:
                            report.write(f'{taxa} - {shape1} vs {shape2}:\n')
                            report.write('  Insufficient data for statistical test (one of the groups is missing).\n\n')
            report.write('\n')
        print(f"Statistical test results saved to: {os.path.join(args.output_folder, 'statistical_tests_cell_shape.txt')}")

    # Overall boxplot of num_crispr_arrays by cell_shape
    plt.figure(figsize=(10,6))
    sns.boxplot(x='cell_shape', y='num_crispr_arrays', data=merged_df)
    plt.title('Distribution of Number of CRISPR Arrays by Cell Shape')
    plt.xlabel('Cell Shape')
    plt.ylabel('Number of CRISPR Arrays')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'num_crispr_arrays_by_cell_shape_overall.png'))
    plt.close()
    print(f"Overall boxplot saved to: {os.path.join(args.output_folder, 'num_crispr_arrays_by_cell_shape_overall.png')}")

    # Violin plots for cell shape across taxa
    for rank in taxonomic_ranks:
        # Identify taxa with multiple cell shapes
        shape_counts = merged_df.groupby(rank)['cell_shape'].nunique()
        valid_taxa = shape_counts[shape_counts > 1].index  # Taxa with multiple cell shapes

        # Filter the dataframe to include only valid taxa
        filtered_df = merged_df[merged_df[rank].isin(valid_taxa)]

        if filtered_df.empty:
            print(f"No taxa with multiple cell shapes found for rank: {rank}. Skipping violin plot.")
            continue

        # Calculate the overall average num_crispr_arrays per taxon for ordering
        avg_crispr = filtered_df.groupby(rank)['num_crispr_arrays'].mean().sort_values(ascending=False)

        # Define the order based on average number of CRISPR arrays
        order = avg_crispr.index

        plt.figure(figsize=(14,8))
        sns.violinplot(
            x=rank,
            y='num_crispr_arrays',
            hue='cell_shape',
            data=filtered_df,
            split=True,
            inner='quartile',
            palette='Set2',
            order=order
        )
        plt.title(f'Distribution of Number of CRISPR Arrays by Cell Shape and {rank}')
        plt.xlabel(rank)
        plt.ylabel('Number of CRISPR Arrays')
        plt.xticks(rotation=45)
        plt.legend(title='Cell Shape', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        filename = f'violin_crispr_arrays_cell_shape_{rank.lower()}.png'
        plt.savefig(os.path.join(args.output_folder, filename))
        plt.close()
        print(f"Violin plot saved to: {os.path.join(args.output_folder, filename)}")

    # Summary Report
    with open(os.path.join(args.output_folder, 'summary_report.txt'), 'w') as report:
        report.write("=== Summary Report ===\n")
        report.write(f"Total samples after merging: {len(merged_df)}\n\n")
        report.write("Basic Statistics:\n")
        report.write(str(merged_df[['cell_shape', 'num_crispr_arrays']].describe(include='all')))
        report.write("\n\nNumber of Samples per Cell Shape:\n")
        report.write(str(merged_df['cell_shape'].value_counts()))
        report.write("\n\nNumber of Samples per Taxonomic Rank:\n")
        for rank in taxonomic_ranks:
            report.write(f"\n{rank}:\n")
            report.write(str(merged_df[rank].value_counts()))
        report.write("\n\nCorrelation Matrix:\n")
        # Encode categorical variables for correlation matrix
        categorical_vars = ['cell_shape'] + taxonomic_ranks
        encoded_df = pd.get_dummies(merged_df[categorical_vars], drop_first=True)
        correlation_df = pd.concat([merged_df[['num_crispr_arrays']], encoded_df], axis=1)
        correlation = correlation_df.corr()
        report.write(str(correlation))
    print(f"Summary report saved to: {os.path.join(args.output_folder, 'summary_report.txt')}")

if __name__ == "__main__":
    main()