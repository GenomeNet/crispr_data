import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def main():
    parser = argparse.ArgumentParser(description="Analyze the relationship between num_crispr_arrays, spore_formation, and taxonomic classifications.")
    parser.add_argument("--summary_table", required=True, help="Path to the summary table CSV containing num_crispr_arrays.")
    parser.add_argument("--taxa_table", required=True, help="Path to the taxa CSV file containing all taxonomic ranks.")
    parser.add_argument("--metadata_table", required=True, help="Path to the metadata CSV file containing spore formation information.")
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

    # Check if 'sample' and 'spore_formation' columns exist in metadata_df
    metadata_required_columns = ['sample', 'spore_formation']
    for col in metadata_required_columns:
        if col not in metadata_df.columns:
            print(f"Error: '{col}' column not found in metadata table.")
            sys.exit(1)

    # Clean 'sample' columns by stripping whitespace and standardizing file extensions
    summary_df['sample'] = summary_df['sample'].str.strip()
    taxa_df['sample'] = taxa_df['sample'].str.strip()
    metadata_df['sample'] = metadata_df['sample'].str.strip()

    # Merge the dataframes
    merged_df = summary_df.merge(metadata_df[['sample', 'spore_formation']], on='sample', how='left')
    merged_df = merged_df.merge(taxa_df, on='sample', how='left')

    # Handle missing values in 'spore_formation'
    merged_df['spore_formation'] = merged_df['spore_formation'].fillna(False).astype(bool)
    print("Filled missing 'spore_formation' values with False.")

    # Ensure all taxonomic ranks are available
    taxonomic_ranks = ['Superkingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    for rank in taxonomic_ranks:
        if rank not in merged_df.columns:
            print(f"Warning: '{rank}' column not found in taxa table.")
            merged_df[rank] = 'Unknown'

    # Define significance levels and their symbols
    significance_levels = {
        0.001: '***',
        0.01: '**',
        0.05: '*',
        1.0: 'ns'  # not significant
    }
    
    ### Analysis focusing on spore formation across taxonomic ranks ###
    
    # For each taxonomic rank, plot the average num_crispr_arrays for spore-forming and non-spore-forming bacteria
    for rank in taxonomic_ranks:
        # Identify taxa with both spore-forming and non-spore-forming samples
        spore_counts = merged_df.groupby(rank)['spore_formation'].nunique()
        valid_taxa = spore_counts[spore_counts == 2].index  # Taxa with both True and False

        # Filter the dataframe to include only valid taxa
        filtered_df = merged_df[merged_df[rank].isin(valid_taxa)]

        if filtered_df.empty:
            print(f"No taxa with both spore-forming and non-spore-forming samples found for rank: {rank}. Skipping plot.")
            continue

        # Calculate the overall average num_crispr_arrays per taxon and sort
        avg_crispr_total = filtered_df.groupby(rank)['num_crispr_arrays'].mean().sort_values(ascending=False)
        order = avg_crispr_total.index.tolist()

        # Prepare data for plotting
        plot_data = []
        p_values = []

        for taxon in order:
            taxon_data = filtered_df[filtered_df[rank] == taxon]
            spore_forming = taxon_data[taxon_data['spore_formation']]['num_crispr_arrays']
            non_spore_forming = taxon_data[~taxon_data['spore_formation']]['num_crispr_arrays']
            
            plot_data.append({
                rank: taxon,
                'spore_formation': False,
                'num_crispr_arrays': non_spore_forming.mean(),
                'sample_size': len(non_spore_forming)
            })
            plot_data.append({
                rank: taxon,
                'spore_formation': True,
                'num_crispr_arrays': spore_forming.mean(),
                'sample_size': len(spore_forming)
            })
            
            if len(spore_forming) > 1 and len(non_spore_forming) > 1:
                t_stat, p_val = ttest_ind(spore_forming, non_spore_forming, equal_var=False, nan_policy='omit')
                p_values.append(p_val)
            else:
                p_values.append(None)

        plot_df = pd.DataFrame(plot_data)

        # Create the plot
        plt.figure(figsize=(12,8))
        ax = sns.barplot(
            x=rank,
            y='num_crispr_arrays',
            hue='spore_formation',
            data=plot_df,
            ci=None,
            palette='Set1',
            order=order
        )

        plt.title(f'Average Number of CRISPR Arrays by Spore Formation Status across {rank}')
        plt.xlabel(rank)
        plt.ylabel('Average Number of CRISPR Arrays')
        plt.xticks(rotation=45)
        plt.legend(title='Spore Formation')

        # Add sample size annotations
        for i, taxon in enumerate(order):
            taxon_data = plot_df[plot_df[rank] == taxon]
            for j, (_, row) in enumerate(taxon_data.iterrows()):
                x = i + [-0.2, 0.2][j]  # -0.2 for non-spore, 0.2 for spore
                y = row['num_crispr_arrays']
                spore_status = 'Spore' if row['spore_formation'] else 'Non-spore'
                ax.text(x, y + 0.05, f'{spore_status}\nn={row["sample_size"]}', ha='center', va='bottom', fontsize=8)

        # Apply FDR correction
        valid_indices = [i for i, p in enumerate(p_values) if p is not None]
        valid_p_values = [p_values[i] for i in valid_indices]
        if valid_p_values:
            corrected = multipletests(valid_p_values, alpha=0.05, method='fdr_bh')
            corrected_p = corrected[1]

            # Map corrected p-values back to taxa
            corrected_p_values = [None] * len(p_values)
            for idx, p_corr in zip(valid_indices, corrected_p):
                corrected_p_values[idx] = p_corr

            # Add significance annotations
            for i, (taxon, p_val) in enumerate(zip(order, corrected_p_values)):
                if p_val is not None:
                    for thresh, symbol in significance_levels.items():
                        if p_val < thresh:
                            sig = symbol
                            break
                    else:
                        sig = 'ns'
                    y = max(plot_df[plot_df[rank] == taxon]['num_crispr_arrays'])
                    ax.text(i, y + (y * 0.05), sig, ha='center', va='bottom', color='black', fontsize=12)

        # Add significance legend
        from matplotlib.patches import Patch
        significance_labels = [f"{symbol}: p < {thresh}" for thresh, symbol in significance_levels.items() if symbol != 'ns']
        # Create custom legend entries
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor='none', edgecolor='none', label='Significance Levels'))
        labels.append('')
        for label in significance_labels:
            handles.append(Patch(facecolor='none', edgecolor='none', label=label))
            labels.append(label)
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        filename = f'avg_crispr_arrays_spore_{rank.lower()}.png'
        plt.savefig(os.path.join(args.output_folder, filename))
        plt.close()
        print(f"Bar plot saved to: {os.path.join(args.output_folder, filename)}")

    # Statistical analysis to test if the difference is significant at each taxonomic rank
    with open(os.path.join(args.output_folder, 'statistical_tests_spore_formation.txt'), 'w') as report:
        report.write("Statistical Tests for Difference in num_crispr_arrays between Spore-forming and Non-Spore-forming Bacteria\n\n")
        for rank in taxonomic_ranks:
            report.write(f'=== {rank} Level ===\n')
            # Identify taxa with both spore-forming and non-spore-forming samples
            spore_counts = merged_df.groupby(rank)['spore_formation'].nunique()
            valid_taxa = spore_counts[spore_counts == 2].index  # Taxa with both True and False

            # Filter the dataframe to include only valid taxa
            filtered_df = merged_df[merged_df[rank].isin(valid_taxa)]

            for taxa in filtered_df[rank].unique():
                sub_df = filtered_df[filtered_df[rank] == taxa]
                spore_forming = sub_df[sub_df['spore_formation']]['num_crispr_arrays']
                non_spore_forming = sub_df[~sub_df['spore_formation']]['num_crispr_arrays']

                if len(spore_forming) > 1 and len(non_spore_forming) > 1:
                    t_stat, p_value = ttest_ind(spore_forming, non_spore_forming, equal_var=False, nan_policy='omit')
                    report.write(f'{taxa}:\n')
                    report.write(f'  Spore-forming mean: {spore_forming.mean():.2f} (n={len(spore_forming)})\n')
                    report.write(f'  Non-spore-forming mean: {non_spore_forming.mean():.2f} (n={len(non_spore_forming)})\n')
                    report.write(f'  t-statistic: {t_stat:.3f}, p-value: {p_value:.3e}\n\n')
                else:
                    report.write(f'{taxa}:\n')
                    report.write('  Insufficient data for statistical test (one of the groups has less than 2 samples).\n\n')
        print(f"Statistical test results saved to: {os.path.join(args.output_folder, 'statistical_tests_spore_formation.txt')}")

    # Overall boxplot of num_crispr_arrays by spore_formation
    plt.figure(figsize=(8,6))
    sns.boxplot(x='spore_formation', y='num_crispr_arrays', data=merged_df)
    plt.title('Distribution of Number of CRISPR Arrays by Spore Formation')
    plt.xlabel('Spore Formation')
    plt.ylabel('Number of CRISPR Arrays')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'num_crispr_arrays_by_spore_formation_overall.png'))
    plt.close()
    print(f"Overall boxplot saved to: {os.path.join(args.output_folder, 'num_crispr_arrays_by_spore_formation_overall.png')}")

    # Remove Violin Plots
    # The violin plots have been removed as per the request.

    # Summary Report
    with open(os.path.join(args.output_folder, 'summary_report.txt'), 'w') as report:
        report.write("=== Summary Report ===\n")
        report.write(f"Total samples after merging: {len(merged_df)}\n\n")
        report.write("Basic Statistics:\n")
        report.write(str(merged_df[['spore_formation', 'num_crispr_arrays']].describe(include='all')))
        report.write("\n\nNumber of Samples per Spore Formation Status:\n")
        report.write(str(merged_df['spore_formation'].value_counts()))
        report.write("\n\nNumber of Samples per Taxonomic Rank:\n")
        for rank in taxonomic_ranks:
            report.write(f"\n{rank}:\n")
            report.write(str(merged_df[rank].value_counts()))
        report.write("\n\nCorrelation Matrix:\n")
        # Encode categorical variables for correlation matrix
        categorical_vars = ['spore_formation'] + taxonomic_ranks
        encoded_df = pd.get_dummies(merged_df[categorical_vars], drop_first=True)
        correlation_df = pd.concat([merged_df[['num_crispr_arrays']], encoded_df], axis=1)
        correlation = correlation_df.corr()
        report.write(str(correlation))
    print(f"Summary report saved to: {os.path.join(args.output_folder, 'summary_report.txt')}")

if __name__ == "__main__":
    main()