import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Train a regression model to predict CRISPR-related variables.")
    parser.add_argument("--summary_table", required=True, help="Path to the summary table CSV generated by summarize_crisprs.py.")
    parser.add_argument("--metadata_table", required=True, help="Path to the metadata CSV file (e.g., limited.csv).")
    parser.add_argument("--taxa_table", required=True, help="Path to the taxa CSV file (e.g., taxa.csv).")
    parser.add_argument("--model", choices=['linear', 'random_forest'], default='random_forest', help="Type of regression model to use.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--output_folder", default="model_output", help="Folder to save model outputs and results.")
    parser.add_argument("--target_variable", choices=[
        'num_crispr_arrays', 'average_crispr_length', 'average_AT', 
        'percent_plus_orientation', 'percent_minus_orientation',
        'average_DR_length', 'sd_DR_length', 'average_spacers',
        'sd_spacers', 'average_array_length', 'sd_array_length',
        'average_conservation_DRs', 'sd_conservation_DRs',
        'average_conservation_spacers', 'sd_conservation_spacers',
        'most_common_DR_consensus', 'leftflank_kmer_freq',
        'rightflank_kmer_freq'],
        default='num_crispr_arrays', help="Target variable to predict.")
    parser.add_argument("--taxon_level", choices=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'],
                        default='Class', help="Taxonomic level to include in the model.")
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Read the summary table (semicolon-separated)
    try:
        summary_df = pd.read_csv(args.summary_table, delimiter=';')
    except Exception as e:
        print(f"Error reading summary table: {e}")
        sys.exit(1)

    # Check if 'sample' column and target variable exist
    if 'sample' not in summary_df.columns:
        print("Error: 'sample' column not found in summary table.")
        print("Available columns:", summary_df.columns.tolist())
        sys.exit(1)
    if args.target_variable not in summary_df.columns:
        print(f"Error: '{args.target_variable}' column not found in summary table.")
        print("Available columns:", summary_df.columns.tolist())
        sys.exit(1)

    # Read the metadata table (comma-separated)
    try:
        metadata_df = pd.read_csv(args.metadata_table, delimiter=',')
    except Exception as e:
        print(f"Error reading metadata table: {e}")
        sys.exit(1)

    # Read the taxa table (comma-separated)
    try:
        taxa_df = pd.read_csv(args.taxa_table, delimiter=',')
    except Exception as e:
        print(f"Error reading taxa table: {e}")
        sys.exit(1)

    # Merge the dataframes using a left join to include only samples from summary_df
    summary_df['sample'] = summary_df['sample'].str.strip()
    metadata_df['sample'] = metadata_df['sample'].str.strip()
    taxa_df['sample'] = taxa_df['sample'].str.strip()

    # Ensure consistent file extensions if necessary
    summary_df['sample'] = summary_df['sample'].str.replace('.fasta', '.fasta', regex=False)
    metadata_df['sample'] = metadata_df['sample'].str.replace('.fasta', '.fasta', regex=False)
    taxa_df['sample'] = taxa_df['sample'].str.replace('.fasta', '.fasta', regex=False)

    # Perform left merge to retain only samples present in summary_df
    merged_df = pd.merge(summary_df, metadata_df, on='sample', how='left')
    merged_df = pd.merge(merged_df, taxa_df[['sample', args.taxon_level]], on='sample', how='left')

    # Handle missing values in the target variable without assuming zeros
    if args.target_variable in merged_df.columns:
        if merged_df[args.target_variable].isnull().any():
            if args.target_variable == 'num_crispr_arrays':
                # Depending on the context, you might choose to drop these rows or handle differently
                merged_df = merged_df.dropna(subset=[args.target_variable])
                print(f"Dropped samples with missing '{args.target_variable}' values.")
            else:
                merged_df = merged_df.dropna(subset=[args.target_variable])
                print(f"Dropped samples with missing '{args.target_variable}' values.")
    else:
        print(f"Error: '{args.target_variable}' column not found after merging.")
        sys.exit(1)

    merged_df[args.taxon_level] = merged_df[args.taxon_level].fillna('Unknown')

    if merged_df.empty:
        print("The merged dataframe is empty. Please check if the sample names match between the tables.")
        sys.exit(1)

    print(f"Number of samples after merging: {len(merged_df)}")

    # Prepare the features and target
    y = merged_df[args.target_variable]
    X = merged_df.drop(columns=[
        'sample',
        # Remove all potential target variables
        'num_crispr_arrays', 'average_crispr_length', 'average_AT', 
        'percent_plus_orientation', 'percent_minus_orientation',
        'average_DR_length', 'sd_DR_length', 'average_spacers',
        'sd_spacers', 'average_array_length', 'sd_array_length',
        'average_conservation_DRs', 'sd_conservation_DRs',
        'average_conservation_spacers', 'sd_conservation_spacers',
        'most_common_DR_consensus', 'dr_nucleotide_distribution', 
        'spacer_nucleotide_distribution', 'spacer_kmer_frequencies',
        'leftflank_kmer_freq', 'rightflank_kmer_freq', 'leader_flank_kmer_freq'
    ])

    # Handle categorical variables using one-hot encoding
    categorical_cols = X.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Check for missing values
    if X.isnull().any().any():
        print("Missing values detected in the features. Filling missing values with the mean of each column.")
        X = X.fillna(X.mean())

    # Perform random split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    print(f"Train/Test split completed using random split.")

    # Train the model
    if args.model == 'linear':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel: {args.model}")
    print(f"Target Variable: {args.target_variable}")
    print(f"Taxonomic Level: {args.taxon_level}")
    print(f"Mean squared error (MSE): {mse:.2f}")
    print(f"R² score: {r2:.2f}\n")

    # Create a scatter plot of predicted vs real values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual Values ({args.model.capitalize()} Regression)\nTarget: {args.target_variable}, Taxon: {args.taxon_level}')
    plt.tight_layout()
    plot_path = os.path.join(args.output_folder, f'predicted_vs_actual_plot_{args.target_variable}_{args.taxon_level}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Predicted vs Actual plot saved to: {plot_path}")

    # Save the model outputs
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    results_df.to_csv(os.path.join(args.output_folder, f'prediction_results_{args.target_variable}_{args.taxon_level}.csv'), index=False)
    print(f"Prediction results saved to: {os.path.join(args.output_folder, f'prediction_results_{args.target_variable}_{args.taxon_level}.csv')}")

    if args.model == 'random_forest':
        # Feature importance
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        feature_importances = feature_importances.sort_values(ascending=False)
        print("\nFeature Importances:")
        print(feature_importances)
        
        # Calculate correlation between features and target
        correlations = X.corrwith(y)
        correlations = correlations.sort_values(ascending=False)
        print("\nFeature Correlations with Target:")
        print(correlations)
        
        # Combine feature importances and correlations into a single DataFrame
        feature_analysis = pd.DataFrame({
            'Feature Importance': feature_importances,
            'Correlation with Target': correlations
        })
        feature_analysis = feature_analysis.sort_values(by='Feature Importance', ascending=False)
        
        # Save feature analysis
        feature_analysis.to_csv(os.path.join(args.output_folder, f'feature_analysis_{args.target_variable}_{args.taxon_level}.csv'))
        print(f"Feature analysis (importance & correlation) saved to: {os.path.join(args.output_folder, f'feature_analysis_{args.target_variable}_{args.taxon_level}.csv')}")
        
        # === Updated Code: Plot Feature Importance Colored by Correlation ===
        plt.figure(figsize=(12, 20))  # Increased height
        
        # Scatter plot: Feature Importance vs. Correlation with Target
        scatter = plt.scatter(
            feature_analysis['Feature Importance'],
            range(len(feature_analysis)),  # Use range for y-axis
            c=feature_analysis['Correlation with Target'],
            cmap='coolwarm',
            edgecolor='k',
            s=100
        )
        
        plt.colorbar(scatter, label='Correlation with Target')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title(f'Feature Importance and Correlation with Target\nModel: Random Forest, Target: {args.target_variable}, Taxon: {args.taxon_level}')
        
        # Set y-ticks and labels
        plt.yticks(range(len(feature_analysis)), feature_analysis.index)
        
        # Invert y-axis to have most important features at the top
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        # Save the plot as PDF
        plot_path = os.path.join(args.output_folder, f'feature_importance_correlation_plot_{args.target_variable}_{args.taxon_level}.pdf')
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Feature importance vs. correlation plot saved to: {plot_path}")
        # ================================================================
    
    elif args.model == 'linear':
        # Extract coefficients and intercept
        coefficients = pd.Series(model.coef_, index=X.columns)
        intercept = model.intercept_
        print("\nRegression Coefficients:")
        print(coefficients)
        print(f"\nIntercept: {intercept:.4f}")
        
        # Save coefficients and intercept
        coefficients.to_csv(os.path.join(args.output_folder, f'regression_coefficients_{args.target_variable}_{args.taxon_level}.csv'))
        with open(os.path.join(args.output_folder, f'regression_equation_{args.target_variable}_{args.taxon_level}.txt'), 'w') as f:
            equation = f"{args.target_variable} = {intercept:.4f}"
            for feature, coef in coefficients.items():
                equation += f" + ({coef:.4f} * {feature})"
            f.write(equation)
        
        print(f"Regression coefficients saved to: {os.path.join(args.output_folder, f'regression_coefficients_{args.target_variable}_{args.taxon_level}.csv')}")
        print(f"Regression equation saved to: {os.path.join(args.output_folder, f'regression_equation_{args.target_variable}_{args.taxon_level}.txt')}")
    
        # Optionally, print the regression equation in a readable format
        print("\nRegression Equation:")
        print(equation)
        
        # === Optional: Plot Regression Coefficients ===
        # (You can add a similar plot for linear regression coefficients if desired)
        # ===================================================
    
if __name__ == '__main__':
    main()