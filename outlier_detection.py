from Chapter3.OutlierDetection import DistributionBasedOutlierDetection, DistanceBasedOutlierDetection
import pandas as pd
import numpy as np
# Load the final dataset
final_df = pd.read_csv('/Users/macbookair/Desktop/ML4QS_Group125/data/merged_data_0.1.csv')

# Columns to check for outliers (excluding non-numeric ones)
numeric_columns = final_df.select_dtypes(include=[np.number]).columns.tolist()
# Initialize outlier detection classes
dist_based = DistributionBasedOutlierDetection()
dist_outlier = DistanceBasedOutlierDetection()

# Apply Chauvenet criterion
for col in numeric_columns:
    final_df = dist_based.chauvenet(final_df, col, C=1)
    print('Outliers found for column:', col, 'are:', final_df[col + '_outlier'].sum())
# Apply Mixture Model
for col in numeric_columns:
    final_df = dist_based.mixture_model(final_df, col)
    print('Outliers found for column:', col, 'are:', final_df[col + '_outlier'].sum())

# Apply Local Outlier Factor (LOF)
k = 5  # Number of neighbors
final_df = dist_outlier.local_outlier_factor(final_df, numeric_columns, 'euclidean', k)

# Save the results
final_df.to_csv('/Users/macbookair/Desktop/ML4QS_Group125/data/merged_outliers_detected.csv', index=False)

print("Outlier detection complete. Results saved to 'merged_outliers_detected.csv'.")
