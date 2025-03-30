# #write a script to join 3 csv with simmilar headers into one csv 
# # and look for duplicates in the joined csv
# import pandas as pd
# import os
# import numpy as np
# import csv
# import re
# import logging
# from tqdm import tqdm

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)    

# # Set the directory where your CSV files are located
# directory = 'github-users'

# # Get all CSV files in the directory
# csv_files = [file for file in os.listdir(directory) if file.endswith('.csv') and file.startswith('github_profiles')]
# logger.info(f"Found {len(csv_files)} CSV files: {csv_files}")

# # Create an empty DataFrame to store the combined data
# combined_df = pd.DataFrame()

# # Loop through all the files in the directory
# for file in csv_files:
#     file_path = os.path.join(directory, file)
#     logger.info(f"Reading file: {file_path}")
    
#     try:
#         # Read the CSV file
#         df = pd.read_csv(file_path)
#         logger.info(f"File {file} has {len(df)} rows and columns: {list(df.columns)}")
        
#         # Append the data to the combined DataFrame
#         combined_df = pd.concat([combined_df, df], ignore_index=True)
#     except Exception as e:
#         logger.error(f"Error reading {file}: {str(e)}")

# # Check if we have any data
# if combined_df.empty:
#     logger.error("No data was loaded from the CSV files.")
#     exit(1)

# logger.info(f"Combined data has {len(combined_df)} rows and columns: {list(combined_df.columns)}")

# # Check for exact duplicates based on all columns
# logger.info("Checking for exact duplicates (all columns)...")
# exact_duplicates = combined_df[combined_df.duplicated()]
# logger.info(f"Found {len(exact_duplicates)} exact duplicate rows (all columns)")

# # If there are duplicates, remove them
# if len(exact_duplicates) > 0:
#     combined_df_no_dupes = combined_df.drop_duplicates()
#     logger.info(f"After removing exact duplicates, combined data has {len(combined_df_no_dupes)} rows")
# else:
#     combined_df_no_dupes = combined_df

# # Check for duplicates based on specific columns (like username/login)
# if 'login' in combined_df.columns:
#     logger.info("Checking for duplicates based on 'login' column...")
#     login_duplicates = combined_df_no_dupes[combined_df_no_dupes.duplicated(subset=['login'], keep=False)]
#     logger.info(f"Found {len(login_duplicates)} rows with duplicate login values")
    
#     # Group duplicate logins
#     if not login_duplicates.empty:
#         duplicate_logins = login_duplicates.groupby('login').size().reset_index(name='count')
#         duplicate_logins = duplicate_logins[duplicate_logins['count'] > 1]
#         logger.info(f"Found {len(duplicate_logins)} unique login values that have duplicates")

# # Save the combined dataset (without exact duplicates)
# output_file = "github_users_combined.csv"
# combined_df_no_dupes.to_csv(output_file, index=False)
# logger.info(f"Combined data saved to {output_file}")

# # Save duplicates to a separate file for review
# if 'login' in combined_df.columns and not login_duplicates.empty:
#     duplicates_file = "github_users_duplicates.csv"
#     login_duplicates.to_csv(duplicates_file, index=False)
#     logger.info(f"Duplicate entries saved to {duplicates_file}")

# # Generate a report of duplicates
# report_file = "duplicate_report.txt"
# with open(report_file, "w") as f:
#     f.write(f"Total rows in original combined data: {len(combined_df)}\n")
#     f.write(f"Exact duplicates removed: {len(exact_duplicates)}\n")
#     f.write(f"Total rows after exact deduplication: {len(combined_df_no_dupes)}\n\n")
    
#     if 'login' in combined_df.columns:
#         f.write(f"Records with duplicate 'login' values: {len(login_duplicates)}\n")
#         f.write(f"Unique 'login' values with duplicates: {len(duplicate_logins) if not login_duplicates.empty else 0}\n\n")
        
#         if not login_duplicates.empty:
#             f.write("Sample of login values with duplicates (top 20):\n")
#             for i, (login, count) in enumerate(zip(duplicate_logins['login'].head(20), duplicate_logins['count'].head(20))):
#                 f.write(f"  {login}: {count} occurrences\n")

# logger.info(f"Duplicate analysis report saved to {report_file}")