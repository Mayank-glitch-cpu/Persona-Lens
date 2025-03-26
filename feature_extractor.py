import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from rake_nltk import Rake
from datetime import datetime
import seaborn as sns

# Load the dataset
file_path = '/home/maya/persona-lens/Persona-Lens/Dataset/github_users_combined.csv'  # Replace with the actual path to your dataset
data = pd.read_csv(file_path)

# Get the column names
column_names = data.columns.tolist()

# Get the number of rows and columns
num_rows, num_columns = data.shape

# Print the results
# print("Column Names:", column_names)
# print("Number of Rows:", num_rows)
# print("Number of Columns:", num_columns)

# Load the new dataset
file_path = '/home/maya/persona-lens/Persona-Lens/Dataset/github_users_combined.csv'  # Replace with the actual path to your dataset
data = pd.read_csv(file_path)

# 1. Define the Columns to Keep
required_columns = [
    'username', 'bio', 'profile_links', 'public_repos', 'total_stars', 'total_forks',
    'most_used_language', 'followers', 'following', 
    'account_created', 'location', 'company', 'email', 'blog', 'language_stats'
]

# 2. Filter the Dataset to Keep Only the Required Columns
filtered_data = data[required_columns]

# 3. Save the Cleaned Dataset
output_file_path = 'cleaned_dataset.csv'  # Replace with your desired output file path
filtered_data.to_csv(output_file_path, index=False)

print("Cleaned dataset saved to:", output_file_path)
print("Columns in the cleaned dataset:", filtered_data.columns.tolist())
print("Number of rows in the cleaned dataset:", len(filtered_data))


# Load the cleaned dataset
file_path = 'cleaned_dataset.csv'  # Replace with the actual path to your cleaned dataset
data = pd.read_csv(file_path)

# 1. Calculate Experience Level
def calculate_experience(account_created):
    if pd.isna(account_created):
        return 0
    created_date = pd.to_datetime(account_created)  # Convert to datetime using pandas
    current_time = pd.Timestamp.now()  # Get current time dynamically
    return (current_time - created_date).days / 365.25  # Use 365.25 to account for leap years

data['experience_years'] = data['account_created'].apply(calculate_experience)
# 2. Extract Skill Keywords from Bio
# Create a set of valid skills
skills_data = {
    'Python', 'Java', 'C++', 'SQL', 'NoSQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Cassandra', 
    'Redis', 'Apache Spark', 'Hadoop', 'Kafka', 'Airflow', 'ETL Pipelines', 'Data Warehousing',
    'Data Modeling', 'Tableau', 'Power BI', 'Looker', 'Excel', 'Google Sheets', 'Jupyter Notebook',
    'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Scikit-learn', 'TensorFlow', 'PyTorch', 'Keras',
    'XGBoost', 'LightGBM', 'Deep Learning', 'Neural Networks', 'Natural Language Processing',
    'Computer Vision', 'Reinforcement Learning', 'AWS', 'Azure', 'Google Cloud Platform',
    'Docker', 'Kubernetes', 'Terraform', 'Ansible', 'Jenkins', 'Git', 'GitHub', 'Bitbucket',
    'CI/CD', 'Linux', 'Bash', 'PowerShell', 'Microservices', 'REST', 'GraphQL', 'HTML',
    'CSS', 'JavaScript', 'TypeScript', 'React', 'Vue', 'Angular', 'Node.js', 'Django',
    'Flask', 'FastAPI'
}

def extract_skills(bio):
    if pd.isna(bio):
        return []
    # Convert bio to lowercase for case-insensitive matching
    bio_lower = bio.lower()
    # Find all skills that appear in the bio
    found_skills = [skill for skill in skills_data 
                   if skill.lower() in bio_lower]
    return list(set(found_skills))  # Remove duplicates

data['skills'] = data['bio'].apply(extract_skills)

# 3. Calculate Popularity Score
# Current weights (40/60) - consider if you meant 0.4/0.6 instead?
weighted_score = data['total_stars'] * 40 + data['total_forks'] * 60

# Apply logarithmic transformation to reduce outlier impact
log_score = np.log1p(weighted_score)  # log(1 + x)

# Normalize between 1-10
min_log = log_score.min()
max_log = log_score.max()

data['popularity_score'] = 1 + 9 * ((log_score - min_log) / (max_log - min_log))

# 4. Parse Language Stats (if needed)
# Assuming language_stats is a JSON string or dictionary
import ast
def parse_language_stats(language_stats):
    if pd.isna(language_stats):
        return {}
    return ast.literal_eval(language_stats)

data['language_stats_parsed'] = data['language_stats'].apply(parse_language_stats)

# Save the dataset with extracted features
output_file_path = 'dataset_with_extracted_features.csv'  # Replace with your desired output file path
data.to_csv(output_file_path, index=False)

print("Dataset with extracted features saved to:", output_file_path)

#************************************************************************************
#***********************Data Visualization*******************************************
#************************************************************************************


# Load the dataset with extracted features
file_path = 'dataset_with_extracted_features.csv'  # Replace with the actual path to your dataset
data = pd.read_csv(file_path)

# 1. Distribution of Experience Years
plt.figure(figsize=(10, 6))
sns.histplot(data['experience_years'], bins=20, kde=True)
plt.title('Distribution of Experience Years')
plt.xlabel('Experience Years')
plt.ylabel('Frequency')
plt.savefig('experience_years_distribution.png')
plt.show()

# 2. Distribution of Popularity Score
plt.figure(figsize=(10, 6))
sns.histplot(data['popularity_score'], bins=20, kde=True)
plt.title('Distribution of Popularity Score')
plt.xlabel('Popularity Score')
plt.ylabel('Frequency')
plt.savefig('popularity_score_distribution.png')
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10, 6))
corr = data[['public_repos', 'total_stars', 'total_forks', 'followers', 'following', 'experience_years', 'popularity_score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()