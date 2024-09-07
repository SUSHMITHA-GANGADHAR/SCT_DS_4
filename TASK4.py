'''Analyze traffic accident data to identify patterns
related to road conditions, weather, and time of day.
Visualize accident hotspots and contributing factors.'''

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load a sample of the dataset for faster processing
data_filepath = "D:/us_accident.csv"  # Specify the correct path to your dataset
df = pd.read_csv(data_filepath, nrows=100000)
print("Data loaded successfully.")
print(df.head(10))

# General information about the dataset
print("Number of columns:", len(df.columns))
print("Number of rows:", len(df))
df.info()

# Drop columns with a high percentage of missing values
missing_threshold = 0.3  # Set a threshold for dropping columns with more than 30% missing values
df = df.dropna(thresh=len(df) * (1 - missing_threshold), axis=1)
print("Dropped columns with more than 30% missing values.")

# Separate categorical and numerical features
df_cat = df.select_dtypes('object').drop(['ID'], axis=1, errors='ignore')
df_num = df.select_dtypes(include=np.number)

# Drop columns that are not useful for analysis
df_cat.drop(['Description', 'Zipcode', 'Weather_Timestamp'], axis=1, inplace=True, errors='ignore')
df.drop(['Airport_Code'], axis=1, inplace=True, errors='ignore')

# Convert 'Start_Lng', 'Start_Lat', and 'Temperature(F)' to numeric (correct column names)
df['Start_Lng'] = pd.to_numeric(df['Start_Lng'], errors='coerce')
df['Start_Lat'] = pd.to_numeric(df['Start_Lat'], errors='coerce')
df['Temperature(F)'] = pd.to_numeric(df['Temperature(F)'], errors='coerce')

# Display the cleaned dataset
print("Cleaned dataset:")
print(df.head(10))

# Heatmap of numerical feature correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df_num.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Plot the top 10 accident hotspots by city
if 'City' in df.columns:
    plt.figure(figsize=(10, 6))
    top_cities = df['City'].value_counts().nlargest(10)
    sns.barplot(x=top_cities.values, y=top_cities.index, palette='viridis')
    plt.title('Top 10 Accident Hotspots by City')
    plt.ylabel('City')
    plt.xlabel('Number of Accidents')
    plt.show()

# Accident severity count
if 'Severity' in df.columns:
    plt.figure(figsize=(10, 6))
    severity_count = df['Severity'].value_counts()
    plt.pie(severity_count, labels=severity_count.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Severity Count')
    plt.show()

# Road conditions analysis (assuming there's a 'Road_Condition' column)
if 'Road_Condition' in df.columns:
    plt.figure(figsize=(10, 6))
    df['Road_Condition'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
    plt.title('Accidents by Road Condition')
    plt.ylabel('')
    plt.show()

# Weather conditions analysis
if 'Weather_Condition' in df.columns:
    plt.figure(figsize=(10, 6))
    df['Weather_Condition'].value_counts().nlargest(10).plot(kind='bar', color='coral')
    plt.title('Accidents by Weather Conditions')
    plt.xlabel('Weather Condition')
    plt.ylabel('Number of Accidents')
    plt.show()

# Time of day analysis
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['Hours'] = df['Start_Time'].dt.hour
sns.histplot(df['Hours'], bins=20, kde=False, color='blue')
plt.title('Accidents by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Number of Accidents')
plt.tight_layout()
plt.show()

# Scatter plot of Longitude vs Latitude
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Start_Lng', y='Start_Lat', data=df, hue='State', palette='hsv', alpha=0.8, legend=False)
plt.title('Accidents Location')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

for state in df['State'].unique():
    state_data = df[df['State'] == state]
    plt.text(state_data['Start_Lng'].mean(), state_data['Start_Lat'].mean(), state,
             horizontalalignment='center', size='medium', color='black', weight='bold')

plt.tight_layout()
plt.show()

# Box plot for temperature by accident severity
if 'Severity' in df.columns and 'Temperature(F)' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Severity', y='Temperature(F)', data=df)
    plt.title('Boxplot of Temperature by Accident Severity')
    plt.ylabel('Temperature (F)')
    plt.xlabel('Severity')
    plt.show()

# Violin plot of wind speed by accident severity (if wind speed exists)
if 'Wind_Speed(mph)' in df.columns and 'Severity' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Severity', y='Wind_Speed(mph)', data=df)
    plt.title('Violin Plot of Wind Speed by Accident Severity')
    plt.ylabel('Wind Speed (mph)')
    plt.xlabel('Severity')
    plt.show()

# Density plot for temperature
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Temperature(F)'].dropna(), fill=True, color='green')
plt.title('Density of Temperature by Accident Temperature')
plt.ylabel('Density')
plt.xlabel('Temperature (F)')
plt.show()

# Adjust the sample size for the pairplot based on the dataframe size
sample_size = min(len(df), 5000)  # Choose the smaller value between 5000 and the length of the dataframe

df_sample = df.sample(n=sample_size, random_state=42)
pairplot = sns.pairplot(df_sample[df_num.columns], diag_kind='kde', corner=True,
                        height=2.5, aspect=1.0, plot_kws={'s': 25, 'alpha': 0.6})
plt.suptitle('Pairplot of Sampled Numerical Features', y=1.0, fontsize=24, weight='bold')

pairplot.fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)

for ax in pairplot.axes.flat:
    if ax is not None:
        ax.set_ylabel(ax.get_ylabel(), rotation=0, labelpad=30)

plt.show()
