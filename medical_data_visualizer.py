import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data from medical_examination.csv
df = pd.read_csv("medical_examination.csv")

# 2. Create the overweight column in the df variable
df['overweight'] = df['weight'] / (df['height'] / 100) ** 2  # BMI calculation
df['overweight'] = (df['overweight'] > 25).astype(int)  # 1 if overweight, 0 otherwise

# 3. Normalize data by making 0 always good and 1 always bad for cholesterol and gluc
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)  # 1 if above normal, 0 if normal
df['gluc'] = (df['gluc'] > 1).astype(int)  # 1 if above normal, 0 if normal

# 4. Draw the Categorical Plot
def draw_cat_plot():
    # 5. Convert data into long format
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 6. Group and reformat data to show counts of each feature
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    
    # Rename 'size' column to 'total' to match expected output
    df_cat.rename(columns={'size': 'total'}, inplace=True)

    # 7. Create the catplot using seaborn
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig

    # 8. Get the figure for the output and store it in the fig variable
    fig.savefig('catplot.png')
    return fig

# 10. Draw the Heat Map
def draw_heat_map():
    # 11. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Diastolic pressure should be less than or equal to systolic pressure
        (df['height'] >= df['height'].quantile(0.025)) &  # Height within 2.5th to 97.5th percentile
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Plot the heatmap using seaborn
    sns.heatmap(corr, annot=True, mask=mask, square=True, fmt='.1f', center=0, cmap='coolwarm')

    # 16. Save the heatmap
    fig.savefig('heatmap.png')
    return fig