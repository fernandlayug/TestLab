import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from Excel file
df = pd.read_excel('balanced_data.xlsx')

# Compute correlation matrix
corr = df.corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
