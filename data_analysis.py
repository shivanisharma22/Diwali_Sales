# Import Python libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib
import os  # To handle folder creation

# Use a non-interactive backend to save plots as images
matplotlib.use('Agg')

# Ensure the output folder exists
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Import the CSV file
df = pd.read_csv(r'D:\Python_diwali\Data\Diwali Sales Data.csv', encoding='unicode_escape')

# Data overview
print(df.shape)
print(df.head())
print(df.info())

# Drop unrelated/blank columns
df.drop(['Status', 'unnamed1'], axis=1, inplace=True)

# Check for null values and handle them
print(pd.isnull(df).sum())
df.dropna(inplace=True)

# Change the data type of the 'Amount' column
df['Amount'] = df['Amount'].astype('int')
print(df['Amount'].dtypes)

# Describe the data
print(df.describe())
print(df[['Age', 'Orders', 'Amount']].describe())

# Visualizations
sns.set(rc={'figure.figsize': (10, 5)})

# 1. Bar chart: Gender and its count
ax = sns.countplot(x='Gender', data=df)
for bars in ax.containers:
    ax.bar_label(bars)
plt.title('Gender Count')
plt.savefig(f'{output_dir}/gender_count.png')  # Save the plot
plt.close()

# 2. Bar chart: Gender vs Total Amount
sales_gen = df.groupby(['Gender'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
sns.barplot(x='Gender', y='Amount', data=sales_gen)
plt.title('Total Amount by Gender')
plt.savefig(f'{output_dir}/gender_vs_amount.png')  # Save the plot
plt.close()

# 3. Bar chart: Age Group vs Gender
ax = sns.countplot(data=df, x='Age Group', hue='Gender')
for bars in ax.containers:
    ax.bar_label(bars)
plt.title('Age Group and Gender Distribution')
plt.savefig(f'{output_dir}/age_group_gender.png')  # Save the plot
plt.close()

# 4. Total Amount vs Age Group
sales_age = df.groupby(['Age Group'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
sns.barplot(x='Age Group', y='Amount', data=sales_age)
plt.title('Total Amount by Age Group')
plt.savefig(f'{output_dir}/age_group_vs_amount.png')  # Save the plot
plt.close()

# 5. Total number of orders from top 10 states
sales_state_orders = df.groupby(['State'], as_index=False)['Orders'].sum().sort_values(by='Orders', ascending=False).head(10)
sns.barplot(data=sales_state_orders, x='State', y='Orders')
plt.title('Top 10 States by Orders')
plt.savefig(f'{output_dir}/state_orders.png')  # Save the plot
plt.close()

# 6. Total amount/sales from top 10 states
sales_state_amount = df.groupby(['State'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False).head(10)
sns.barplot(data=sales_state_amount, x='State', y='Amount')
plt.title('Top 10 States by Amount')
plt.savefig(f'{output_dir}/state_amount.png')  # Save the plot
plt.close()

# 7. Marital Status vs Total Amount
sales_marital = df.groupby(['Marital_Status', 'Gender'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
sns.barplot(data=sales_marital, x='Marital_Status', y='Amount', hue='Gender')
plt.title('Marital Status and Total Amount by Gender')
plt.savefig(f'{output_dir}/marital_status_vs_amount.png')  # Save the plot
plt.close()

# 8. Occupation vs Total Amount
sales_occupation = df.groupby(['Occupation'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
sns.barplot(data=sales_occupation, x='Occupation', y='Amount')
plt.title('Occupation and Total Amount')
plt.xticks(rotation=45)
plt.savefig(f'{output_dir}/occupation_vs_amount.png')  # Save the plot
plt.close()

# 9. Product Category vs Total Amount
sales_product_category = df.groupby(['Product_Category'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False).head(10)
sns.barplot(data=sales_product_category, x='Product_Category', y='Amount')
plt.title('Top 10 Product Categories by Amount')
plt.xticks(rotation=45)
plt.savefig(f'{output_dir}/product_category_vs_amount.png')  # Save the plot
plt.close()

# 10. Most sold products
top_products = df.groupby('Product_ID')['Orders'].sum().nlargest(10)
top_products.plot(kind='bar', figsize=(12, 7), title='Top 10 Most Sold Products')
plt.xlabel('Product ID')
plt.ylabel('Number of Orders')
plt.savefig(f'{output_dir}/top_products.png')  # Save the plot
plt.close()

print(f"All plots have been saved in the '{output_dir}' folder.")
