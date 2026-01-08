#libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# add merge code also


#display of data
restaurant = pd.read_csv("Zomato_merged.csv")
pd.set_option('display.max_columns', None)
# print(restaurant.head())


#Names of columns
print("Column names: ", restaurant.columns)


#To count Rows and columns
print("Total count of rows and columns: ",restaurant.shape)

# duplicate values count
print("Duplicate values: ",restaurant.duplicated().sum())
restaurant.drop_duplicates(inplace=True)
print("After removal count: ", restaurant.duplicated().sum())
print("Total count of rows and columns: ",restaurant.shape)


#To count missing values(visualizing)
print("Missing values count per column ",)
print(restaurant.isnull().sum())

missing_counts = restaurant.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]  # only columns with missing values

# for col in missing_counts.index:
#     missing = restaurant[col].isnull().sum()
#     not_missing = restaurant[col].notnull().sum()
#     plt.figure(figsize=(6,6))
#     plt.pie([missing, not_missing], labels=['Missing', 'Available'], autopct='%1.1f%%',
#             colors=['red','green'], startangle=90, shadow=True)
#     plt.title(f"Missing Values in '{col}' Column")
#     plt.show()

#unique values for each value
for col in restaurant.columns:
    unique_values = restaurant[col].unique()  # get unique values
    print(f"Column: {col}")
    print(f"Number of unique values: {len(unique_values)}")
    print(f"Unique values (sample): {unique_values[:10]}")  # show first 10 only
    print("-"*50)


#data wrangling


#data visualization