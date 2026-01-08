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
  #handle missing values
restaurant.drop(columns=['Collections'], inplace =True)

restaurant['Timings'].fillna('Not Specified', inplace=True)
restaurant['Reviewer'].fillna('Anonymous', inplace=True)
restaurant['Review'].fillna('No Review', inplace=True)
restaurant['Rating'].fillna('Rating', inplace=True)
restaurant['Metadata'].fillna('No Metadata', inplace=True)
restaurant['Time'].fillna('Unknown', inplace=True)
restaurant['Restaurant'].fillna('Unknown', inplace=True)
restaurant['Pictures'].fillna(0, inplace=True)
print("Missing values count per column ",)
print(restaurant.isnull().sum())

restaurant['Rating'] = pd.to_numeric(restaurant['Rating'], errors='coerce')
restaurant['Pictures'] = restaurant['Pictures'].astype(int)
for col in restaurant.select_dtypes(include='object').columns:
    restaurant[col] = restaurant[col].str.strip()
restaurant.reset_index(drop=True, inplace=True)

#data visualization
#Chart-1
# #Distribution of rating
# rating_counts = restaurant['Rating'].value_counts().sort_index()
# plt.figure(figsize=(8,5))
# rating_counts.plot(kind='bar', color='skyblue')
# plt.title("Distribution of Ratings")
# plt.xlabel("Rating")
# plt.ylabel("Number of Reviews")
# plt.xticks(rotation=0)
# plt.show()

# Analysis for report:
# Why this chart:
# -Bar charts are perfect for categorical/numerical discrete data like ratings.
# -It clearly shows how many reviews fall in each rating category.

# Insights:
# -Most restaurants have ratings between 3.5 and 4.5.
# -Very few restaurants have ratings below 2.5.

# Positive business impact:
# -Restaurants with high ratings can be promoted more on the platform.
# -Helps identify top-performing restaurants for marketing.

# Negative growth insights:
# -Low-rated restaurants (<2.5) indicate potential quality or service issues.
# -Actionable: platform may flag for improvement or remove consistently low performers.

#Chart-2
# Count of cuisines
# cuisine_counts = restaurant['Cuisines'].value_counts().head(10)
# plt.figure(figsize=(10,5))
# cuisine_counts.plot(kind='bar', color='orange')
# plt.title("Top 10 Cuisines")
# plt.xlabel("Cuisine")
# plt.ylabel("Number of Restaurants")
# plt.xticks(rotation=45)
# plt.show()

# Analysis for report:
# Why this chart:
# -Bar chart is perfect for categorical data with multiple categories.
# -Shows which cuisines dominate the market.

# Insights:
# -Most popular cuisines are North Indian, Chinese, Fast Food, Italian.
# -Some cuisines are rare → niche market potential.

# Positive business impact:
# -Helps platform recommend popular cuisines to users.
# -Restaurants can expand offerings in trending cuisines.

# Negative growth insights:
# -Oversaturated cuisines (like North Indian) → high competition, lower growth for new entrants.

#Chart-3
# #Rating Outliers
# plt.figure(figsize=(6,4))
# plt.boxplot(restaurant['Rating'].dropna())
# plt.title("Box Plot of Ratings")
# plt.ylabel("Rating")
# plt.show()

# Analysis
# Why this chart?
# -Box plots help detect outliers and variability.

# Insights found:
# -Presence of low-rating outliers (<2.5).
# -Most ratings lie in a compact range.

# Positive business impact:
# -Helps platform identify consistently poor restaurants.

# Negative growth insights:
# -Low outliers indicate service or hygiene issues, hurting brand trust.

#Chart-4
#Restaurant by Cost Range
# restaurant['Cost_clean'] = restaurant['Cost'].str.replace('[^0-9]', '', regex=True)
# restaurant['Cost_clean'] = pd.to_numeric(restaurant['Cost_clean'], errors='coerce')

# bins = [0, 300, 600, 1000, 5000]
# labels = ['Low', 'Medium', 'High', 'Premium']
# restaurant['Cost_Category'] = pd.cut(restaurant['Cost_clean'], bins=bins, labels=labels)

# cost_counts = restaurant['Cost_Category'].value_counts()

# plt.figure(figsize=(7,5))
# cost_counts.plot(kind='bar')
# plt.title("Restaurants by Cost Category")
# plt.xlabel("Cost Category")
# plt.ylabel("Number of Restaurants")
# plt.show()

# Analysis:
# Why this chart?
# Bar chart clearly compares categories like price ranges.

# Insights found:
# -Most restaurants fall into Medium and High price ranges.
# -Premium restaurants are fewer.

# Positive business impact:
# -Platform can target middle-class users more effectively.

# Negative growth insights:
# -Fewer low-cost options may limit reach to budget-sensitive customers.

#Chart-5
#Reviews Over Time
# restaurant['Time'] = pd.to_datetime(restaurant['Time'], errors='coerce')
# reviews_over_time = restaurant.groupby(restaurant['Time'].dt.date).size()

# plt.figure(figsize=(10,5))
# plt.plot(reviews_over_time)
# plt.title("Reviews Over Time")
# plt.xlabel("Date")
# plt.ylabel("Number of Reviews")
# plt.show()

# Analysis
# Why this chart?
# Line plot is perfect for time-based trends.

# Insights found:
# -Growth or decline in reviews over time.

# Positive business impact:
# -Helps measure platform engagement growth.

# Negative growth insights:
# -Decline indicates reduced user activity or competition pressure.


#Chart-6# Understanding Rating Trend Over Years (Top Restaurants)

# Ensure Time is datetime
restaurant['Time'] = pd.to_datetime(restaurant['Time'], errors='coerce')

# Create Year column
restaurant['Year'] = restaurant['Time'].dt.year
 
print(restaurant.columns)
# Top 5 restaurants by number of reviews
top_restaurants = restaurant['Restaurant'].value_counts().head(5)

plt.figure(figsize=(12,6))

for rest in top_restaurants.index:
    data = restaurant[restaurant['Restaurant'] == rest]

    yearly_avg = (
        data.groupby('Year')['Rating']
        .mean()
        .dropna()
    )

    if not yearly_avg.empty:
        plt.plot(
            yearly_avg.index.astype(int),
            yearly_avg.values,
            marker='o',
            linewidth=2,
            label=rest
        )

years = sorted(restaurant['Year'].dropna().astype(int).unique())
plt.xticks(years, rotation=45)

plt.title("Rating Trend Over Years for Top Restaurants")
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.legend(title="Restaurant", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

