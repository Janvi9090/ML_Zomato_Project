# =========================
# Zomato Restaurant Analysis
# =========================

# -------------------------
# 1. Libraries
# -------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, pearsonr
import re
from textblob import TextBlob

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# -------------------------
# 2. Load and Merge Datasets
# -------------------------
D1 = pd.read_csv('Zomato_Restaurant_names_and_Metadata.csv')
D2 = pd.read_csv('Zomato_Restaurant_reviews.csv')

merged_D = pd.merge(D1, D2, left_on='Name', right_on='Restaurant', how='inner')
merged_D.to_csv('Zomato_merged.csv', index=False)
print("Datasets merged and saved successfully!")

# -------------------------
# 3. Load Merged Data
# -------------------------
restaurant = pd.read_csv("Zomato_merged.csv")
pd.set_option('display.max_columns', None)

print(restaurant.head())
print("Column names: ", restaurant.columns)
print("Total count of rows and columns: ", restaurant.shape)

# -------------------------
# 4. Handle Duplicate Values
# -------------------------
print("Duplicate values: ", restaurant.duplicated().sum())
restaurant.drop_duplicates(inplace=True)
print("After removal of duplicates: ", restaurant.duplicated().sum())
print("Total count of rows and columns: ", restaurant.shape)

# -------------------------
# 5. Missing Values Analysis
# -------------------------
print("Missing values per column:\n", restaurant.isnull().sum())

missing_counts = restaurant.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]

# Pie charts for missing values
for col in missing_counts.index:
    missing = restaurant[col].isnull().sum()
    not_missing = restaurant[col].notnull().sum()
    plt.figure(figsize=(6,6))
    plt.pie([missing, not_missing],
            labels=['Missing', 'Available'],
            autopct='%1.1f%%',
            colors=['red','green'],
            startangle=90, shadow=True)
    plt.title(f"Missing Values in '{col}' Column")
    plt.show()

# -------------------------
# 6. Unique Values per Column
# -------------------------
for col in restaurant.columns:
    unique_values = restaurant[col].unique()
    print(f"Column: {col}")
    print(f"Number of unique values: {len(unique_values)}")
    print(f"Sample unique values: {unique_values[:10]}")
    print("-"*50)

# -------------------------
# 7. Data Cleaning / Wrangling
# -------------------------
# Drop unnecessary columns
if 'Collections' in restaurant.columns:
    restaurant = restaurant.drop(columns=['Collections'])

# Fill missing values
restaurant['Timings'] = restaurant['Timings'].fillna('Not Specified')
restaurant['Reviewer'] = restaurant['Reviewer'].fillna('Anonymous')
restaurant['Review'] = restaurant['Review'].fillna('No Review')
restaurant['Rating'] = restaurant['Rating'].fillna('Rating')
restaurant['Metadata'] = restaurant['Metadata'].fillna('No Metadata')
restaurant['Time'] = restaurant['Time'].fillna('Unknown')
restaurant['Restaurant'] = restaurant['Restaurant'].fillna('Unknown')
restaurant['Pictures'] = restaurant['Pictures'].fillna(0)

# Convert data types
restaurant['Rating'] = pd.to_numeric(restaurant['Rating'], errors='coerce')
restaurant['Pictures'] = restaurant['Pictures'].astype(int)

# Strip whitespace from strings
for col in restaurant.select_dtypes(include='object').columns:
    restaurant[col] = restaurant[col].str.strip()

restaurant.reset_index(drop=True, inplace=True)

print("Missing values after cleaning:\n", restaurant.isnull().sum())

# -------------------------
# 8. Data Visualization
# -------------------------

# 8.1 Distribution of Ratings
plt.figure(figsize=(8,5))
restaurant['Rating'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.show()

# 8.2 Top 10 Cuisines
plt.figure(figsize=(10,5))
restaurant['Cuisines'].value_counts().head(10).plot(kind='bar', color='orange')
plt.title("Top 10 Cuisines")
plt.xlabel("Cuisine")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.show()

# 8.3 Rating Outliers (Boxplot)
plt.figure(figsize=(6,4))
plt.boxplot(restaurant['Rating'].dropna())
plt.title("Box Plot of Ratings")
plt.ylabel("Rating")
plt.show()

# 8.4 Restaurants by Cost Category
restaurant['Cost_clean'] = pd.to_numeric(restaurant['Cost'].str.replace('[^0-9]', '', regex=True), errors='coerce')
bins = [0, 300, 600, 1000, 5000]
labels = ['Low', 'Medium', 'High', 'Premium']
restaurant['Cost_Category'] = pd.cut(restaurant['Cost_clean'], bins=bins, labels=labels)

plt.figure(figsize=(7,5))
restaurant['Cost_Category'].value_counts().plot(kind='bar', color='purple')
plt.title("Restaurants by Cost Category")
plt.xlabel("Cost Category")
plt.ylabel("Number of Restaurants")
plt.show()

# 8.5 Reviews Over Time
restaurant['Time'] = pd.to_datetime(restaurant['Time'], errors='coerce')
reviews_over_time = restaurant.groupby(restaurant['Time'].dt.date).size()

plt.figure(figsize=(10,5))
plt.plot(reviews_over_time)
plt.title("Reviews Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Reviews")
plt.show()

# -------------------------
# 9. Hypothesis Testing
# -------------------------

# 9.1 Hypothesis 1: Cost vs Rating
anova_data = restaurant[['Cost_Category', 'Rating']].dropna()
groups = [anova_data[anova_data['Cost_Category'] == cat]['Rating'] for cat in anova_data['Cost_Category'].unique()]
f_stat, p_value = f_oneway(*groups)
print("H1 - Cost vs Rating | F-statistic:", f_stat, "P-value:", p_value)

# 9.2 Hypothesis 2: Pictures vs Rating
corr_data = restaurant[['Pictures', 'Rating']].dropna()
corr, p_value = pearsonr(corr_data['Pictures'], corr_data['Rating'])
print("H2 - Pictures vs Rating | Correlation:", corr, "P-value:", p_value)

# 9.3 Hypothesis 3: Cuisine vs Rating
top_cuisines = restaurant['Cuisines'].value_counts().head(5).index
cuisines_data = restaurant[restaurant['Cuisines'].isin(top_cuisines)]
anova_data = cuisines_data[['Cuisines', 'Rating']].dropna()
groups = [anova_data[anova_data['Cuisines'] == c]['Rating'] for c in top_cuisines]
f_stat, p_value = f_oneway(*groups)
print("H3 - Cuisine vs Rating | F-Statistic:", f_stat, "P-Value:", p_value)

# -------------------------
# 10. Sentiment Analysis
# -------------------------
# Clean Review Text
def clean_review(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

restaurant = restaurant.dropna(subset=['Review'])
restaurant['Cleaned_Review'] = restaurant['Review'].apply(clean_review)

# Sentiment Polarity
restaurant['Sentiment'] = restaurant['Cleaned_Review'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Sentiment Labels
def sentiment_label(score):
    if score > 0.1: return 'Positive'
    elif score < -0.1: return 'Negative'
    else: return 'Neutral'

restaurant['Sentiment_Label'] = restaurant['Sentiment'].apply(sentiment_label)

# Sentiment Distribution
plt.figure(figsize=(6,6))
restaurant['Sentiment_Label'].value_counts().plot(
    kind='pie', labels=restaurant['Sentiment_Label'].value_counts().index,
    autopct='%1.1f%%', startangle=90
)
plt.title("Distribution of Review Sentiment")
plt.show()

# Hypothesis 4: Sentiment vs Rating
restaurant['Rating_Round'] = restaurant['Rating'].round()
groups = [restaurant[restaurant['Rating_Round'] == r]['Sentiment'] for r in restaurant['Rating_Round'].dropna().unique()]
f_stat, p_value = f_oneway(*groups)
print("H4 - Sentiment vs Rating | F-Statistic:", f_stat, "P-value:", p_value)

# Hypothesis 5: Sentiment vs Cuisine
anova_data = restaurant[restaurant['Cuisines'].isin(top_cuisines)]
groups = [anova_data[anova_data['Cuisines'] == c]['Sentiment'] for c in top_cuisines]
f_stat, p_value = f_oneway(*groups)
print("H5 - Sentiment vs Cuisine | F-Statistic:", f_stat, "P-value:", p_value)

# -------------------------
# 11. Feature Engineering and ML Model
# -------------------------
# Rating Categories for Classification
restaurant['Rating_Category'] = pd.cut(
    restaurant['Rating'],
    bins=[0, 2.5, 3.5, 5],
    labels=['Low', 'Medium', 'High']
)

restaurant = restaurant.dropna(subset=['Rating_Category', 'Cleaned_Review'])
# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3)
X = tfidf.fit_transform(restaurant['Cleaned_Review'])
y = restaurant['Rating_Category']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE Oversampling
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
lr.fit(X_train_res, y_train_res)

# Predictions and Evaluation
y_pred = lr.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
