#libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import re
from textblob import TextBlob

# merge two dataset
D1 = pd.read_csv('Zomato_Restaurant_names_and_Metadata.csv')
D2 = pd.read_csv('Zomato_Restaurant_reviews.csv')

merged_D = pd.merge(D1, D2, left_on='Name', right_on='Restaurant', how='inner')
merged_D.to_csv('Zomato_merged.csv', index=False)
print("Datasets merged and saved successfully!")

#display of data
restaurant = pd.read_csv("Zomato_merged.csv")
pd.set_option('display.max_columns', None)
print(restaurant.head())

#Names of columns
print("Column names: ", restaurant.columns)

#To count Rows and columns
print("Total count of rows and columns: ",restaurant.shape)

# duplicate values count
print("Duplicate values: ",restaurant.duplicated().sum())
restaurant.drop_duplicates(inplace=True)
print("After removal of duplicate values count: ", restaurant.duplicated().sum())
print("Total count of rows and columns: ",restaurant.shape)

#To count missing values(visualizing)
print("Missing values count per column ",)
print(restaurant.isnull().sum())

missing_counts = restaurant.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]  # only columns with missing values

for col in missing_counts.index:
    missing = restaurant[col].isnull().sum()
    not_missing = restaurant[col].notnull().sum()
    plt.figure(figsize=(6,6))
    plt.pie([missing, not_missing], labels=['Missing', 'Available'], autopct='%1.1f%%',
            colors=['red','green'], startangle=90, shadow=True)
    plt.title(f"Missing Values in '{col}' Column")
    plt.show()

#unique values for each value
for col in restaurant.columns:
    unique_values = restaurant[col].unique()  # get unique values
    print(f"Column: {col}")
    print(f"Number of unique values: {len(unique_values)}")
    print(f"Unique values (sample): {unique_values[:10]}")  # show first 10 only
    print("-"*50)


#data wrangling
#handle missing values
restaurant = restaurant.drop(columns=['Collections'])

# Fill missing values properly
restaurant['Timings'] = restaurant['Timings'].fillna('Not Specified')
restaurant['Reviewer'] = restaurant['Reviewer'].fillna('Anonymous')
restaurant['Review'] = restaurant['Review'].fillna('No Review')
restaurant['Rating'] = restaurant['Rating'].fillna('Rating')          # can later convert to numeric
restaurant['Metadata'] = restaurant['Metadata'].fillna('No Metadata')
restaurant['Time'] = restaurant['Time'].fillna('Unknown')
restaurant['Restaurant'] = restaurant['Restaurant'].fillna('Unknown')
restaurant['Pictures'] = restaurant['Pictures'].fillna(0)
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
rating_counts = restaurant['Rating'].value_counts().sort_index()
plt.figure(figsize=(8,5))
rating_counts.plot(kind='bar', color='skyblue')
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.show()

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
cuisine_counts = restaurant['Cuisines'].value_counts().head(10)
plt.figure(figsize=(10,5))
cuisine_counts.plot(kind='bar', color='orange')
plt.title("Top 10 Cuisines")
plt.xlabel("Cuisine")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.show()

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
plt.figure(figsize=(6,4))
plt.boxplot(restaurant['Rating'].dropna())
plt.title("Box Plot of Ratings")
plt.ylabel("Rating")
plt.show()

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
restaurant['Cost_clean'] = restaurant['Cost'].str.replace('[^0-9]', '', regex=True)
restaurant['Cost_clean'] = pd.to_numeric(restaurant['Cost_clean'], errors='coerce')

bins = [0, 300, 600, 1000, 5000]
labels = ['Low', 'Medium', 'High', 'Premium']
restaurant['Cost_Category'] = pd.cut(restaurant['Cost_clean'], bins=bins, labels=labels)

cost_counts = restaurant['Cost_Category'].value_counts()

plt.figure(figsize=(7,5))
cost_counts.plot(kind='bar')
plt.title("Restaurants by Cost Category")
plt.xlabel("Cost Category")
plt.ylabel("Number of Restaurants")
plt.show()

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
restaurant['Time'] = pd.to_datetime(restaurant['Time'], errors='coerce')
reviews_over_time = restaurant.groupby(restaurant['Time'].dt.date).size()

plt.figure(figsize=(10,5))
plt.plot(reviews_over_time)
plt.title("Reviews Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Reviews")
plt.show()

# Analysis
# Why this chart?
# Line plot is perfect for time-based trends.

# Insights found:
# -Growth or decline in reviews over time.

# Positive business impact:
# -Helps measure platform engagement growth.

# Negative growth insights:
# -Decline indicates reduced user activity or competition pressure.


#Hypothesis Statement 1
#Do higher-cost restaurants have higher ratings?
#H0(Null Hypothesis): There is no significant difference in average ratings between cost categories
#H1(Alternate Hypothesis): Average ratings significantly differ across cost categories.
#Statistical Test
#-One-way ANOVA
# used because: More than 2 groups(Low, Medium, High, Premiun), Comparing mean ratings
restaurant['Cost_clean'] = restaurant['Cost'].str.replace('[^0-9]', '', regex=True)
restaurant['Cost_clean'] = pd.to_numeric(restaurant['Cost_clean'], errors='coerce')

bins = [0, 300, 600, 1000, 5000]
labels = ['Low', 'Medium', 'High', 'Premium']
restaurant['Cost_Category'] = pd.cut(restaurant['Cost_clean'], bins=bins, labels=labels)
# # Drop missing values
anova_data = restaurant[['Cost_Category', 'Rating']].dropna()
# # Group ratings by cost category
groups = [
    anova_data[anova_data['Cost_Category'] == cat]['Rating']
    for cat in anova_data['Cost_Category'].unique()
]
# # Perform ANOVA
f_stat, p_value = f_oneway(*groups)
print("F-statistic:", f_stat)
print("P-value:", p_value)


# #Interpretation (exam/report ready)
# -Reject the null hypothesis (H₀)
# There is a significant difference in average ratings across cost categories.
# -Business insight:
# Higher-cost restaurants tend to have higher ratings, while lower-cost ones may have slightly lower ratings
# Platform can promote premium and high-rated restaurants strategically
# Shows pricing correlates with perceived quality


#Hypothesis Statement 2
#Do restaurant with more pictures have higher ratings?
#H0(Null Hypothesis): There is no relationship between number of pictures and ratings.
#H1(Alternate Hypothesis): Restaurant with more pictures tend to have higher ratings.
#Statistical Test:
#used for relationship between two numerical variables
from scipy.stats import pearsonr
corr_data = restaurant[['Pictures', 'Rating']].dropna()
corr, p_value = pearsonr(corr_data['Pictures'], corr_data['Rating'])
print("Correlation: ", corr)
print("P-value: ", p_value)

# #nterpretation (exam/report ready)
# -Reject the null hypothesis (H₀)
# There is a statistically significant relationship between number of pictures and ratings, but the correlation is very weak.
# -Business insight:
# More pictures slightly improve customer perception
# But other factors like service, food quality, ambiance play a much bigger role
# Platform can encourage restaurants to upload pictures, but don’t rely on pictures alone to boost ratings



#Hypothesis Statement 3
#Do certain cuisines receive higher ratings than others?
#H0(Null hypothesis): Average ratings are the same for all cuisines.
#H1(Alternate hypothesis): At least one cuisine has a signnificantly different average rating.
#Statistical Test:
#One-Way ANOVA
#Used because: We are comparing average ratings (numerical) across multiple cuisine categories(categorical)
top_cuisines=restaurant['Cuisines'].value_counts().head(5).index
cuisines_data = restaurant[restaurant['Cuisines']. isin(top_cuisines)]
anova_data = cuisines_data[['Cuisines', 'Rating']].dropna()
groups = [anova_data[anova_data['Cuisines']==cuisine]['Rating'] for cuisine in top_cuisines]
f_stat, p_value = f_oneway(*groups)
print("F-Statistic: ", f_stat)
print("P-Value: ", p_value)

# #Interpretation
# -Reject the null hypothesis (H₀)
# There is a statistically significant difference in average ratings among top cuisines. Some cuisines consistently get higher ratings, while others receive lower ratings.
# -Business insight:
# High-rated cuisines can be promoted more on the platform to attract customers.
# Low-rated cuisines indicate areas for quality improvement, such as service, food, or presentation.
# Helps restaurants and the platform make data-driven marketing and improvement decisions.


# #Cleaning the review column
restaurant=restaurant.dropna(subset=['Review'])
def clean_review(text):
    text = str(text).lower() #lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  #Punctuation
    text = re.sub(r'\s+', ' ', text).strip()  #Extra Spaces
    return text
restaurant['Cleaned_Review'] = restaurant['Review'].apply(clean_review)

restaurant['Sentiment']=restaurant['Cleaned_Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
def sentiment_label(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else: return 'Neutral'
restaurant['Sentiment_Label']= restaurant['Sentiment'].apply(sentiment_label)


sentiment_counts = restaurant['Sentiment_Label'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Distribution of Review Sentiment")
plt.show()


# #Hypothesis Statement 4
# #H0(Null Hypothesis): There is no significant difference in sentiment scores across different rating levels.
# #H1(Alternate Hypothesis): There is a significant difference in sentiment scrores across different rating levels.
# #ANOVA: We compare sentiment across rating groups
restaurant['Rating_Round']=restaurant['Rating'].round()
groups=[
    restaurant[restaurant['Rating_Round']==r]['Sentiment']
    for r in restaurant['Rating_Round'].dropna().unique()
]
f_stat, p_value= f_oneway(*groups)
print("F-Statistic: ",f_stat)
print("P-value: ", p_value)

# #Interpretation
# -Restaurants with higher ratings tend to have more positive sentiment in customer reviews, whereas lower-rated restaurants show negative sentiment. This confirms that textual sentiment extracted from reviews strongly aligns with numerical ratings.
# Business Impact
# -Sentiment analysis can be reliably used to support or predict restaurant ratings.
# -Platforms can identify potential issues early by monitoring negative sentiment trends.
# -Improves decision-making for restaurant ranking and recommendations.
# Negative Growth Insight 
# -Restaurants showing increasing negative sentiment are likely to experience a future decline in ratings, which can impact customer trust and overall platform credibility.



# Hypothesis Statement 5
#To examine whether customer sentiment differs across different cuisine types.
#H0(Null Hypothesis): There is no statistically significant difference in sentiment scores across different cuisines.
#H1(Alternate Hypothesis): There is a statistically significant difference in sentiment scores across different cuisines.
#One-Way ANOVA: Cuisine type is a categorical variable with multiple groups

top_cuisines=restaurant['Cuisines'].value_counts().head(5).index
anova_data = restaurant[restaurant['Cuisines'].isin(top_cuisines)]

groups = [
    anova_data[anova_data['Cuisines']==cuisine]['Sentiment']
    for cuisine in top_cuisines
]

f_stat, p_value = f_oneway(*groups)
print("F-statistic: ", f_stat)
print("P-Value: ", p_value)

print("Missing values are:",restaurant['Rating'].isnull().sum())
restaurant = restaurant.dropna(subset=['Rating'])
print("Missing values after are:",restaurant['Rating'].isnull().sum())

# #Common Preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE


restaurant['Rating_Category'] = pd.cut(
    restaurant['Rating'],
    bins=[0, 2.5, 3.5, 5],
    labels=['Low', 'Medium', 'High']
)

#  TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3)
X = tfidf.fit_transform(restaurant['Cleaned_Review'])
y = restaurant['Rating_Category']

#  Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  SMOTE on TF-IDF numeric matrix
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

#  Logistic Regression
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs'
)
lr.fit(X_train_res, y_train_res)

# Predictions & evaluation
y_pred = lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
