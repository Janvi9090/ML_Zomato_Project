# Zomato Restaurant Analysis Project

# Github Link- https://github.com/Janvi9090/ML_Zomato_Project

## Project Overview

This project analyzes Zomato restaurant data, including restaurant metadata and customer reviews. The goal is to perform Exploratory Data Analysis (EDA), Sentiment Analysis, Hypothesis Testing, and Machine Learning to classify restaurant ratings.

## Problem Statement

Online food platforms like Zomato generate a large volume of restaurant metadata and customer reviews. However, this data is raw, unstructured, and contains missing values, duplicates, and noise, making it difficult to directly extract meaningful insights.
The challenge is to clean, integrate, and analyze Zomato restaurant metadata and customer reviews to:

1. understand factors influencing restaurant ratings,

2. analyze customer sentiment from textual reviews,

3. identify relationships between cost, cuisine, sentiment, and ratings, and

4. build a machine learning model that can predict restaurant rating categories (Low, Medium, High) based on review text.

This project addresses the problem by applying data preprocessing, exploratory data analysis, statistical hypothesis testing, sentiment analysis, and supervised machine learning to transform raw Zomato data into actionable insights and predictive outcomes.

## Dataset

* `Zomato_Restaurant_names_and_Metadata.csv`: Contains restaurant names, metadata, cost, cuisines, timings, and pictures.
* `Zomato_Restaurant_reviews.csv`: Contains customer reviews, reviewer names, ratings, and timestamps.
* `Zomato_merged.csv`: Merged dataset combining metadata and reviews.

## What did you know about your dataset?
After merging and preprocessing the Zomato datasets, the final dataset provided the following insights:
1. The dataset represents restaurant-level information combined with customer review data, making it suitable for both numerical and text-based analysis.
2. It contains multiple feature types:
 - Numerical features: Rating, Pictures, Cost_clean
 - Categorical features: Cuisines, Cost_Category, Timings
 - Text data: Review (used for sentiment analysis and ML modeling)
 - Time-based data: Time (used to analyze review trends over time)
3. Duplicate records were present initially and were successfully removed to ensure data integrity.
4. Several columns contained missing values, which were handled using appropriate strategies:
 - Text fields were filled with meaningful placeholders (e.g., Anonymous, No Review).
 - Numerical fields were converted and cleaned, with invalid values coerced to NaN and removed where necessary.
 5. The Rating column was converted to numeric and further transformed into rating categories (Low, Medium, High) for classification tasks.
 6. A cleaned and processed review column (Cleaned_Review) was created, enabling sentiment analysis using TextBlob.
 7. Additional engineered features such as Cost_Category, Rating_Round, and Sentiment_Label enriched the dataset for deeper analysis.
 8. After preprocessing, the dataset became consistent, structured, and analysis-ready, supporting:
 - Exploratory Data Analysis (EDA)
 - Hypothesis testing (ANOVA and correlation)
 - Sentiment analysis
 - Machine learning classification using TF-IDF and Logistic Regression

Overall, the final merged dataset is well-structured, rich in information, and suitable for deriving actionable business insights from customer behavior and restaurant performance.

### Columns

* **Restaurant**: Name of the restaurant.
* **Reviewer**: Name of the person reviewing.
* **Review**: Text of the review.
* **Rating**: Numerical rating given by the reviewer.
* **Metadata**: Additional information about the restaurant.
* **Time**: Timestamp of the review.
* **Pictures**: Number of pictures uploaded for the restaurant.
* **Cost**: Cost information of the restaurant.
* **Cuisines**: Type(s) of cuisines offered.
* **Timings**: Operational timings of the restaurant.

## Key Steps

1. **Data Cleaning and Wrangling**

   * Remove duplicates.
   * Fill missing values with appropriate defaults.
   * Convert data types (`Rating` to numeric, `Pictures` to integer).
   * Clean `Cost` and `Review` columns.

- What all manipulations have you done and insights you found?
 **Data Manipulations**
* Merged restaurant metadata and review datasets using restaurant name.
* Removed duplicate records and handled missing values with meaningful defaults.
* Converted Rating to numeric and categorized it into Low, Medium, High.
* Cleaned the Cost column and created cost categories (Low, Medium, High, Premium).
* Preprocessed review text and generated sentiment scores and labels.
* Engineered additional features such as cost category and sentiment-based variables.
* Applied TF-IDF vectorization and used SMOTE to handle class imbalance before modeling.
 **Key Insights**
* Most restaurants have ratings between 3.5 and 4.5.
* Higher-cost restaurants generally receive higher average ratings.
* Restaurants with more pictures show a slight positive correlation with ratings.
* Sentiment polarity strongly aligns with numerical ratings.
* Review text is an effective predictor of restaurant rating categories.

- What all missing value imputation techniques have you used and why did you use those techniques?
 **Techniques used:**
 * Filling missing values with default values:
 * Timings → 'Not Specified'
 * Reviewer → 'Anonymous'
 * Review → 'No Review'
 * Rating → temporarily 'Rating' (later converted to numeric with pd.to_numeric)
 * Metadata → 'No Metadata'
 * Time → 'Unknown'
 * Restaurant → 'Unknown'
 * Pictures → 0
 **Why**
 * These are simple imputation methods for textual/categorical data to avoid NaN issues in analysis and ML. Ensures no missing values remain that could cause errors in visualizations, statistical tests, or ML models.

- What all outlier treatment techniques have you used and why did you use those techniques?
 **Techniques used:**
 * Box plot visualization to detect rating outliers.
 * No explicit removal was applied in your code for ratings or cost, but you converted cost to numeric and categorized it, which indirectly reduces the impact of extreme values.
 **Why**
  * Boxplots provide a visual understanding of extreme values. Categorizing Cost into bins (Low, Medium, High, Premium) helps limit the influence of extreme cost values on analysis and ML.

- What all categorical encoding techniques have you used & why did you use those techniques?
 **Techniques used:**
 * Label encoding via pd.cut for numeric ranges:
   Rating_Category → Low, Medium, High
   Cost_Category → Low, Medium, High, Premium
 * TF-IDF vectorization for text (Cleaned_Review):
   Converts textual reviews into numeric features for ML.
 **Why**
 * pd.cut converts continuous or numeric data into categories for classification. TF-IDF captures textual information in a numeric form suitable for ML algorithms like Logistic Regression.


2. **Exploratory Data Analysis (EDA)**

   * Distribution of Ratings.
   * Top 10 Cuisines.
   * Rating outliers (boxplots).
   * Restaurants by cost category.
   * Reviews over time.

* **Visualizations**

1. Bar charts for rating distribution.
- Why this chart was chosen
A bar chart clearly shows the number of reviews for each rating. It’s easy to compare categories and spot which ratings are most common.
- Insights from the graph 
Most restaurants have ratings of 4–5 stars. Very few restaurants have ratings below 2.
- Will the gained insights help creating a positive business impact?
Yes, high-rated restaurants can be promoted to attract more customers and build trust.
- Are there any insights that lead to negative growth? Justify.
Yes, restaurants with 1–2 star ratings indicate poor service or quality. These need improvement to prevent losing customers.

2. Bar Chart for Top Cuisines
- Why this chart was chosen
A bar chart is ideal for comparing the number of restaurants across different cuisines. It clearly shows which cuisines are most popular.
- Insights from the graph 
North Indian and Chinese cuisines dominate the market, followed by Continental, Biryani, and Desserts. Other cuisines have fewer restaurants.
- Will the gained insights help creating a positive business impact?
Yes, restaurants can focus on high-demand cuisines to attract more customers and plan marketing strategies.
- Are there any insights that lead to negative growth? Justify.
Yes, cuisines with very few restaurants may struggle with visibility and revenue, indicating areas of lower demand or competition.

3. Box plots for rating outliers.
- Why this chart was chosen
A box plot helps identify outliers and the spread of ratings. It shows the median, quartiles, and extreme values clearly.
- Insights from the graph 
Most ratings are clustered between 3 and 5, with very few extremely low ratings around 1–2. This indicates generally positive reviews.
- Will the gained insights help creating a positive business impact?
Yes, businesses can focus on maintaining high-quality services since most restaurants already receive good ratings.
- Are there any insights that lead to negative growth? Justify.
Yes, the few low-rating outliers indicate some restaurants may have quality or service issues, which could affect reputation if not addressed.

4. Bar Chart for Restaurant by Cost Category
- Why this chart was chosen
A bar chart clearly shows the number of restaurants in each cost category. It is easy to compare categories at a glance.
- Insights from the graph 
Most restaurants fall under Medium, Premium, and High categories, while Low-cost restaurants are fewer. This shows the market is skewed toward mid-to-high-end dining.
- Will the gained insights help creating a positive business impact?
Yes, businesses can target promotions for Medium and High-cost segments, which dominate the market.
- Are there any insights that lead to negative growth? Justify.
Yes, fewer Low-cost options indicate limited affordable choices, which might reduce customer reach among budget-conscious diners.

5. Line charts for reviews over time.
- Why this chart was chosen
A line chart shows trends over time clearly. It helps visualize how review counts change across dates.
- Insights from the graph 
Reviews increased sharply after mid-2018, indicating growing customer engagement and popularity of restaurants over time.
- Will the gained insights help creating a positive business impact?
Yes, businesses can identify peak periods and plan marketing campaigns or promotions during high engagement months.
- Are there any insights that lead to negative growth? Justify.
Periods with very few reviews suggest low customer interaction in early years, which could indicate underperformance or lack of awareness.

4. Pie charts for sentiment distribution.
- Why this chart was chosen
A pie chart shows proportions of positive, neutral, and negative reviews clearly. It’s ideal for understanding sentiment distribution.
- Insights from the graph 
Most reviews (68%) are positive, 17% neutral, and 15% negative, indicating overall customer satisfaction.
- Will the gained insights help creating a positive business impact?
Yes, businesses can highlight positive sentiment in marketing and identify areas of improvement from negative feedback.
- Are there any insights that lead to negative growth? Justify.
The 15% negative reviews indicate some dissatisfaction; addressing these issues can prevent loss of customers and improve reputation.

5. Pie charts for missing value analysis.
- Why this chart was chosen
Pie charts clearly show the proportion of missing vs available values for each column, making it easy to identify data quality issues.
- Insights from the graph 
Some columns, like Reviewer, Review, and Rating, have a small percentage of missing values, while others are complete. This helps prioritize cleaning steps.
- Will the gained insights help creating a positive business impact?
Yes, ensuring missing values are handled properly improves the reliability of analysis and ML models, leading to better business decisions.
- Are there any insights that lead to negative growth? Justify.
Even a small amount of missing critical data (like ratings or reviews) can bias insights and predictions, potentially affecting marketing and service improvements.


3. **Hypothesis Testing**

   * **H1:** Cost vs Rating (ANOVA).
   Hypothesis- Do higher-cost restaurants have higher ratings?
   -H0(Null Hypothesis): There is no significant difference in average ratings between cost categories
   -H1(Alternate Hypothesis): Average ratings significantly differ across cost categories.
   -Statistical test used to obtain P-Value: One-way ANOVA (Analysis of Variance)
   -Why this test was chosen: ANOVA is suitable because we are comparing the mean ratings across more than two independent groups (Cost Categories: Low, Medium, High, Premium) to see if at least one group differs significantly. It checks if cost impacts the ratings given by customers.

   * **H2:** Pictures vs Rating (Pearson Correlation).
   Hypothesis- Do restaurant with more pictures have higher ratings?
   -H0(Null Hypothesis): There is no relationship between number of pictures and ratings
   -H1(Alternate Hypothesis): Restaurant with more pictures tend to have higher ratings.
   - Statistical test used to obtain P-Value: Pearson Correlation
   - Why this test was chosen: Because both Pictures and Rating are numeric variables, and we wanted to check if there’s a linear relationship between them.

   * **H3:** Cuisine vs Rating (Top 5 cuisines, ANOVA).
   Hypothesis- Do certain cuisines receive higher ratings than others?
   -H0(Null hypothesis): Average ratings are the same for all cuisines.
   -H1(Alternate hypothesis): At least one cuisine has a signnificantly different average rating.
   - Statistical Test Used to Obtain P-Value: One-way ANOVA (Analysis of Variance)
   - Why this test was chosen: Because Cuisine is a categorical variable (Top 5 cuisines) and Rating is numeric. ANOVA helps check if the average ratings differ significantly across different cuisines.

   * **H4:** Sentiment vs Rating (ANOVA).
   -H0(Null hypothesis): The mean sentiment scores are the same across all rounded rating categories.
   -H1(Alternate hypothesis): At least one rating category has a different mean sentiment score.
   - Statistical Test Used to Obtain P-Value: One-way ANOVA (f_oneway)
   - Why this test was chosen: Because Rating_Round is categorical and Sentiment is numeric. ANOVA checks if sentiment varies significantly across different rating groups.

   * **H5:** Sentiment vs Cuisine (ANOVA).
   -H0(Null Hypothesis): The mean sentiment scores are the same across the top 5 cuisines.
   -H1(Alternate Hypothesis): At least one cuisine has a different mean sentiment score.
   - Statistical Test Used to Obtain P-Value: One-way ANOVA (f_oneway)
   - Why this test was chosen: Cuisine is a categorical variable and sentiment is numeric. ANOVA checks if sentiment significantly differs across the selected cuisine groups.

4. **Sentiment Analysis**

   * Clean and preprocess review text.
   * Calculate sentiment polarity using TextBlob.
   * Classify sentiment as Positive, Negative, or Neutral.

 **Low Casing**
 * Converted all review text to lowercase to standardize words for analysis.
 **Removing Punctuations**
 * Removed punctuation marks to reduce noise in text for sentiment analysis and TF-IDF vectorization.
 **Removing White Spaces**
 * Extra spaces were removed using regex to clean the text.
 **Text Vectorization**
 * Used TF-IDF (Term Frequency-Inverse Document Frequency) to convert cleaned review text into numeric vectors. Chosen because it captures important words and their importance while reducing noise from frequent words.
 **Feature Selection**
 1. What all feature selection methods have you used and why?
   Manual Selection: We selected features based on relevance to the task rather than applying automated methods.
   * For ML model: Cleaned_Review as predictor, Rating_Category as target.
   * For analysis & hypothesis testing: Rating, Cost_Category, Cuisines, Pictures, Sentiment.
   Reason: This avoids overfitting, reduces noise from irrelevant columns like Metadata or Timings, and ensures that the selected features directly contribute to predictive performance and meaningful insights.

 2. Which all features you found important and why?
   * Important Features: Cleaned_Review, Rating, Cost_Category, Cuisines, Pictures, Sentiment_Label
   * Reason: They directly influence ratings or help analyze customer sentiment and restaurant performance; other columns were less relevant for modeling.
 **Data Transformation**
 1. Do you think that your data needs to be transformed? If yes, which transformation have you used. Explain Why?
   * Yes, the `Rating` column was transformed into categorical values (`Low`, `Medium`, `High`) to simplify classification. This helps the ML model predict rating categories instead of raw numeric ratings, making the task more meaningful for business insights.
 **Data Splitting**
 1. What data splitting ratio have you used and why?
   * I used an 80:20 train-test split. This ensures enough data for training the model while keeping a sufficient portion for testing to evaluate performance accurately.
 **Handling Imbalanced Dataset**
 1. Do you think the dataset is imbalanced? Explain Why.
   * Yes, the dataset is imbalanced because the number of reviews in each rating category (Low, Medium, High) is not equal—most reviews are in the Medium/High category, while Low ratings are fewer.
 2. What technique did you use to handle the imbalance dataset and why? (If needed to be balanced)
   * To handle this, I used SMOTE (Synthetic Minority Oversampling Technique) to oversample the minority classes in the training data. This prevents the model from being biased toward the majority class and improves classification performance.

5. **Machine Learning Model**

   * Create rating categories: Low, Medium, High.
   * TF-IDF vectorization of review text.
   * Train/Test split with stratification.
   * SMOTE oversampling for imbalanced classes.
   * Train Logistic Regression classifier.
   * Evaluate with accuracy and classification report.

* Hyperparameter Optimization:
  Used class_weight='balanced' in Logistic Regression to handle class imbalance. This improved recall and F1-score for minority classes, with overall accuracy ~79% and weighted F1-score 0.81.
1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.
 * ML Model Used:
 A logistic Regression to classify restaurant ratings (Low, Medium, High) from review text using TF-IDF features. SMOTE handled class imbalance, and data was split 80/20 for training/testing.
 * Performance: 
 Overall accuracy ~79%. High and Low ratings are predicted  well, while Medium ratings have lower F1-score due to fewer samples. Weighted F1-score is 0.81, showing good overall model reliability.

2. Which Evaluation metrics did you consider for a positive business impact and why?
 * Accuracy, Precision, Recall, F1-Score were considered.
 * Reason: Precision and recall help identify how well high/low-rated restaurants are classified, which is critical for business decisions like promotions or improvements. F1-score balances both, ensuring reliable predictions.
 
3. Which ML model did you choose from the above created models as your final prediction model and why?
 * Chosen ML Model: Logistic Regression with TF-IDF features.
 * Reason: It performed best on the textual review data, handled class imbalance with SMOTE, gave good accuracy (~79%), and is interpretable, showing which words influence restaurant rating predictions.

4. Explain the model which you have used and the feature importance using any model explainability tool?
 * Model Used: Logistic Regression with TF-IDF vectorized review text.
 * Explanation & Feature Importance:
      Logistic Regression predicts the rating category (Low, Medium, High) based on the words in reviews.
      TF-IDF captures the importance of each word relative to the dataset, giving higher weight to words that are frequent in a review but not common across all reviews.
      Feature importance can be interpreted via the coefficients of the Logistic Regression model: words with higher positive coefficients push predictions toward higher ratings, while words with negative coefficients push predictions toward lower ratings.
      Example: Words like "excellent," "delicious," "friendly" contribute to High ratings, whereas words like "bad," "slow," "bland" contribute to Low ratings.
 
## Insights & Business Impact

* High-rated restaurants can be promoted; low-rated flagged for improvement.
* Popular cuisines indicate market trends and opportunities.
* Cost categories help target marketing strategies.
* Sentiment analysis aligns with ratings, helping to monitor customer satisfaction.
* ML model can predict rating categories from review text.

## Requirements

* Python 3.x
* Libraries: pandas, matplotlib, seaborn, scipy, re, textblob, sklearn, imblearn

## Usage

1. Load datasets.
2. Merge metadata and review datasets.
3. Perform cleaning, EDA, and sentiment analysis.
4. Conduct hypothesis testing.
5. Train and evaluate ML model.

## Conclusion
This project cleaned and analyzed Zomato restaurant data to uncover insights from ratings, cuisines, costs, and customer reviews. Sentiment and hypothesis analyses highlighted factors affecting ratings, while the Logistic Regression ML model predicted rating categories with ~79% accuracy. Overall, the analysis provides actionable insights for improving service, targeting marketing, and enhancing customer satisfaction.

