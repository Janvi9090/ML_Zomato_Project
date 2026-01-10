# Zomato Restaurant Analysis Project

## Project Overview

This project analyzes Zomato restaurant data, including restaurant metadata and customer reviews. The goal is to perform Exploratory Data Analysis (EDA), Sentiment Analysis, Hypothesis Testing, and Machine Learning to classify restaurant ratings.

## Dataset

* `Zomato_Restaurant_names_and_Metadata.csv`: Contains restaurant names, metadata, cost, cuisines, timings, and pictures.
* `Zomato_Restaurant_reviews.csv`: Contains customer reviews, reviewer names, ratings, and timestamps.
* `Zomato_merged.csv`: Merged dataset combining metadata and reviews.

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

2. **Exploratory Data Analysis (EDA)**

   * Distribution of Ratings.
   * Top 10 Cuisines.
   * Rating outliers (boxplots).
   * Restaurants by cost category.
   * Reviews over time.

3. **Hypothesis Testing**

   * **H1:** Cost vs Rating (ANOVA).
   * **H2:** Pictures vs Rating (Pearson Correlation).
   * **H3:** Cuisine vs Rating (Top 5 cuisines, ANOVA).
   * **H4:** Sentiment vs Rating (ANOVA).
   * **H5:** Sentiment vs Cuisine (ANOVA).

4. **Sentiment Analysis**

   * Clean and preprocess review text.
   * Calculate sentiment polarity using TextBlob.
   * Classify sentiment as Positive, Negative, or Neutral.

5. **Machine Learning Model**

   * Create rating categories: Low, Medium, High.
   * TF-IDF vectorization of review text.
   * Train/Test split with stratification.
   * SMOTE oversampling for imbalanced classes.
   * Train Logistic Regression classifier.
   * Evaluate with accuracy and classification report.

## Visualizations

* Bar charts for rating distribution and cuisines.
* Box plots for rating outliers.
* Line charts for reviews over time.
* Pie charts for sentiment distribution.
* Pie charts for missing value analysis.

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

## Notes

* Replace placeholders in the code with actual screenshots if needed.
* Ensure missing values are properly handled to avoid errors in ML steps.
* Customize visualizations and analysis based on project requirements.
