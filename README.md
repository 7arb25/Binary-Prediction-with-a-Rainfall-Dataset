# Binary-Prediction-with-a-Rainfall-Dataset

---
![imge here]()
---

My Full Answer For Kaggle's PlayGround Comptiition 

[![Open in VS Code](images/vscode.png)](https://vscode.dev/github/7arb25/Binary-Prediction-with-a-Rainfall-Dataset)
---

**Table of Contents**

- [Project Overview](#project-overview)
- [Dataset Feature Descriptions](#dataset-feature-descriptions)
- [Tools](#tools)
- [EDA](#eda)
- [Data Modeling](#data-modeling)
- [Model Limitations](#model-limitations)
- [Results](#results)
- [Recommendations](#recommendations)
- [Installation](#installation)
- [License](#license)


## Project Overview

─── data/
|   |── plain/
│            ├── train.csv
│            └── test.csv
│            ├── train_temporal.csv
│            └── test_temporal.csv
|   |── preprocessed/
|            |─── train.csv
│            └── test.csv
|── submissions/
|            |─── submission_xgboost.csv
│            └── test.csv
├── notebooks/
│   |── EDA.ipynb
│   |── Feature_Engineering.ipynb
│   ├── xgboost.ipynb
│   ├── randomforest.ipynb
│   ├── sgd_classifier.ipynb
│   ├── logistic_reg.ipynb
│   └── sgd_classifier.ipynb
├── models/
│   ├── xgboost.ipynb
│   ├── randomforest.ipynb
│   ├── sgd_classifier.ipynb
│   ├── logistic_reg.ipynb
│   └── sgd_classifier.ipynb
├── README.md
─── requirements.txt


[![License](https://img.shields.io/badge/license-[YourLicense]-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-[VersionNumber]-green.svg)](CHANGELOG.md)

---

## Dataset Feature Descriptions

| Feature        | Description                                                                  | Data Type |
|----------------|------------------------------------------------------------------------------|-----------|
| `id`           | Unique identifier for each data point/observation.                           | Integer   |
| `day`          | The day of the observation.                                                  | Integer   |
| `pressure`     | Atmospheric pressure measured in hPa (hectopascals).                         | Float     |
| `maxtemp`      | Maximum temperature recorded during the day, in degrees Celsius.          | Float     |
| `temperature`  | Average temperature recorded during the day, in degrees Celsius.              | Float     |
| `mintemp`      | Minimum temperature recorded during the day, in degrees Celsius.          | Float     |
| `dewpoint`     | The dew point temperature, in degrees Celsius.                               | Float     |
| `humidity`     | Relative humidity, expressed as a percentage (%).                            | Float     |
| `cloud`        | Cloud cover, indicating the fraction of the sky obscured by clouds.         | Float     |
| `sunshine`     | Duration of sunshine, in hours.                                             | Float     |
| `winddirection`| Wind direction, likely in degrees (e.g., 0-360) or categorical (e.g., N, S).| Float/Categorical |
| `windspeed`    | Wind speed, likely in km/h or m/s.                                          | Float     |
| `rainfall`     | Binary target variable: 1 indicates rainfall, 0 indicates no rainfall.     | Integer (Binary) |

---

## EDA

### Histogram Analysis

This section provides insights into the distribution of numerical features in the dataset.

**Questions Addressed:**

1.  **What is the overall shape of each feature's distribution?** (e.g., normal, skewed, bimodal, uniform)
2.  **Are there any outliers or unusual patterns?**
3.  **Are the features roughly symmetrically distributed, or are they skewed?**
4.  **Are there multiple modes in the data?**
5.  **What is the range of values for each feature?**
6.  **Are there any features with a large number of zero values?**
7.  **How do the train and test distributions compare?** (If applicable)

**Histograms:**

[Histigram image here]

**Observations:**

* **pressure:** Roughly symmetrical, slightly right-skewed.
* **maxtemp:** Left-skewed, suggesting a higher concentration of warmer temperatures.
* **temparature:** Bimodal, with peaks around 15 and 25.
* **mintemp:** Left-skewed, similar to maxtemp.
* **dewpoint:** Right-skewed, with a concentration of lower dew points.
* **humidity:** Bimodal, with peaks around 70 and 90.
* **cloud:** Right-skewed, with a high concentration of low cloud cover.
* **sunshine:** Highly right-skewed, with a significant number of days with low sunshine.
* **winddirection:** Bimodal, with peaks around 0-50 and 200-300.
* **windspeed:** Right-skewed, with a concentration of lower wind speeds.
* **rainfall:** Extremely right-skewed, with a very high concentration of zero or near-zero rainfall.

**Potential Implications:**

* **Skewed Distributions:** Features like dewpoint, cloud, sunshine, windspeed, and rainfall may benefit from log or other power transformations.
* **Bimodal Distributions:** Temperature and humidity might require feature engineering to capture the different modes (e.g., creating binary features based on the modes).
* **Outliers:** Pressure may have a few low outliers that should be examined.
* **Zero Values:** Rainfall has a very high concentration of zero values, which might be handled by creating a binary "rain" feature or using a log transform with an offset.

---

### Interpretation of Meteorological Data Graphs


[image here]()

The provided image contains a series of six line graphs and three scatter plots, comparing various meteorological variables against the day of the year (ranging from 0 to 350, likely representing a year). The variables analyzed include pressure, maximum temperature, temperature, minimum temperature, dew point, humidity, cloud cover, sunshine, wind speed, and wind direction. Each graph compares these variables under two conditions: with rainfall (orange line, labeled "rainfall = 1") and without rainfall (blue line, labeled "rainfall = 0").

### Key Observations:

#### Line Graphs:
1. **Pressure vs. Day**:
   - Pressure fluctuates between approximately 1000 and 1035 hPa.
   - No significant trend is observed, but periodic drops are noticeable around days 100-150 and 250-300, possibly correlating with rainfall even
2. **Maximum Temperature vs. Day**:
   - Ranges from about 15°C to 35°C, showing a clear seasonal pattern with a peak around days 150-200 (summer) and a dip around days 300-350 (winter).
   - Rainfall (orange) tends to lower the maximum temperature compared to no rainfall (blue).

3. **Temperature vs. Day**:
   - Follows a seasonal trend with a peak around mid-year and a decline toward the year's end.
   - Rainfall slightly reduces the average temperature.

4. **Minimum Temperature vs. Day**:
   - Ranges from 5°C to 25°C, with a similar seasonal pattern.
   - Rainfall has a less pronounced effect on minimum temperatures compared to maximum temperatures.

5. **Dew Point vs. Day**:
   - Varies between 5°C and 20°C, following a seasonal trend.
   - Rainfall increases the dew point, indicating higher moisture levels.

6. **Humidity vs. Day**:
   - Ranges from 40% to 100%, with higher values during rainy periods (orange).
   - A seasonal trend is less clear, but humidity spikes are more frequent with rainfall.

7. **Cloud vs. Day**:
   - Cloud cover ranges from 20 to 100, with higher values during rainy days, suggesting more cloudiness with rainfall.

8. **Sunshine vs. Day**:
   - Sunshine hours range from 0 to 12 hours, peaking around mid-year and dropping toward the year's end.
   - Rainfall significantly reduces sunshine hours.

9. **Wind Speed vs. Day**:
   - Fluctuates between 10 and 50 units (possibly m/s), with no strong seasonal trend.
   - Rainfall appears to increase wind speed slightly.

10. **Wind Direction vs. Day**:
   - Varies widely (0 to 300 degrees), with no clear seasonal pattern.
   - Rainfall does not seem to significantly alter wind direction.

#### Scatter Plots:
   - The three scatter plots (sunshine vs. rainfall, wind speed vs. rainfall, wind direction vs. rainfall) show weak or no correlation between these variables and rainfall, as the data points are scattered without a clear trend.

### General Insights:
   - The data suggests a strong seasonal influence on temperature-related variables (max temp, min temp, dew point), with a peak in the middle of the year.
   - Rainfall consistently affects meteorological conditions, generally increasing humidity, cloud cover, dew point, and wind speed while decreasing temperature and sunshine.
   - The lack of strong correlations in the scatter plots indicates that rainfall's impact on sunshine, wind speed, and wind direction might be context-dependent or influenced by other factors not shown here.

---

![image here ]()

### Rainfall Analysis by Month

This repository contains an analysis of monthly rainfall data, visualized in a horizontal bar chart.

#### Chart Description

The chart displays the rainfall count for each month of the year, ordered from January (1) to December (12). The rainfall count is represented on the horizontal axis, ranging from 0 to 200.


#### Key Findings

-   **Consistent Rainfall:** The rainfall is relatively consistent throughout the year.
-   **Peak Months:** January and March exhibit slightly higher rainfall counts.
-   **Lowest Months:** February and November show the lowest rainfall counts.
-   **Range:** All months fall within a rainfall count range of approximately 160 to 200.

#### Interpretation

The data suggests a fairly uniform distribution of rainfall across the months, with minor variations. This indicates that the region experiences consistent rainfall throughout the year.

---

![image here]()

### **EDA - Boxplots of Meteorological Features vs. Day**

This section of the exploratory data analysis (EDA) visualizes the distribution of meteorological features against the "day" variable using boxplots.

### Contents

- [Visual Description](#visual-description)
- [Observations](#observations)

#### Visual Description

The boxplots represent the distribution of various meteorological features across different days:

- **Pressure vs. Day**  
  - Range: ~1000 to 1025 hPa  
  - Median: ~1015 hPa  
  - Outliers: A few below 1005 hPa  

- **Max Temp vs. Day**  
  - Range: ~10°C to 35°C  
  - Median: ~25°C  

- **Temperature vs. Day**  
  - Range: ~10°C to 30°C  
  - Median: ~22°C  

- **Min Temp vs. Day**  
  - Range: ~5°C to 25°C  
  - Median: ~15°C  

- **Dewpoint vs. Day**  
  - Range: ~0°C to 25°C  
  - Median: ~15°C  
  - Outliers: Below 5°C  

- **Humidity vs. Day**  
  - Range: ~40% to 100%  
  - Median: ~80%  
  - Outliers: Below 60%  

- **Cloud vs. Day**  
  - Range: ~0 to 100  
  - Median: ~50  
  - Outliers: Below 20  

- **Sunshine vs. Day**  
  - Range: ~0 to 12 hours  
  - Median: ~5 hours  

- **Wind Direction vs. Day**  
  - Range: ~0 to 300 degrees  
  - Median: ~150 degrees  

- **Wind Speed vs. Day of Week**  
  - Range: ~0 to 50  
  - Median: ~30  
  - Outliers: Above 50  

- **Week of Year vs. Day**  
  - Range: ~0 to 50  
  - Median: ~25  

- **Month vs. Day**  
  - Range: 1 to 12  
  - Median: ~6  

- **Day of Week vs. Day**  
  - Range: 0 to 6  
  - Median: ~3  

- **Rainfall vs. Day (Bottom Two Plots)**  
  - Range: 0 to 1 (binary or normalized)  
  - Median: ~0.5  
  - Likely represents rain (1) vs. no rain (0)

#### Observations

- **Pressure**: Generally stable around 1015 hPa, with minor variations and a few low-pressure outliers.
- **Temperature (Max, Min, Mean)**: Significant variation, with maximum temperatures reaching 35°C and minimum temperatures dropping to 5°C.
- **Humidity and Dewpoint**: High median humidity (~80%) and dewpoint (~15°C) indicate humid conditions, with some drier outliers.
- **Cloud and Sunshine**: Wide range in cloud cover and sunshine hours suggests variable weather.
- **Wind**: Wind direction and speed vary, with some high wind speed outliers.
- **Temporal Features**: Month, week of year, and day of week are distributed as expected, with medians around the middle of their respective ranges.
- **Rainfall**: The binary-like distribution (0 to 1) suggests the dataset is suited for binary classification (rain vs. no rain), with a balanced occurrence of both outcomes."

---

![image here]()

### **EDA - Rainfall (0/1) Distribution**

This section of the exploratory data analysis (EDA) examines the distribution of the binary rainfall variable in the dataset. The visualization is a histogram showing the count of rainy and non-rainy days.

### Contents

- [Visual Description](#visual-description)
- [Observations](#observations)

#### Visual Description

The histogram below represents the distribution of the binary `rainfall` variable:

- **Title**: Rainfall (0/1) Distribution
- **X-axis**: Rainfall (binary values)
  - `0`: No rain
  - `1`: Rain
- **Y-axis**: Count of occurrences
- **Histogram Bars**:
  - `rainfall = 0` (no rain): ~500 instances
  - `rainfall = 1` (rain): ~1500 instances

#### Observations

- The dataset contains approximately 2000 data points (500 for no rain + 1500 for rain).
- The distribution is imbalanced:
  - **No Rain (0)**: ~500 instances, accounting for ~25% of the data.
  - **Rain (1)**: ~1500 instances, accounting for ~75% of the data.
- This imbalance indicates that rainy days are significantly more frequent in the dataset compared to non-rainy days.
- For a binary classification task (e.g., predicting rain vs. no rain), this imbalance may lead to a model bias towards predicting rain. To mitigate this, techniques such as oversampling the minority class (no rain), undersampling the majority class (rain), or applying class weights during model training may be necessary.

---

![table here]()

### Observations

- The highest rainfall is observed in Month 4 (0.924324), while the lowest is in Month 7 (0.641304).
- The highest temperature and dewpoint are recorded in Month 7 (29.508152 and 25.266848, respectively), indicating warmer and more humid conditions.
- Pressure values vary from 1007.841304 (Month 7) to 1019.794924 (Month 1), reflecting changes in atmospheric conditions.

---

### **EDA - Monthly Average Weather Features**

This section of the exploratory data analysis (EDA) visualizes the normalized average values of weather features (pressure, temperature, dewpoint, humidity, and rainfall) across each month of the year. The visualization is a bar chart showing how these features vary month by month.

### Contents

- [Visual Description](#visual-description)
- [Observations](#observations)

#### Visual Description

The bar chart below represents the normalized average values of weather features across each month:

- **Title**: Monthly Average Weather Features
- **X-axis**: Month (1 to 12)
- **Y-axis**: Average value (normalized, ranging from -1.5 to 1.0)
- **Legend**:
  - **Pressure**: Blue
  - **Temperature**: Orange
  - **Dewpoint**: Green
  - **Humidity**: Red
  - **Rainfall**: Purple
- **Monthly Breakdown**:
  - **January (Month 1)**: Pressure (~0.5), temperature (~-1.0), dewpoint (~-1.0), humidity (slightly negative), rainfall (slightly negative).
  - **February (Month 2)**: Pressure (~0.5), temperature (~-1.0), dewpoint (~-1.0), humidity (slightly negative), rainfall (slightly negative).
  - **March (Month 3)**: Pressure (~0.2), temperature (~-0.5), dewpoint (~-0.5), humidity (slightly negative), rainfall (slightly negative).
  - **April (Month 4)**: Pressure (slightly negative), temperature (~0.5), dewpoint (~0.5), humidity (~0.5), rainfall (~0.5).
  - **May (Month 5)**: Pressure (slightly positive), temperature (~0), dewpoint (~0), humidity (~0), rainfall (~0).
  - **June (Month 6)**: Pressure (~-0.5), temperature (~0.8), dewpoint (~0.8), humidity (~0.2), rainfall (~0.2).
  - **July (Month 7)**: Pressure (~-0.5), temperature (~1.0), dewpoint (~1.0), humidity (~0.2), rainfall (~0.2).
  - **August (Month 8)**: Pressure (~-0.5), temperature (~0.8), dewpoint (~0.8), humidity (~0.2), rainfall (~0.2).
  - **September (Month 9)**: Pressure (slightly negative), temperature (~0.5), dewpoint (~0.5), humidity (~0.2), rainfall (~0.2).
  - **October (Month 10)**: Pressure (slightly positive), temperature (~0.5), dewpoint (~0.5), humidity (~0.2), rainfall (~0.2).
  - **November (Month 11)**: Pressure (~0.5), temperature (~0.2), dewpoint (~0.2), humidity (~0.2), rainfall (~0.5).
  - **December (Month 12)**: Pressure (~0.5), temperature (~-1.0), dewpoint (~-1.0), humidity (slightly negative), rainfall (slightly negative).

#### Observations

- **Pressure**: Higher in colder months (January, February, November, December) with values around 0.5, and lower in warmer months (June, July, August) with values around -0.5.
- **Temperature and Dewpoint**: Both features peak in July (~1.0) and dip in January and December (~-1.0), showing a strong seasonal trend.
- **Humidity**: Relatively stable, with slight positive values (around 0.2 to 0.5) in most months, peaking in April.
- **Rainfall**: Peaks in April and November (~0.5), with lower values in colder months (January, February, December).
- **Seasonal Trends**: Warmer months (June to August) have higher temperatures and dewpoints, lower pressure, and moderate rainfall, while colder months (December to February) have higher pressure and lower temperatures.

---

![heatmap]()

### **EDA - Correlation Matrix of Weather Features**

This section of the exploratory data analysis (EDA) examines the relationships between weather features (pressure, temperature, dewpoint, humidity, and rainfall) using a correlation matrix. The matrix visualizes the Pearson correlation coefficients, highlighting the strength and direction of relationships between variables.

### Contents

- [Visual Description](#visual-description)
- [Observations](#observations)

#### Visual Description

The correlation matrix below represents the Pearson correlation coefficients between five weather features:

- **Title**: Correlation Matrix
- **Features**: Pressure, Temperature, Dewpoint, Humidity, Rainfall
- **Color Scale**:
  - Blue: Negative correlation (closer to -1.0)
  - Red: Positive correlation (closer to 1.0)
  - White/Neutral: Weak or no correlation (around 0.0)
- **Correlation Values**:
  - **Pressure vs. Pressure**: 1.00
  - **Pressure vs. Temperature**: -0.82
  - **Pressure vs. Dewpoint**: -0.82
  - **Pressure vs. Humidity**: -0.12
  - **Pressure vs. Rainfall**: -0.05
  - **Temperature vs. Temperature**: 1.00
  - **Temperature vs. Dewpoint**: 0.93
  - **Temperature vs. Humidity**: -0.03
  - **Temperature vs. Rainfall**: -0.05
  - **Dewpoint vs. Dewpoint**: 1.00
  - **Dewpoint vs. Humidity**: 0.15
  - **Dewpoint vs. Rainfall**: 0.08
  - **Humidity vs. Humidity**: 1.00
  - **Humidity vs. Rainfall**: 0.45
  - **Rainfall vs. Rainfall**: 1.00

#### Observations

- **Strong Correlations**:
  - **Temperature and Dewpoint (0.93)**: A very strong positive correlation indicates that as temperature increases, dewpoint tends to increase as well, reflecting their close meteorological relationship.
  - **Pressure with Temperature and Dewpoint (-0.82 for both)**: Strong negative correlations suggest that higher pressure is associated with lower temperatures and dewpoints, often indicative of cooler, drier conditions.
- **Moderate Correlation**:
  - **Humidity and Rainfall (0.45)**: A moderate positive correlation suggests that higher humidity is associated with a higher likelihood of rainfall, which aligns with meteorological intuition.
- **Weak Correlations**:
  - **Pressure and Humidity (-0.12)**: A weak negative correlation indicates a slight tendency for higher pressure to be associated with lower humidity.
  - **Dewpoint and Humidity (0.15)**: A weak positive correlation shows a mild relationship between these variables.
  - **Rainfall with Temperature (-0.05) and Dewpoint (0.08)**: Very weak correlations suggest that rainfall is not strongly influenced by temperature or dewpoint in this dataset.
- **Implications**:
  - The strong correlation between temperature and dewpoint suggests potential multicollinearity, which may need to be addressed in a predictive model (e.g., by removing one feature or using dimensionality reduction techniques like PCA).
  - The moderate correlation between humidity and rainfall indicates that humidity could be a useful predictor for rainfall.
  - The weak correlations between rainfall and other features (pressure, temperature, dewpoint) suggest that these features alone may not be strong predictors of rainfall, and additional features or transformations might be needed.

---

![rainfall per week]()

### **EDA - Rainfall Count by Week of Year**

This section of the exploratory data analysis (EDA) visualizes the count of rainfall events across the weeks of a year. The graph provides insights into the seasonal patterns of rainfall, which can be valuable for weather prediction models.

### Contents

- [Visual Description](#visual-description)
- [Observations](#observations)

#### Visual Description

The line graph below represents the rainfall count by week of year:

- **Title**: Rainfall Count by Week of Year
- **X-axis**: Week of Year (1 to 52)
- **Y-axis**: Rainfall Count (0 to 50)
- **Data Trend**:
  - Starts at ~40-45 in weeks 1-5.
  - Fluctuates with peaks around weeks 10, 20, and 35-40 (~45).
  - Generally hovers between 30 and 45 from weeks 5-45.
  - Declines sharply after week 45, dropping below 10 by week 50.

#### Observations

- **Seasonal Pattern**: The graph indicates a seasonal trend, with higher rainfall counts in the earlier and middle parts of the year (weeks 1-40) and a significant decrease towards the end (weeks 45-52).
- **Peak Rainfall**: Highest counts occur around weeks 10, 20, and 35-40, suggesting possible rainy seasons or periods of higher precipitation.
- **Decline**: A sharp decline after week 45 points to a dry season or period with minimal rainfall towards the year's end.
- **Stability**: The rainfall count remains relatively stable between 30 and 45 for most of the year (weeks 5-45), despite fluctuations.
- **Implications**: This pattern can aid in rainfall prediction, with the end of the year indicating lower rainfall likelihood. Further analysis could correlate this with other weather features like temperature or humidity.'

---

![image here]()

### Interpretation of the Weather Features Graph  

#### Overview  
This line graph represents the **weekly average fluctuations** of key weather features throughout the year. The parameters include:  

- **Pressure (hPa)** - A dominant feature fluctuating around **1000 hPa**.  
- **Temperature (°C)** - Shows slight seasonal variations.  
- **Dewpoint (°C)** - Closely follows the temperature trend.  
- **Humidity (%)** - Generally stays between **80-100%** with minor fluctuations.  
- **Rainfall (mm)** - Low values with occasional peaks, indicating seasonal rainfall.  

#### Key Observations  
- **Pressure** is relatively stable across all weeks.  
- **Temperature and Dewpoint** show expected seasonal changes.  
- **Humidity** remains consistently high with slight variations.  
- **Rainfall** shows periodic increases, suggesting wet and dry seasons.  

#### Conclusion  
This visualization is useful for analyzing seasonal trends and understanding climate behavior over time.

---
## Data Modeling

I trained four different models and will just discussed two ones

### Rainfall Prediction using XGBoost  

#### Overview  
This project focuses on building an XGBoost-based classifier to predict rainfall based on meteorological data. The dataset includes weather attributes such as temperature, pressure, humidity, and wind speed.  

#### **Model Performance ** 

#### Confusion Matrix  
![Confusion Matrix](xg_cm.png)  

#### Feature Importance  
![Feature Importance](xg_featureimportance.png)  

#### ROC Curve  
![ROC Curve](xg_roc.png)  

- **AUC Score:** 0.78 (Indicates good predictive performance)  
- **Most Important Features:**  
  - Day  
  - Pressure  
  - Max Temperature  
  - Wind Speed  
  - Sunshine  

#### Results and Observations  
- The model performs well, achieving a good balance of precision and recall.  
- Feature importance analysis reveals that atmospheric pressure and temperature play significant roles in rainfall prediction.  

#### Future Improvements  "will not be done here for the tome shortage"
- **Hyperparameter tuning**: Further optimization of XGBoost parameters.  
- **Feature selection**: Reducing less significant features to enhance efficiency.  
- **Additional Data Sources**: Incorporating external weather patterns for better forecasting.  

---

### Rainfall Prediction using Random Forest  

#### **Model Performance **

#### Confusion Matrix  
![Confusion Matrix](randomforest_heatmap.png)  

#### Feature Importance  
![Feature Importance](randomfirest_featureimportanxe.png)  

#### ROC Curve  
![ROC Curve](rf_roc.png)  

- **AUC Score:** 0.79 (Indicates strong predictive performance)  
- **Most Important Features:**  
  - Cloud Cover  
  - Sunshine  
  - Humidity  
  - Dew Point  

#### Results and Observations  
- The Random Forest model performs slightly better than XGBoost in terms of AUC.  
- Feature importance suggests that cloud cover and sunshine are critical factors in predicting rainfall.  

#### Future Improvements  
- **Hyperparameter tuning**: Further optimization of Random Forest parameters.  
- **Feature engineering**: Exploring interactions between weather variables.  
- **Comparison with other models**: Evaluating Gradient Boosting, SVM, and Neural Networks.  

---
## Data Limitation

While the dataset provides valuable insights for rainfall prediction, it has certain limitations:

- **Class Imbalance:** The dataset contains significantly more rainy days (75%) than non-rainy days (25%), which can lead to biased predictions. Resampling techniques (e.g., oversampling, undersampling) or adjusting class weights may be necessary to balance the dataset.  
- **Weak Correlations:** Some key meteorological features, such as temperature and dewpoint, show only weak correlations with rainfall. This suggests that additional features or transformations may improve predictive performance.  
- **Seasonal Variability:** The dataset indicates strong seasonal patterns, with temperature, dewpoint, and rainfall fluctuating significantly throughout the year. However, these patterns may not generalize well to other locations or time periods.  
- **Limited Feature Interactions:** Certain features like wind speed and direction show little to no correlation with rainfall, suggesting that more complex interactions or additional weather variables (e.g., atmospheric pressure changes over time) may enhance model accuracy.

---

## Results

Two machine learning models were trained and evaluated for rainfall prediction:

### **Random Forest Classifier**
- **Accuracy:** Higher than XGBoost, with improved performance.  
- **AUC Score:** 0.79 (better predictive capability).  
- **Feature Importance:** Cloud cover, sunshine, humidity, and dew point were the most influential variables.  
- **Confusion Matrix:**
  - True Positives: 1540  
  - False Positives: 191  
  - True Negatives: 349  
  - False Negatives: 110  

### **XGBoost Classifier**
- **Accuracy:** Good predictive performance but slightly lower than Random Forest.  
- **AUC Score:** 0.78.  
- **Feature Importance:** Day, pressure, max temperature, wind speed, and sunshine levels had the most impact.  
- **Confusion Matrix:**
  - True Positives: 1521  
  - False Positives: 196  
  - True Negatives: 344  
  - False Negatives: 129  

Both models demonstrated solid performance, with Random Forest outperforming XGBoost slightly in terms of classification accuracy and AUC score.


---
## Recommendations
To further improve rainfall prediction, the following steps are suggested:

- **Address Class Imbalance:** Implement SMOTE (Synthetic Minority Over-sampling Technique) or class weighting to ensure the model does not favor the dominant class (rainy days).  
- **Feature Engineering:** Introduce additional weather variables such as past rainfall trends, atmospheric pressure gradients, and advanced meteorological indices to capture hidden patterns.  
- **Hyperparameter Optimization:** Use grid search or Bayesian optimization to fine-tune model parameters for better performance.  
- **Use Ensemble Methods:** Combining multiple models (e.g., stacking Random Forest and XGBoost) may improve accuracy by leveraging their strengths.  
- **Explore Deep Learning:** Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTM) models could be tested for time-series analysis of weather patterns.  
- **Expand Dataset:** Incorporating real-time weather API data or larger datasets from different regions may help improve model generalization.  

By implementing these strategies, the rainfall prediction model can be further refined, leading to more accurate and reliable forecasts.

---
## Installation

``` bash
git clone git@github.com:7arb25/Binary-Prediction-with-a-Rainfall-Dataset.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# You will need access to CSM-1B and Llama-3.2-1B
huggingface-cli login
```

---

## License

This project is licensed under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** License.  

You are free to:  
- **Share** — copy and redistribute the material in any medium or format.  
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.  

Under the following terms:  
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.  
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.  

For more details, see the full license [here](https://creativecommons.org/licenses/by-sa/4.0/).

## Author

Abdelrahman G. A. Ebrahim

- [LinkedIn](https://LinkedIn.com/in/3bd0g0m3aa)
- 3bd0.g0m3aa@gmail.com
