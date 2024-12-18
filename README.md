# US Accidents Data Analysis (2016–2023) 🚗💥

## **Project Overview**  📊

The objective of this project is to analyze the US Accidents dataset to identify key factors contributing to road accidents, understand accident patterns, and predict accident severity using machine learning models. The analysis provides insights into geographic, temporal, and weather-related accident trends, offering data-driven solutions for improving road safety.

## **Dataset** 📁

Source: Kaggle - US Accidents Dataset [ https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/ ] 
Description:
  Over 7.7 million records of traffic accidents from 2016 to 2023.
  Key features: Start_Time, Weather_Condition, Severity, Temperature(F),
                Visibility(mi), Humidity(%), Pressure(in), Precipitation(in), 
                Location Coordinates, etc.
Scope: A subset of 10,000 records was used for analysis to ensure computational efficiency.


## **Data Mining Task** 🧠

We approached the problem using classification techniques:

### **Binary Classification**: 

Predicting accident severity as Severe or Non-Severe.

### **Multiclass Classification:** 

Classifying accidents into multiple severity levels (1 to 4).


## **Methodology**

**1. Data Preprocessing:** ✅

 Missing Value Handling: Mean/Mode imputation and critical removal.
 Outlier Detection: Used the IQR method to cap unrealistic values.
 Feature Engineering: Label Encoding, One-Hot Encoding, and Standardization using StandardScaler and MinMaxScaler.

**2.Exploratory Data Analysis (EDA):** ✅

 Temporal trends (yearly, weekly, hourly patterns).
 Geospatial analysis to identify accident hotspots.
 Impact of weather and road conditions on accident severity.

**3.Models Used:** ✅

 Binary Classification:
  Random Forest, Bagging Classifier, SVC, Perceptron
 Multiclass Classification:
  Random Forest, Bagging Classifier, Logistic Regression, KNN
 
**4.Performance Metrics:** ✅
 
  Accuracy, Precision, Recall, F1 Score, and ROC-AUC.


 **Deployment**: ❌ 
  
  Planned for future work.



## **Key Insights**

1. **Temporal Trends**: Accidents peak during rush hours (7-9 AM, 4-6 PM) and in winter months (Nov-Dec).

2. **State-Level Trends:** California, Texas, and Florida have the highest accident rates.

3. **Weather Conditions:** Rain, fog, and overcast weather significantly influence accident frequency.

4. **Top-Performing Models:**
Random Forest Classifier and Bagging Classifier demonstrated the highest accuracy and AUC scores across both binary and multiclass tasks.


## **Results** 🎯

**Binary Classification Performance:**

| Model                       | Test Accuracy |   AUC   |  F1 Score |
| --------------------------- | ------------- | ------- | --------- |
| RandomForestClassifier	  |    82.4%      |   0.76  |   High    |
| BaggingClassifier	          |    82.1%	  |   0.76  |  Balanced |

 

**Multiclass Classification Performance:**

| Model                       | Test Accuracy |   AUC   |
| --------------------------- | ------------- |  -----  |
| RandomForestClassifier	  |    81%        |   0.81  | 
| BaggingClassifier	          |    81%  	  |   0.76  |



## **Visualizations** 📊

1. **Temporal Analysis:** Trends across years, weekdays, and hours.
2. **Geospatial Analysis:** Accident hotspots visualized using heatmaps and 3D scatter plots.
3. **Weather Impact:** Distribution of accidents under different weather conditions.
4. **Model Performance:** Comparative ROC curves and accuracy metrics for classifiers.

## **Tools and Technologies** 🛠️

**Python Libraries 🐍:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Folium  📊
 
**Notebooks:** Jupyter Notebook  📓
 
**Visualization Tools:** Matplotlib, Plotly (for interactive charts)  📊
 

## **Conclusion**

This project successfully analyzed the US Accidents dataset and identified actionable insights into accident trends. The top-performing classification models, Random Forest and Bagging Classifier, demonstrated strong predictive capabilities, making them useful for future accident prediction and prevention strategies.

## **Contributors** 🤝

Pratik Harlikar

Aditya Jadhav

Yatziri Presmont

Muhammad Batikh

We welcome contributions to improve this project! Feel free to:

- Fork the repository 🍴

- Create a pull request 🔄

- Suggest improvements in issues ✍️

## **License** 📝
 
This project is licensed under the MIT License.


