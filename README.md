### Employee Attrition: Modeling for Attrition Prediction
**Faranak Yousefi**

#### Executive Summary
This project analyzes factors driving employee attrition through Exploratory Data Analysis (EDA) and machine learning classification models. Key findings highlight overtime, low monthly income, job satisfaction, commute distance, and short tenure as primary predictors. A tuned Gradient Boosting ensemble outperforms baselines, achieving a weighted F1-score of ~0.94 and AUC-ROC of ~0.94, with a minority class F1 of ~0.42 after threshold optimization. SHAP interpretability reveals interactions, such as overtime amplifying attrition risk for low-income employees. These insights enable targeted HR interventions to reduce turnover, potentially saving costs in recruitment, training, and productivity by 10-20%.

#### Why This Question is Important
Employee attrition imposes significant costs on organizations, including recruitment, training, lost knowledge, and reduced productivity. By identifying key predictors, this analysis empowers HR teams to implement proactive retention strategies—such as improving work-life balance, adjusting compensation for at-risk groups, or enhancing manager training—ultimately boosting morale, organizational performance, and retention rates in competitive markets.

#### Research Question
What are the primary factors influencing employee attrition in a company, and how do they contribute to predicting turnover?

#### Data Sources
The IBM HR Analytics Employee Attrition & Performance dataset, available at https://ieee-dataport.org/documents/ibm-hr-analytics-employee-attrition-performance (also mirrored on Kaggle for accessibility). It contains 1470 records with 35 features, including Age, Job Satisfaction, Monthly Income, OverTime, DistanceFromHome, and the binary target variable Attrition (imbalanced at ~16% "Yes").

#### Methodology
- **EDA and Data Understanding**: Conducted in a dedicated notebook using Pandas for cleaning (outlier capping, removal of constants like EmployeeCount, Over18, StandardHours), Matplotlib/Seaborn for visualizations (distributions, correlations, boxplots, countplots, heatmaps), and statistical tests (t-tests, chi-square) to identify imbalances and associations (e.g., attrition linked to age, satisfaction, overtime).
- **Feature Engineering**: Added AgeBin (categorical bins: 18-30, 31-40, etc.) and TenureRatio (YearsAtCompany / TotalWorkingYears) to capture nonlinear effects.
- **Preprocessing**: Used ColumnTransformer for scaling numerical features (StandardScaler) and one-hot encoding categoricals (OneHotEncoder). Handled imbalance with SMOTE on training data.
- **Modeling**: Baselines included Logistic Regression, KNN, Decision Tree, SVM. Advanced models: Random Forest and Gradient Boosting with GridSearchCV tuning (e.g., n_estimators, learning_rate, max_depth). Ensemble via VotingClassifier (soft voting). Evaluation focused on F1-score (weighted and minority class) and AUC-ROC for imbalance. Added threshold tuning via precision-recall curve to optimize minority class F1.
- **Interpretability**: Permutation importance for feature ranking; SHAP (SHapley Additive exPlanations) for global/local explanations and interactions (e.g., dependence plots for OverTime_Yes and MonthlyIncome).

#### Results
EDA confirmed attrition correlations with younger age, lower job satisfaction/income, overtime, longer commutes, sales roles, and single marital status. Baseline Logistic Regression: weighted F1 ~0.89, AUC ~0.92. Tuned Gradient Boosting: weighted F1 ~0.94, AUC ~0.94, with minority class F1 improved to ~0.42 via threshold tuning (optimal threshold ~0.32 for better recall). Ensemble (VotingClassifier) slightly edges individual models, validating combination benefits. Top features from permutation importance: OverTime_Yes (~0.037), Department_Sales (~0.017), BusinessTravel_Frequently (~0.017), JobInvolvement (~0.016), JobRole_Human Resources (~0.016). SHAP analysis revealed nonlinear interactions, e.g., overtime's attrition push is amplified for low-income employees, providing directional insights (positive SHAP for high-risk conditions).

#### Limitations
The dataset is synthetic and may not capture real-world nuances like economic factors or cultural differences. Its small size and imbalance could lead to overfitting despite mitigation techniques. External variables (e.g., job market conditions) are absent, and interpretations are correlational, not causal. SHAP is model-dependent, and results may vary with hyperparameter changes.

#### Future Work
Validate on real-world, diverse datasets across industries. Incorporate time-series elements for trend prediction. Explore advanced models like XGBoost or neural networks. Add external features (e.g., sentiment analysis from surveys) and deploy as interactive tools (e.g., Streamlit apps) with cost-sensitive learning. Conduct A/B testing to measure actual turnover reductions.

#### Outline of Project
- [Link to EDA Notebook](EDA-DataUnderstanding.ipynb)
- [Link to Modeling Notebook](EmployeeAttrition.ipynb)
- [Link to Presentation](Presentation/EmployeeAttritionPresentation.pptx)

##### Contact and Further Information
For questions or collaboration, contact Faranak Yousefi. Additional resources:[[GitHub Repository]](https://github.com/Faranakysf/PredictEmployeeAttrition).
