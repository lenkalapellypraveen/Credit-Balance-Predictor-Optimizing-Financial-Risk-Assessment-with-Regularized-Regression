# Credit Card Balance Prediction

## Project Overview
This project develops a sophisticated predictive analytics framework for forecasting credit card balances by statistically modeling customer demographics and financial attributes. The implementation follows a structured data science workflow encompassing exploratory analysis, feature engineering, model development, and comprehensive validation methodologies to deliver production-grade predictive capabilities.

The core technical approach leverages regularization-based regression techniques:
- **Elastic Net Regression**: Combining L1 and L2 penalties to handle feature selection and multicollinearity
- **Ridge Regression**: Applying L2 regularization to stabilize predictions with correlated features
- **Lasso Regression**: Utilizing L1 regularization for feature selection and coefficient sparsity

Through hyperparameter tuning and cross-validation, the models achieved the following performance metrics:
- Elastic Net: MSE of 8233.94, R² of 0.950717, with alpha=1.0 and l1_ratio=1.0
- Ridge: MSE of 8016.14, R² of 0.952020, with alpha=0.1
- Lasso: MSE of 8233.94, R² of 0.950717, with alpha=1.0

Notably, the Ridge model slightly outperformed the others, though all models demonstrated strong predictive power with R² values above 0.95. Feature importance analysis revealed that Credit Limit (462.83), Income (261.84), and Rating (138.91) were the most significant predictors, while demographic factors like Married (1.29) and Education (2.67) had minimal impact. Student status emerged as an unexpectedly important predictor (124.55), ranking fourth in importance.

The project includes comprehensive data exploration, feature engineering, model evaluation, and interpretation components, providing both accurate predictions and actionable insights from the modeling process.

## Business Context
Understanding and accurately predicting customer credit balances is crucial for:
- Risk management and credit scoring
- Financial forecasting and portfolio management
- Targeted marketing and product offerings
- Customer financial behavior analysis

## Dataset
The dataset contains information about 400 credit card customers with the following features presented in tabular format:

| Feature    | Description                                                     |
|------------|-----------------------------------------------------------------|
| Income     | Annual income in dollars                                        |
| Limit      | Credit card limit                                               |
| Rating     | Credit score (similar to TransUnion/Equifax, range: 138-949)    |
| Cards      | Number of credit cards owned                                    |
| Age        | Customer age in years                                           |
| Education  | Years of education                                              |
| Gender     | Male/Female                                                     |
| Student    | Student status (Yes/No)                                         |
| Married    | Marital status (Yes/No)                                         |
| Balance    | Outstanding credit card balance (target variable)               |

Balance represents the actual outstanding amount on customers' credit cards - essentially how much they've borrowed. This is distinct from their Credit Limit, which was one of our predictive features. Understanding and predicting customer balance helps assess credit utilization ratios and potential default risk.

## Methodology
The project follows a comprehensive data science workflow:

### 1. Project Setup and Data Acquisition
- Definition of business objectives and success metrics
- Data loading and initial examination

### 2. Data Exploration and Understanding
- Statistical analysis of dataset properties
- Visualization of feature distributions and correlations
- Identification of patterns and relationships between features and target variable

### 3. Data Preprocessing
- Conversion of categorical variables (Gender, Student, Married) to binary format
- Feature standardization to ensure model stability
- Target variable centering

### 4. Feature Selection
- Correlation analysis to identify relationships with target variable
- Assessment of feature importance for balance prediction

### 5. Train-Test Split
- Division of data into training (80%) and testing (20%) sets
- Ensuring consistent evaluation methodology

### 6. Model Selection and Implementation
- Implementation of regularized regression models:
  - Elastic Net (combines L1 and L2 penalties)
  - Ridge Regression (L2 penalty)
  - Lasso Regression (L1 penalty)

### 7. Hyperparameter Tuning
- Grid search with 5-fold cross-validation for optimal parameters
- Exploration of alpha (regularization strength) and l1_ratio (Elastic Net mixing parameter)

### 8. Model Training and Evaluation
- Training models with optimal hyperparameters
- Evaluation using Mean Squared Error (MSE) and R² metrics
- Comparison of model performance

### 9. Coefficient Analysis and Feature Importance
- Examination of how regularization affects feature coefficients
- Identification of key predictors for credit balance

### 10. Visualization of Results
- Regularization path visualization
- Actual vs. predicted balance plots
- Feature importance charts

### 11. Model Deployment Preparation
- Creation of preprocessing and prediction functions
- Sample customer prediction demonstration

### 12. Documentation and Reporting
- Performance metrics summary
- Feature importance ranking
- Business insights extraction

## Key Findings
- Ridge Regression achieved the best performance with an MSE of 8016.14 and R² of 0.952020
- Elastic Net and Lasso models performed similarly with an MSE of 8233.94 and R² of 0.950717
- The top predictors of credit balance were:
  1. Credit Limit (462.83)
  2. Income (261.84)
  3. Rating (138.91)
  4. Student Status (124.55)
  5. Cards (26.85)
- Demographic factors such as Gender, Education, and Marital status had minimal impact on balance prediction

## Technologies Used
- Python 3.x
- pandas for data manipulation
- numpy for numerical operations
- scikit-learn for modeling and evaluation
- matplotlib and seaborn for visualization

## Project Structure
```
.
├── data/
│   └── Credit_M400_p9.csv
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_model_building.ipynb
│   └── 4_model_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
├── models/
│   └── best_elastic_net_model.pkl
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Use
1. Clone this repository
   ```
   git clone https://github.com/username/credit-balance-prediction.git
   ```

2. Install required packages
   ```
   pip install -r requirements.txt
   ```

3. Run the notebooks to reproduce the analysis or use the prediction function in `src/model_training.py`

## Future Work
- Incorporate additional financial behavior data for more accurate predictions
- Explore non-linear relationships using advanced algorithms
- Develop a customer segmentation strategy based on predicted balance behavior
- Create a web application for interactive balance prediction

## Contact
For questions or collaboration, please contact [praveen.lenkalapelly9@gmail.com].
