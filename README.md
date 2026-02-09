# CHD Risk Prediction Model

Predicting 10-year coronary heart disease risk using the Framingham Heart Study dataset.

## Project Overview

This project builds a machine learning model to predict whether a patient will develop coronary heart disease within 10 years based on demographic, behavioral, and clinical factors. Multiple optimization techniques are applied to improve model performance on this imbalanced medical dataset.

## Dataset

The Framingham dataset includes:
- Demographics: age, sex, education
- Behavioral factors: smoking status, cigarettes per day
- Clinical measurements: blood pressure, cholesterol, BMI, glucose, heart rate
- Medical history: hypertension, diabetes, stroke, BP medication use

Target variable: TenYearCHD (1 = CHD event within 10 years, 0 = no event)

Class distribution: 85% negative (no CHD), 15% positive (CHD) - highly imbalanced

## Methodology

### Data Preprocessing
- Missing value imputation using mean values
- Outlier detection and replacement using IQR method
- Feature standardization using StandardScaler

### Feature Engineering
Created interaction and domain-specific features:
- Age interactions: age × smoking, age × systolic BP, age × diabetes
- Clinical interactions: cholesterol × BP, BMI × smoking
- Cardiovascular risk indicators: pulse pressure, high cholesterol flag, high glucose flag

### Model Development
- Train/validation/test split: 70/15/15
- Baseline model: L2 regularized logistic regression
- Advanced model: XGBoost with class imbalance handling
- Hyperparameter tuning: Grid search with 5-fold cross-validation
- Threshold optimization: Adjusted decision boundary for better recall

### Optimizations Implemented

**1. Efficient Hyperparameter Search**
- Replaced manual nested loops with GridSearchCV
- Parallel processing using all CPU cores (n_jobs=-1)
- Cross-validation for more robust parameter selection

**2. Class Imbalance Handling**
- Added class_weight='balanced' to logistic regression
- Used scale_pos_weight in XGBoost
- Optimized for ROC-AUC instead of accuracy

**3. Feature Engineering**
- Created 8 new features based on domain knowledge
- Interaction terms capture non-linear relationships
- Clinical thresholds for high-risk indicators

**4. Model Comparison**
- Logistic Regression: Interpretable baseline
- XGBoost: Captures non-linear patterns and feature interactions
- Threshold optimization: Improves recall on positive cases

**5. Threshold Optimization**
- Adjusted decision threshold from default 0.5
- Targeted 70% recall to detect more CHD cases
- Trade-off between precision and recall based on medical context

## Results

### Actual Model Performance (Test Set)

**Final Logistic Regression Model:**
- **ROC-AUC: 0.699** - Decent discrimination between classes
- **Recall: 56.2%** - Detected 54 out of 96 CHD cases
- **Precision: 25.8%** - 155 false positives (acceptable for screening)
- **Accuracy: 69.0%** - Better than baseline (85% naive accuracy)
- **F1-Score: 0.354**

**Confusion Matrix:**
- True Negatives: 385 (correctly predicted no CHD)
- False Positives: 155 (false alarms - acceptable for screening)
- False Negatives: 42 (missed CHD cases - room for improvement)
- True Positives: 54 (correctly detected CHD)

**XGBoost Comparison:**
- ROC-AUC: 0.637 (performed worse than logistic regression)
- Logistic regression was the better model for this dataset

**Key Achievements:**
- Improved recall from ~5% (baseline) to 56.2% (11x improvement)
- Successfully handled severe class imbalance (85% negative class)
- Created 8 meaningful interaction features
- Optimized hyperparameters using cross-validation

### Key Risk Factors Identified
- Age (strongest predictor)
- Systolic blood pressure
- Hypertension
- Diabetes
- Age-smoking interaction
- Cholesterol-BP interaction

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
xgboost
```

Install with:
```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

## Usage

1. Ensure `framingham.csv` is in the project directory
2. Run all cells in the Jupyter notebook
3. Model will automatically:
   - Engineer features
   - Compare logistic regression vs XGBoost
   - Tune hyperparameters
   - Optimize decision threshold
   - Evaluate on test set

## Key Insights

### Why These Optimizations Matter

**Feature Engineering:** Medical risk factors often interact (e.g., older smokers have exponentially higher risk than young smokers). Interaction features capture these relationships. Added 8 features increased model complexity but improved pattern recognition.

**Class Imbalance Handling:** With 85% negative cases, a naive model can achieve 85% accuracy by always predicting "no CHD". Proper handling (class_weight='balanced') ensures the model actually learns to detect CHD, improving recall from 5% to 56%.

**XGBoost vs Logistic Regression:** Surprisingly, logistic regression (ROC-AUC: 0.699) outperformed XGBoost (ROC-AUC: 0.637). This suggests the relationships in the data are relatively linear, and the interpretability of logistic regression is a bonus.

**Threshold Optimization:** In medical screening, missing a CHD case (false negative) is more costly than a false alarm (false positive). The threshold optimization section allows adjusting the decision boundary to prioritize recall over precision.

### Current Performance Context

**What the numbers mean:**
- **ROC-AUC 0.699**: Model is better than random (0.5) but not excellent (0.8+)
- **56.2% Recall**: Catching about half of CHD cases - significant improvement from 5% baseline
- **25.8% Precision**: For every 4 positive predictions, 3 are false alarms - acceptable for screening
- **69% Accuracy**: Misleading metric due to class imbalance - recall is more important

**Clinical Interpretation:**
- Model successfully identifies high-risk patients for further testing
- False positives lead to additional screening (acceptable cost)
- False negatives (42 missed cases) are the main concern
- Best used as a first-line screening tool, not diagnostic

## Model Limitations

1. **Moderate recall (56.2%)**: Still missing 44% of CHD cases - threshold tuning could improve this
2. **Low precision (25.8%)**: High false positive rate means many unnecessary follow-ups
3. **ROC-AUC of 0.699**: Decent but not excellent - room for improvement with advanced techniques
4. **Limited features**: Dataset doesn't include family history, detailed lifestyle factors, genetic markers
5. **Temporal validation**: Model should be validated on more recent data
6. **Generalization**: Framingham study population may not represent all demographics
7. **XGBoost underperformed**: Tree-based model didn't improve over logistic regression

## Future Improvements

**High-Impact Optimizations (Expected +10-20% performance):**
- **SMOTE** (Synthetic Minority Oversampling): Generate synthetic CHD cases for better balance
  - Expected: +10-15% recall, +3-5% ROC-AUC
- **Ensemble Methods**: Combine multiple models (Voting Classifier)
  - Expected: +3-5% ROC-AUC, more robust predictions
- **Feature Selection**: Remove low-impact features to reduce overfitting
  - Expected: +1-2% ROC-AUC
- **Calibration**: Improve probability estimates for better threshold optimization
  - Expected: +2-3% recall

**Other Potential Enhancements:**
- Polynomial features for non-linear relationships
- RandomizedSearchCV for wider hyperparameter exploration
- Cost-sensitive learning with explicit false negative penalties
- Temporal cross-validation for time-series nature of medical data
- External validation on different populations
- SHAP values for better model interpretability

**Target Performance with Optimizations:**
- ROC-AUC: 0.75-0.80 (currently 0.699)
- Recall: 70-75% (currently 56.2%)
- Better precision-recall balance

## Clinical Context

This model is designed for screening purposes where:
- High recall is prioritized (don't miss CHD cases)
- False positives lead to additional testing (acceptable cost)
- Model assists but doesn't replace clinical judgment
- Threshold can be adjusted based on healthcare setting

**Important:** This is a demonstration project for learning purposes. Any medical application would require:
- Clinical validation studies
- Regulatory approval
- Integration with clinical workflows
- Continuous monitoring and updating
- Ethical review and bias assessment

## Project Structure

```
.
├── Coronary_Heart_Disease_Risk_Prediction.ipynb  # Main analysis notebook
├── framingham.csv                                 # Dataset
├── README.md                                      # This file
└── OPTIMIZATIONS.md                               # Technical optimization details
```

## Author Notes

This project demonstrates:
- Handling imbalanced medical data
- Feature engineering for healthcare applications
- Model comparison and selection
- Threshold optimization for medical screening
- Trade-offs between precision and recall in clinical contexts

The focus is on practical ML engineering skills applicable to real-world healthcare problems.
