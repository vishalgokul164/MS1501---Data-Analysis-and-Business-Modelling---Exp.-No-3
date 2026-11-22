# EXPERIMENT NO.3 : REGRESSION TECHNIQUES IN BUSINESS DATA (SIMPLE LINEAR REGRESSION)

##  AIM

To implement Simple Linear Regression in Python to study and predict the relationship between two business-related variables.

## DATASET

A sample business dataset representing the relationship between **Advertising Expenditure** (in Lakhs) and **Sales Revenue** (in Crores). The goal is to analyze how changes in advertising investment affect sales performance.

| Index | Advertising (Lakhs) | Sales (Crores) |
| :---: | :---: | :---: |
| 0 | 18 | 12 |
| 1 | 22 | 18 |
| 2 | 25 | 20 |
| 3 | 30 | 24 |
| 4 | 35 | 28 |
| 5 | 40 | 35 |
| 6 | 45 | 38 |
| 7 | 50 | 40 |
| 8 | 55 | 42 |
| 9 | 60 | 45 |

## ALGORITHM / STEPS FOLLOWED
1. Import required Python libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. Create and load the dataset into a Pandas DataFrame.
3. Display the dataset.
4. Select independent ($X$) and dependent ($Y$) variables and reshape $X$ into a 2D array.
5. Build a `LinearRegression` model using `sklearn.linear_model`.
6. Fit the model and calculate the intercept ($a$) and slope ($b$).
7. Predict $Y$ values for given $X$ values and evaluate performance using the $R^2$ score.
8. Plot the actual data points alongside the fitted regression line and predict sales for a new advertising budget.

## FUNCTIONS AND METHODS USED

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Creation / Handling** | `np.array()` | Create numerical arrays for regression input data |
| | `pd.DataFrame()` | Create or manage tabular data for regression |
| **Model Building (Scikit-learn)** | `LinearRegression()` | Initialize the Simple Linear Regression model |
| | `model.fit()` | Train the linear regression model using $X$ and $Y$ values |
| | `model.predict()` | Predict $Y$ values for given $X$ inputs using the fitted model |
| | `model.score()` | Calculate model performance ($R^2$ score) |
| **Visualization (Matplotlib)** | `plt.scatter()` | Plot scatter points (actual data points) |
| | `plt.plot()` | Plot regression line (predicted relationship) |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.show()` | Configure and display the regression plot |

##  PROGRAM AND OUTPUT

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
data = {  
    'Advertising': [18, 22, 25, 30, 35, 40, 45, 50, 55, 60],  
    'Sales': [12, 18, 20, 24, 28, 35, 38, 40, 42, 45]  
}  
df = pd.DataFrame(data)  
print("--- 1. Dataset Loaded Successfully ---")  
print(df)  
X = df[['Advertising']]  
y = df['Sales']  
model = LinearRegression()  
model.fit(X, y)  
intercept = model.intercept_  
slope = model.coef_[0]  
print("\n--- 2. Model Parameters ---")  
print(f"Intercept (a): {intercept:.4f}")  
print(f"Slope (b): {slope:.4f}")  
print(f"Regression Equation: Sales = {slope:.4f} * Advertising + {intercept:.4f}")  
y_pred = model.predict(X)  
r2 = model.score(X, y)  
print("\n--- 3. Model Performance ---")  
print(f"R-squared (R^2) Score: {r2:.4f}")  
new_budget = np.array([[50]])  
predicted_sales = model.predict(new_budget)  
print(f"\nPredicted Sales for an Advertising budget of 50 Lakhs: {predicted_sales[0]:.2f} Crores")  
plt.figure(figsize=(8, 5))  
plt.scatter(X, y, color='blue', label='Actual Data Points', s=70)  
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')  
plt.title('Simple Linear Regression: Advertising Expenditure vs Sales Revenue')  
plt.xlabel('Advertising Expenditure (Lakhs)')  
plt.ylabel('Sales Revenue (Crores)')  
plt.legend()  
plt.grid(True, linestyle='--', alpha=0.6)  
plt.tight_layout()  
plt.show()  

<img width="475" height="321" alt="image" src="https://github.com/user-attachments/assets/43a3a22f-30fb-4575-92bf-94850804f816" />
<img width="738" height="451" alt="image" src="https://github.com/user-attachments/assets/2d770937-957b-440c-b117-b47435215c19" />


##  RESULT

Advertising strongly drives sales, with the model explaining 97.36% of sales variance ($R^2 = 0.9736$).An advertising spend of 50 Lakhs is projected to generate approximately 39.55 Crores in sales.
