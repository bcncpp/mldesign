
# SHAP Explainability with Logistic Regression

## What is SHAP?

SHAP (SHapley Additive exPlanations) is a powerful technique for model explainability. It helps in understanding which features are most important for a model’s predictions. SHAP is based on Shapley values from cooperative game theory, which assign a value to each feature, showing how much it contributes to the model's prediction.

### Is SHAP Online or Offline?

- **Offline**: SHAP is typically used offline to explain model predictions after they have been made. It’s commonly used post-hoc (after training).
- **Online (Real-time)**: While less common in real-time systems, SHAP can be integrated into online pipelines, though it may introduce computational overhead.

## Example: Logistic Regression with SHAP Explainability

We will use a simple **Logistic Regression** model and walk through **SHAP** explainability for a binary classification problem using the **Breast Cancer dataset**.

### Step 1: Import Necessary Libraries

```python
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
```

### Step 2: Load the Dataset

We’ll use the **Breast Cancer dataset** for this classification task.

```python
# Load the Breast Cancer dataset (binary classification)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
```

### Step 3: Split the Data

We will split the data into training and testing sets.

```python
# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Step 4: Train a Logistic Regression Model

Now, we train a Logistic Regression model.

```python
# Initialize and train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
```

### Step 5: SHAP Explainability

Now that the model is trained, we use SHAP to explain the model’s predictions. SHAP gives us the feature importances for each prediction made by the model.

```python
# Initialize the SHAP explainer using the model and training data
explainer = shap.Explainer(model, X_train)

# Get SHAP values for the test data
shap_values = explainer(X_test)
```

### Step 6: Visualizing SHAP Values

We can use SHAP’s built-in plots to visualize feature importance. The most common ones are:

- **SHAP Summary Plot**: Shows the distribution of SHAP values for each feature.
- **SHAP Bar Plot**: Shows the average SHAP value (feature importance).
- **SHAP Force Plot**: Visualizes how each feature contributes to individual predictions.

#### Summary Plot (Shows Overall Feature Importance)

```python
# Visualize the summary plot
shap.summary_plot(shap_values, X_test)
```

This plot will display each feature's contribution to the model's predictions. Features with the most significant SHAP values are considered the most important.

#### Force Plot (Shows Individual Prediction Explanations)

```python
# Visualize the SHAP values for a specific prediction (for the first instance in the test set)
shap.force_plot(shap_values[0], X_test.iloc[0])
```

This plot provides an intuitive view of how each feature influences the prediction for a specific instance.

---

## Key Takeaways

- **SHAP values** help explain how much each feature contributes to a model’s prediction.
- The **Summary Plot** gives an overview of which features are important across all predictions.
- The **Force Plot** provides a detailed view of feature contributions for a single prediction.
- For **Logistic Regression** models, SHAP values can show the positive or negative impact of features on the predicted outcome.

Interpreting SHAP plots helps us understand how individual features contribute to the predictions made by the model. Below, I will explain how to interpret the most common SHAP plots:
1. SHAP Summary Plot

The summary plot provides a high-level view of feature importance and how each feature influences the model's predictions.
Interpretation:

    Y-axis: Features, ordered by their average importance. The features at the top are the most important for the model’s predictions.

    X-axis: The SHAP value, representing the magnitude of influence. It shows how much each feature contributes to the output.

    Color: Each point in the plot is color-coded by the feature value. Typically, blue represents low values of the feature and red represents high values.

    Distribution: The spread of the SHAP values indicates how much a feature’s effect varies across the dataset.

        If the spread is wide, the feature has varying levels of impact for different data points.

        If the spread is narrow, the feature has a more consistent effect across the dataset.

Example:

    If mean radius (a feature from the breast cancer dataset) is at the top and has a wide spread from negative to positive SHAP values, it means that mean radius is the most important feature, and its influence on the prediction changes significantly for different instances.

# Example summary plot
shap.summary_plot(shap_values, X_test)

2. SHAP Bar Plot

The bar plot shows the average absolute SHAP values for each feature, which reflects the importance of each feature.
Interpretation:

    Y-axis: Features, ordered by their average importance.

    X-axis: The average magnitude of the SHAP value for each feature.

    Longer bars mean that the feature has a greater impact on the predictions.

    The bar plot gives a global view of which features are most important across all predictions.

Example:

    If the bar for mean radius is longer than others, it tells us that mean radius has the largest overall effect on predictions across the entire dataset.

# Example bar plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

3. SHAP Force Plot

A force plot explains the contribution of each feature for an individual prediction.
Interpretation:

    Base value (or expected value): This is the model's prediction if no features were included (typically the average output of the model).

    SHAP value of each feature: Each feature’s contribution is shown as a "force" either pushing the prediction higher (positive SHAP values) or lower (negative SHAP values).

    Color: Red typically indicates high values of the feature (positive impact), while blue indicates low values (negative impact).

    Width of the arrow: The wider the arrow, the larger the impact of that feature on the prediction.

Example:

    If a force plot shows that mean radius has a strong positive SHAP value (wide red arrow) for a specific instance, it means that mean radius contributed significantly to increasing the predicted probability of the positive class (e.g., cancer diagnosis).

# Example force plot for a single prediction
shap.force_plot(shap_values[0], X_test.iloc[0])

4. SHAP Dependence Plot

The dependence plot shows how a single feature affects the model's prediction across different values of that feature and how it interacts with other features.
Interpretation:

    X-axis: The values of the feature being analyzed.

    Y-axis: The SHAP values (contribution of that feature to the model’s prediction).

    Color: Indicates the value of the interacting feature, showing how other features interact with the feature being analyzed.

Example:

    If you're looking at mean radius and the SHAP value increases as mean radius increases, it means that higher mean radius values tend to push the prediction towards a higher class (e.g., cancerous).

# Example dependence plot
shap.dependence_plot("mean radius", shap_values, X_test)

Key Insights from SHAP Plots:

    Feature Importance:

        The summary plot and bar plot give an overall view of which features are most important for the model’s predictions.

        Features with higher absolute SHAP values have more influence on the model’s decision-making process.

    Feature Interactions:

        The dependence plot shows how features interact with one another and how one feature’s value may affect the impact of another feature on the prediction.

        Features that are highly correlated will often show interaction effects.

    Individual Predictions:

        The force plot helps interpret individual predictions, showing how much each feature contributes to the model’s final prediction compared to the base value (average prediction).

Example Interpretation (Breast Cancer Dataset):

    Feature Importance: Features like mean radius, mean texture, and mean smoothness might show up as highly important, meaning they contribute strongly to the prediction (whether a tumor is malignant or benign).

    Individual Prediction: For an individual test instance, the force plot may show that mean radius has a large positive contribution, pushing the model’s output towards predicting "malignant" (positive class).

    Feature Interaction: A dependence plot of mean radius might show that as the value of mean radius increases, the SHAP value increases (which means the model is more likely to predict "malignant").