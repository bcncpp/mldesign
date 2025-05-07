# Model Monitoring.

## Data Drift
What it is: A change in the distribution of input data over time compared to the training data.
Example:
    You trained a fraud detection model on user behavior from 2023.
    In 2025, users now use different devices or payment methods → input feature distributions (e.g., device_type, transaction_amount) shift.
Impact:
    Model accuracy degrades because it hasn’t seen this “new kind” of data before.
Detection:
    Use statistical tests like KS test, Wasserstein distance, or tools like Evidently, WhyLabs.

## Data Quality / Schema Drift
    What it is: Changes in the structure or quality of input data.

Types:
- Schema Drift: A column is renamed, added, removed, or changes type (e.g., string → int).
- Data Quality Issues: Missing values, invalid formats, outliers.

Example:

- Column user_age was previously an integer, but now it's a string ("unknown", "N/A", etc.).
- A new feature user_location is added but the model doesn’t use it.

Impact:
    Can cause runtime errors or silent bugs (model receives garbage input).

Detection:

    Use data validation tools like Great Expectations, TensorFlow Data Validation (TFDV).

##  Concept Drift

What it is: The relationship between input and target (label) changes over time.

Example:

    In an email spam detector: in 2022, emails with the word “free” were spam; in 2025, marketing emails using “free trial” are legitimate → same input, label meaning changed.

Impact:

    Even if the input data distribution hasn’t changed, model performance degrades because the underlying pattern has shifted.

Detection:

- Monitor model performance metrics over time (accuracy, precision).
- Retrain with recent data and compare against old model (A/B test).

🎯 Model Bias – Explained Clearly

Model bias refers to systematic errors in a machine learning model that cause it to consistently make inaccurate or unfair predictions, especially for certain groups or types of data.
🧠 Types of Bias in ML Models
Type	Description	Example
Historical bias	Bias already present in the data due to societal or systemic inequalities.	A hiring model trained on past data that favors male applicants.
Representation bias	When some groups are underrepresented in the training data.	A facial recognition model performs poorly on darker skin tones.
Measurement bias	When features or labels are inaccurately recorded.	Using credit history as a proxy for trustworthiness.
Algorithmic bias	When the model’s learning process amplifies bias from the data.	A recommender system that over-personalizes and excludes diversity.
⚠️ Why Model Bias Matters

    Legal risk: Discriminatory outcomes can violate regulations (e.g., GDPR, EEOC).

    Ethical concerns: Users lose trust if a system treats them unfairly.

    Business risk: Biased models can alienate customers and damage brand reputation.

🛠️ How to Detect and Mitigate Bias

Detection:

    Use fairness metrics:

        Demographic parity: Equal positive rates across groups.

        Equal opportunity: Equal true positive rates across groups.

        Disparate impact ratio: Ratio of positive predictions across demographics.

    Use tools: Fairlearn, Aequitas, IBM AI Fairness 360.

Mitigation:

    Preprocessing: Balance or reweigh the training data.

    In-processing: Use bias-aware learning algorithms (e.g., adversarial debiasing).

    Post-processing: Calibrate model outputs differently per group to balance outcomes.

📌 Real-World Example

COMPAS Recidivism Model: Used in US courts to predict reoffending risk. It was found to be biased against Black defendants, labeling them as high risk more often than white defendants for the same outcomes.
