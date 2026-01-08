# Airbnb Listing Price Analysis and Prediction - Master Project Document

> **Document Status**: Active
> **Last Updated**: January 2026
> **Purpose**: This document serves as the Single Source of Truth (SSOT) for the Airbnb Listing Price Analysis project. It details the architectural decisions, methodologies, experimental results, and future roadmap to facilitate maintenance and future AI-driven enhancements.

---

## 1. Executive Summary

This project implements a data science pipeline to analyze and predict Airbnb listing prices in Seattle using the "Seattle Airbnb Open Data" set. By leveraging machine learning regression techniques (Linear Regression, Decision Trees, Random Forests, XGBoost), the system identifies key pricing determinants and provides actionable insights for hosts to optimize revenue. The final model achieves an R² of approximately 0.834, utilizing features such as accommodation capacity, location, and amenities.

---

## 2. Project Definition & Objectives

### 2.1 Problem Statement
Hosts on Airbnb struggle to set optimal prices that balance competitiveness with revenue maximization. Without data-driven insights, pricing strategies are often arbitrary or reactive. This project aims to solve this by modeling the relationship between listing attributes and price.

### 2.2 Core Objectives
1.  **Predictive Modeling**: Develop a regression model to forecast listing prices with high accuracy (Target: R² > 0.80).
2.  **Feature Attribution**: Isolate and rank the most significant variables affecting price (e.g., location vs. amenities).
3.  **Market Analysis**: Characterize the Seattle Airbnb market through temporal (seasonal) and spatial (neighborhood) analysis.
4.  **Operational Guidance**: Translate model findings into concrete recommendations for listing optimization.

---

## 3. Data Architecture & Specifications

**Source**: [Kaggle - Seattle Airbnb Open Data](https://www.kaggle.com/datasets/airbnb/seattle)

### 3.1 Data Schema
The analysis relies on three primary data sources:

| File Name | Records (approx) | Description | Key Features |
| :--- | :--- | :--- | :--- |
| `calendar.csv` | 1.4M | Temporal availability and pricing per listing. | `listing_id`, `date`, `available`, `price` |
| `listings.csv` | 3,818 | Static listing metadata, host details, and location. | `id`, `neighbourhood`, `property_type`, `amenities`, `price` |
| `reviews.csv` | Optional | Unstructured text reviews from guests. | `listing_id`, `comments` (Used for sentiment extraction) |

### 3.2 Data Quality & Preprocessing Rules
*   **Monetary Conversion**: All price fields (e.g., `$100.00`) are sanitized to numeric floats.
*   **Missing Value Imputation**:
    *   *Numeric*: Imputed with Median to resist outliers.
    *   *Categorical*: Imputed with Mode (most frequent).
*   **Filtering**:
    *   Calendar data is filtered to include only "available" dates for price trend analysis.
    *   Irrelevant columns (URLs, scrape metadata) are dropped to reduce dimensionality.

---

## 4. Technical Architecture

### 4.1 Technology Stack
*   **Language**: Python 3.8+
*   **ETL & Analysis**: Pandas (DataFrames), NumPy (Vectorization).
*   **Machine Learning**: Scikit-learn (Pipeline, Preprocessing, Models), XGBoost (Gradient Boosting).
*   **Visualization**: Plotly (Interactive), Seaborn/Matplotlib (Static/Statistical).
*   **Statistical Tools**: SciPy (ANOVA), TextBlob (NLP/Sentiment).

### 4.2 Project Directory Structure
```
/
├── Airbnb_Listing_Price_Analysis_and_Prediction.ipynb  # Core Logic / Notebook
├── data/                                                # Raw Data Storage
│   ├── calendar.csv
│   ├── listings.csv
│   └── reviews.csv
├── venv/                                                # Virtual Environment
├── requirements.txt                                     # Dependency Lockfile
└── MASTER.md                                            # Project Spec (This Document)
```

---

## 5. Implementation Methodology

### 5.1 Phase 1: Exploratory Data Analysis (EDA)
Comprehensive profiling was conducted to understand distributions and correlations:
*   **Price Distribution**: Validated a right-skewed distribution; most listings fall in the $50-$199 range.
*   **Spatial Analysis**: Identified "Downtown", "Magnolia", and "Queen Anne" as high-value clusters.
*   **Temporal Analysis**: Mapped seasonal trends, revealing peak pricing windows in Summer (May-Sept) and demand spikes in Dec/March.

### 5.2 Phase 2: Feature Engineering
Raw data was transformed into machine-readable features:
*   **Temporal Extraction**: Decomposed `date` into `month`, `year`, `day_of_week`.
*   **Text Mining**: Parsed `amenities` string lists into binary presence indicators (One-Hot style for "TV", "Internet", etc.).
*   **Sentiment Analysis**: (Optional) Aggregated polarity scores from `reviews.csv` to quantify guest satisfaction.
*   **Dimensionality Reduction**: Grouped sparse categories (e.g., rare property types) to prevent overfitting.

### 5.3 Phase 3: Model Development Pipeline
A unified `scikit-learn` pipeline was constructed to ensure reproducibility and prevent data leakage:
1.  **Preprocessor**: `ColumnTransformer` applying `StandardScaler` to numerics and `OrdinalEncoder` to categoricals.
2.  **Feature Selection**: `SelectKBest` (f_regression) to filter noise.
3.  **Estimator**: Multiple regression algorithms were trained and cross-validated.

### 5.4 Phase 4: Models Evaluated
1.  **Linear Regression**: Baseline for interpretability.
2.  **Decision Tree Regressor**: Captures non-linear splits.
3.  **Random Forest Regressor**: Bagging ensemble for variance reduction.
4.  **XGBoost Regressor**: Boosting ensemble for bias reduction and performance.

---

## 6. Results & Evaluation

### 6.1 Performance Metrics
Models were evaluated on unseen test data (30% split) using R², RMSE, and MAE.

| Model | R² (Variance Explained) | RMSE (Error) | MAE (Avg Error) | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Decision Tree** | **0.834** | **$42.69** | **$28.10** | **Best Performance**. Captures complex interactions well. |
| XGBoost | 0.822 | $44.26 | $27.81 | Strong runner-up, robust generalization. |
| Random Forest | 0.819 | $44.63 | $29.32 | Stable, but slightly less accurate than single tree here. |
| Linear Regression | 0.652 | $61.77 | $37.44 | Underfitting; implies non-linear price relationships. |

### 6.2 Key Driver Analysis (Feature Importance)
The following features were statistically most significant in predicting price:
1.  **Accommodation Capacity**: Strongest positive correlation.
2.  **Room Type**: "Entire home/apt" commands significant premium over private/shared rooms.
3.  **Neighborhood**: Specific high-demand zip codes act as price multipliers.
4.  **Cleaning Fee**: High correlation with base price (proxy for luxury/size).
5.  **Amenities**: Presence of Kitchen, TV, and Internet acts as a baseline requirement.

### 6.3 Statistical Validation
ANOVA tests confirmed significant variance ($p < 0.05$) in price groups based on "Neighborhood" and "Host Listings Count", validating the inclusion of these categorical features.

---

## 7. Operational Manual

### 7.1 Setup & Execution
1.  **Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Data Placement**: Ensure `calendar.csv`, `listings.csv`, and `reviews.csv` are in `./data/`.
3.  **Execution**: Launch Jupyter and run `Airbnb_Listing_Price_Analysis_and_Prediction.ipynb`.

### 7.2 Troubleshooting
*   **Missing NLTK Data**: Run `python3 -c "import nltk; nltk.download('punkt')"` if text processing fails.
*   **Path Errors**: Verify `DATA_PATH` constant in the notebook matches local structure.

---

## 8. Future Roadmap & AI Context

This section outlines areas for future development and context for AI agents working on this codebase.

### 8.1 Technical Improvements
*   **Modularization**: Refactor the monolithic Jupyter Notebook into Python modules (`src/data_loader.py`, `src/model.py`) for production readiness.
*   **Testing**: Implement `pytest` suite for data validation and pipeline integrity.
*   **Reproducibility**: Dockerize the application environment.

### 8.2 Analytical Enhancements
*   **NLP Deep Dive**: Utilize BERT or TF-IDF on listing descriptions to extract "luxury" keywords.
*   **Geo-Spatial**: Integrate external APIs (Google Maps) to calculate "Walkability" or "Distance to Landmark" scores.
*   **Image Analysis**: Use CNNs on listing photos to quantify visual appeal (not currently implemented).

### 8.3 Deployment
*   **API**: Wrap the best model (Decision Tree/XGBoost) in a FastAPI/Flask endpoint.
*   **Dashboard**: Build a Streamlit app for users to input listing details and get a price recommendation.

---
*End of Master Document*
