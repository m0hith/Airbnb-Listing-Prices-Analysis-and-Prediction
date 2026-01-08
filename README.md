# Airbnb Listing Price Analysis and Prediction

## Project Overview

This project provides comprehensive analysis and predictive modeling for Airbnb listing prices using Seattle Airbnb Open Data. The project combines exploratory data analysis, feature engineering, and multiple machine learning models to predict listing prices and identify key factors influencing pricing decisions.

### Key Objectives

- **Price Prediction**: Build accurate regression models to predict Airbnb listing prices
- **Feature Analysis**: Identify the most important factors influencing listing prices
- **Temporal Analysis**: Understand seasonal and monthly pricing patterns
- **Geographic Insights**: Analyze price variations across different neighborhoods
- **Actionable Insights**: Provide recommendations for hosts and guests

---

## Dataset

**Source**: [Seattle Airbnb Open Data](https://www.kaggle.com/datasets/airbnb/seattle)

### Dataset Components

1. **calendar.csv**: Daily availability and pricing data
   - Contains listing availability and price information for each date
   - ~1.4 million records
   - Key columns: `listing_id`, `date`, `available`, `price`

2. **listings.csv**: Detailed listing information
   - Contains comprehensive property and host information
   - ~3,818 listings
   - 96+ features including property type, amenities, location, host details, reviews

3. **reviews.csv** (Optional): Guest reviews for sentiment analysis
   - Text reviews for sentiment feature engineering
   - Used to extract overall sentiment scores per listing

---

## Technologies & Tools

### Core Libraries
- **Python 3.x**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning models and preprocessing
- **XGBoost**: Gradient boosting for enhanced predictions

### Visualization
- **Plotly**: Interactive visualizations
- **Seaborn**: Statistical visualizations
- **Matplotlib**: Plotting and customization

### Statistical Analysis
- **SciPy**: Statistical tests (ANOVA)
- **TextBlob**: Sentiment analysis (optional)

---

## Project Structure

```
Airbnb-Price-Analysis-and-Prediction/
│
├── Airbnb_Listing_Price_Analysis_and_Prediction.ipynb  # Main analysis notebook
├── README.md                                             # Project documentation
├── requirements.txt                                      # Python dependencies
├── setup_environment.sh                                  # Environment setup script
├── QUICK_START.md                                        # Quick start guide
└── data/                                                # Data directory (not included)
    ├── calendar.csv
    ├── listings.csv
    └── reviews.csv (optional)
```

---

## Getting Started

### Prerequisites

Python 3.8 or higher is required.

### Quick Setup

1. **Run the setup script** (recommended):
   ```bash
   cd /path/to/Airbnb-Price-Analysis-and-Prediction
   ./setup_environment.sh
   ```

2. **Activate environment and start Jupyter**:
   ```bash
   source venv/bin/activate
   jupyter notebook
   ```

### Manual Setup

If the script doesn't work, follow these steps:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install packages
pip install --upgrade pip
pip install -r requirements.txt

# Install NLTK data (for textblob, optional)
python3 -c "import nltk; nltk.download('punkt'); nltk.download('brown'); nltk.download('wordnet')"

# Start Jupyter
jupyter notebook
```

### Data Setup

1. Download the Seattle Airbnb dataset from [Kaggle](https://www.kaggle.com/datasets/airbnb/seattle)
2. Extract the CSV files
3. Create a `data/` directory in the project root
4. Place the following files in `data/`:
   - `calendar.csv`
   - `listings.csv`

### Running the Notebook

1. Open `Airbnb_Listing_Price_Analysis_and_Prediction.ipynb` in Jupyter Notebook or JupyterLab
2. Update the `DATA_PATH` variable in the data loading cell to match your directory structure
3. Run all cells sequentially
4. Review visualizations and model results

### Troubleshooting

**Issue: ModuleNotFoundError**
- Solution: Make sure your virtual environment is activated before installing packages and running Jupyter

**Issue: Jupyter doesn't see your environment**
- Solution: Install ipykernel in your environment:
  ```bash
  pip install ipykernel
  python -m ipykernel install --user --name=airbnb-env
  ```
  Then select the kernel in Jupyter: Kernel → Change Kernel → airbnb-env

**Issue: Data files not found**
- Solution: Update the `DATA_PATH` variable in the notebook to point to your data directory

---

## Methodology

### 1. Data Loading & Exploration
- Load and inspect datasets
- Understand data structure and types
- Identify missing values and data quality issues

### 2. Data Cleaning & Preprocessing
- **Calendar Dataset**:
  - Filter only available listings
  - Clean price columns (remove $ signs, convert to numeric)
  - Handle missing values
  
- **Listings Dataset**:
  - Remove irrelevant columns (URLs, scrape metadata, redundant location data)
  - Clean percentage and dollar columns
  - Handle missing values using median (numeric) and mode (categorical)
  - Standardize data formats

### 3. Exploratory Data Analysis (EDA)

#### Price Distribution Analysis
- Histogram of listing prices
- Statistical summary (mean, median, mode, quartiles)
- Outlier identification

#### Geographic Analysis
- Interactive map of listings by neighborhood
- Price distribution by neighborhood
- Box plots comparing neighborhoods

#### Temporal Analysis
- Average price trends by month
- Listing count by month
- Seasonal patterns
- Price variations by day of week

#### Feature Analysis
- Price by property type
- Price by room type
- Price by number of bedrooms/beds
- Price by accommodation capacity
- Relationship between cleaning fee and price
- Review ratings and price correlation

### 4. Feature Engineering

#### Date Features
- Extract year, month, day of week from date
- Identify seasonal patterns

#### Categorical Simplification
- Simplify property types (e.g., merge similar categories)
- Simplify bed types
- Drop redundant neighborhood columns

#### Amenities Features
- Extract binary features for key amenities:
  - TV, Internet, Air Conditioning, Heating
  - Kitchen, Family/Kid Friendly
  - Washer, Dryer
  - Pets Allowed, Free Parking
  - Hot Tub, Essentials

#### Sentiment Analysis (Optional)
- Extract sentiment scores from reviews using TextBlob
- Calculate overall sentiment per listing
- Merge sentiment features

### 5. Model Development

#### Preprocessing Pipeline
- **Numeric Features**: StandardScaler (standardization)
- **Categorical Features**: OrdinalEncoder (encoding)
- **Pipeline**: ColumnTransformer for mixed data types

#### Models Implemented

1. **Linear Regression**
   - Baseline model
   - Interpretable coefficients
   - Fast training and prediction

2. **Decision Tree Regressor**
   - Non-linear relationships
   - Feature importance analysis
   - Hyperparameter tuning (max_depth, max_features, max_leaf_nodes)

3. **Random Forest Regressor**
   - Ensemble method
   - Robust to overfitting
   - Feature importance ranking

4. **XGBoost Regressor**
   - Gradient boosting
   - High predictive performance
   - Advanced hyperparameter tuning

#### Model Evaluation Metrics

- **MSE (Mean Squared Error)**: Average squared differences
- **RMSE (Root Mean Squared Error)**: Square root of MSE (in original units)
- **MAE (Mean Absolute Error)**: Average absolute differences (robust to outliers)
- **R² (R-Squared)**: Proportion of variance explained (0-1, higher is better)

#### Feature Selection
- SelectKBest with f_regression
- Top K features based on statistical significance
- Comparison with and without feature selection

#### Cross-Validation
- K-fold cross-validation for robust performance estimation
- Prevents overfitting assessment

### 6. Model Comparison & Selection

- Side-by-side comparison of all models
- Box plots of residual sum of squares (RSS)
- Metrics comparison table
- Feature importance visualization
- Best model selection based on multiple metrics

---

## Key Findings

### Price Distribution
- Most listings priced between **$50-$199**
- Median price: **$100**
- Mode price: **$150**
- Price range: $10 - $1,650

### Geographic Insights
- **Top 3 neighborhoods** by median price:
  1. Downtown
  2. Magnolia
  3. Queen Anne

### Temporal Patterns
- **Peak pricing months**: May through September
- **Yearly average**: ~$138
- **Highest demand**: March and December (most listings)
- **Lower availability**: June through August (higher prices)

### Property Features Impact
- **Property Type**: Boats, Condominiums, Apartments have higher prices
- **Room Type**: Entire Home > Private Room > Shared Room
- **Bedrooms**: More bedrooms → Higher prices
- **Accommodation**: Higher capacity → Higher prices
- **Cleaning Fee**: Positive correlation with listing price

### Host Characteristics
- Superhost status shows slight price premium
- Response rate and acceptance rate have minimal direct impact

### Amenities Impact
- Key amenities (TV, Internet, Kitchen, etc.) influence pricing
- Family-friendly properties may command premium

---

## Model Performance

### Best Performing Models

Based on comprehensive evaluation:

1. **Decision Tree Regressor**
   - R²: ~0.834
   - RMSE: ~$42.69
   - MAE: ~$28.10
   - Excellent balance of performance and interpretability

2. **XGBoost Regressor**
   - R²: ~0.822
   - RMSE: ~$44.26
   - MAE: ~$27.81
   - Strong predictive power

3. **Random Forest Regressor**
   - R²: ~0.819
   - RMSE: ~$44.63
   - MAE: ~$29.32
   - Robust ensemble method

4. **Linear Regression**
   - R²: ~0.652
   - RMSE: ~$61.77
   - MAE: ~$37.44
   - Baseline model, most interpretable

### Feature Importance

Top features influencing price (from tree-based models):
- Accommodation capacity
- Number of bedrooms/beds
- Neighborhood
- Cleaning fee
- Property type
- Room type
- Monthly/weekly pricing
- Selected amenities

---

## Code Structure

### Main Sections

1. **Imports & Setup**: Library imports and configuration
2. **Data Loading**: Load and inspect datasets
3. **Data Cleaning**: Clean and preprocess data
4. **EDA**: Exploratory data analysis with visualizations
5. **Feature Engineering**: Create and transform features
6. **Modeling**: Train and evaluate multiple models
7. **Model Comparison**: Compare model performance
8. **Optional: Sentiment Analysis**: Add review sentiment features

### Key Functions

- `print_scores()`: Display model evaluation metrics
- `model_pipeline()`: Create and train model pipeline
- Feature importance visualization functions

---

## Statistical Analysis

### ANOVA Tests

- **Neighborhood vs Price**: Significant differences (p < 0.05)
- **Host Listings Count vs Price**: Significant differences (p < 0.05)

These tests confirm that categorical features have statistically significant impact on pricing.

---

## Insights & Recommendations

### For Hosts

1. **Pricing Strategy**:
   - Consider seasonal pricing: Increase prices during May-September
   - Downtown, Magnolia, and Queen Anne command premium prices

2. **Property Optimization**:
   - More bedrooms and higher accommodation capacity justify higher prices
   - Entire home listings have significant price advantage
   - Key amenities (TV, Internet, Kitchen) are important

3. **Host Profile**:
   - Superhost status provides slight premium
   - Maintain high response and acceptance rates

### For Guests

1. **Best Value**:
   - Consider booking outside peak months (May-September)
   - Private rooms offer better value than entire homes
   - Check neighborhoods for price variations

2. **Amenities**:
   - Prioritize listings with essential amenities
   - Family-friendly properties may cost more

---

## Visualizations

The notebook includes comprehensive visualizations:

- **Distribution Plots**: Price histograms and distributions
- **Geographic Maps**: Interactive neighborhood maps with price overlays
- **Box Plots**: Price comparisons by categories
- **Line Charts**: Temporal trends and patterns
- **Scatter Plots**: Feature relationships
- **Correlation Heatmaps**: Feature correlations
- **Feature Importance**: Model feature rankings

All visualizations are interactive (Plotly) for enhanced exploration.

---

## Customization

- **Data Path**: Modify `DATA_PATH` variable in the data loading section
- **Model Parameters**: Adjust hyperparameters in the modeling section
- **Feature Selection**: Modify `k` value in SelectKBest
- **Train/Test Split**: Adjust `test_size` parameter (default: 0.3)

---

## Future Enhancements

### Potential Improvements

1. **Advanced Feature Engineering**:
   - Distance to city center/landmarks
   - Walkability scores
   - Public transportation proximity
   - Crime statistics by neighborhood

2. **Model Improvements**:
   - Neural networks for complex patterns
   - Ensemble methods combining multiple models
   - Time series models for temporal patterns
   - Deep learning for text features (descriptions)

3. **Additional Analysis**:
   - Price elasticity analysis
   - Competitive pricing analysis
   - Demand forecasting
   - Revenue optimization

4. **Deployment**:
   - Create API for real-time predictions
   - Build interactive dashboard
   - Deploy model for production use

---

## References

- [Seattle Airbnb Open Data](https://www.kaggle.com/datasets/airbnb/seattle)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Plotly Documentation](https://plotly.com/python/)

---

## Acknowledgments

- Airbnb for providing the open dataset
- Seattle data community
- Open-source contributors to the libraries used
