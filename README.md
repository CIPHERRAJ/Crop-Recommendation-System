# Crop Recommendation System

A machine learning-based system that recommends suitable crops based on soil parameters and climate conditions. The system achieves 99.32% accuracy on test data using an optimized Random Forest classifier.

## Overview

This Crop Recommendation System analyzes soil composition (Nitrogen, Phosphorus, Potassium, pH) and climate data (temperature, humidity, rainfall) to suggest the most suitable crop for a given agricultural setting. The system employs multiple machine learning algorithms to provide accurate recommendations.

## Features

- **Data Analysis**: Comprehensive exploratory data analysis of crop requirements
- **Multiple Models**: Trains and compares Random Forest, SVM, and KNN classifiers
- **Hyperparameter Tuning**: Optimizes model parameters for maximum accuracy
- **Feature Importance**: Identifies the most influential factors for crop selection
- **Interactive Interface**: User-friendly command-line interface for recommendations
- **Visualization**: Generates insightful plots and visualizations of the data

## Requirements

- Python 3.6+
- Required libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - pickle

You can install all dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset

The system uses a crop recommendation dataset (`crop_recommendation.csv`) that contains:

- **Soil parameters**: N (Nitrogen), P (Phosphorus), K (Potassium), pH
- **Climate parameters**: Temperature (°C), Humidity (%), Rainfall (mm)
- **Target variable**: Crop label (22 different crops including rice, maize, chickpea, kidney beans, cotton, etc.)

The dataset contains 2,200 records, with 100 records for each crop type, providing a balanced distribution for model training.

## Usage

### Running the System

```bash
python crop_recommendation.py
```

### Interactive Mode

After training, the system allows you to input your own soil and climate parameters:

1. Enter Nitrogen content (kg/ha)
2. Enter Phosphorous content (kg/ha)
3. Enter Potassium content (kg/ha)
4. Enter soil pH value (0-14)
5. Enter temperature (°C)
6. Enter humidity (%)
7. Enter rainfall (mm)

The system will then recommend the most suitable crop for the given conditions.

## Model Performance

The system achieves excellent performance:

- **Accuracy**: 99.32% on test data
- **Cross-Validation Accuracy**: 99.49% 
- **Best Model**: Random Forest with optimized parameters:
  - n_estimators: 200
  - max_depth: None
  - min_samples_split: 2

### Feature Importance

Analysis reveals the following feature importance ranking:
1. Rainfall (22.71%)
2. Humidity (20.99%)
3. Potassium (18.12%)
4. Phosphorus (14.33%)
5. Nitrogen (10.76%)
6. Temperature (7.67%)
7. pH (5.43%)

This indicates that climatic factors (especially rainfall and humidity) have the strongest influence on crop selection.

## Output Files

### Visualizations
The system generates multiple visualization files in the `visualizations` directory:
- Correlation heatmap: Shows relationships between soil and climate parameters
- Feature distributions by crop: Box plots showing how different crops relate to each parameter
- Confusion matrix: Visual representation of model prediction accuracy
- Feature importance plot: Visualizes the importance ranking of each parameter

### Model
The trained model is saved in the `models` directory as `crop_recommendation_model.pkl`. This file contains both the trained Random Forest model and the scaler used for preprocessing, allowing for direct use in predictions without retraining.

## Important Notes

1. **Warning System**: The system will warn you if your input values fall outside the typical ranges, but will still provide a recommendation.

2. **Potential Warning**: When using the system, you may see a warning message: "X does not have valid feature names, but StandardScaler was fitted with feature names". This is an informational warning from scikit-learn and does not affect the prediction accuracy.

3. **Supported Crops**: The system can recommend any of the following 22 crops:
   - Grains: rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil
   - Fruits: apple, banana, mango, grapes, orange, papaya, coconut, pomegranate, watermelon, muskmelon
   - Commercial: cotton, jute, coffee

## Contributing

Feel free to submit pull requests or suggest improvements. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open-source and available under the MIT License.
