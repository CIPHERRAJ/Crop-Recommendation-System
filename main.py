# Crop Recommendation System
# Designed to recommend suitable crops based on soil and climate conditions

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# Create directories if they don't exist
os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Load Dataset
def load_data(filepath='data/crop_recommendation.csv'):
    """
    Load crop recommendation dataset and display basic information
    """
    print("Loading and exploring dataset...\n")
    data = pd.read_csv(filepath)
    print(f"Dataset shape: {data.shape}")
    print("\nFirst 5 rows of dataset:")
    print(data.head())
    print("\nStatistical summary:")
    print(data.describe())
    print("\nCrop distribution:")
    print(data['label'].value_counts())
    
    return data

# 2. Exploratory Data Analysis
def perform_eda(data):
    """
    Perform exploratory data analysis on the dataset
    """
    print("\nPerforming Exploratory Data Analysis...\n")
    
    # Check for missing values
    print("Missing values:")
    print(data.isnull().sum())
    
    # Correlation heatmap - FIXED: exclude categorical 'label' column
    plt.figure(figsize=(10, 8))
    # Only include numerical columns for correlation
    numerical_data = data.drop('label', axis=1)
    sns.heatmap(numerical_data.corr(), annot=True, cmap='viridis')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()
    print("Correlation heatmap saved as 'visualizations/correlation_heatmap.png'")
    
    # Distribution of each feature by crop - use try/except to handle any plotting errors
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    for feature in features:
        try:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='label', y=feature, data=data)
            plt.title(f'Distribution of {feature} by crop')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'visualizations/{feature}_distribution.png')
            plt.close()
            print(f"Saved {feature} distribution plot")
        except Exception as e:
            print(f"Error creating plot for {feature}: {str(e)}")
    
    print("EDA completed. Visualization files saved in 'visualizations' directory.")

# 3. Data Preprocessing
def preprocess_data(data):
    """
    Preprocess data for model training
    """
    print("\nPreprocessing data...\n")
    
    # Split features and target
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# 4. Model Training
def train_models(X_train, y_train):
    """
    Train multiple models and return the best one
    """
    print("\nTraining models...\n")
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    # Store results
    best_accuracy = 0
    best_model = None
    best_model_name = ""
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_cv_score = np.mean(cv_scores)
        
        print(f"{name} - Cross-Validation Accuracy: {mean_cv_score:.4f}")
        
        if mean_cv_score > best_accuracy:
            best_accuracy = mean_cv_score
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Fine-tune the best model
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Optimized {best_model_name} - Best parameters: {grid_search.best_params_}")
    
    return best_model

# 5. Model Evaluation
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data
    """
    print("\nEvaluating model performance...\n")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    try:
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test, y_pred)
        # Get unique classes for labels
        classes = np.unique(y_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png')
        plt.close()
        print("Confusion matrix saved as 'visualizations/confusion_matrix.png'")
    except Exception as e:
        print(f"Error creating confusion matrix plot: {str(e)}")

# 6. Feature Importance Analysis
def analyze_feature_importance(model, feature_names):
    """
    Analyze and visualize feature importance
    """
    # Only works for models that have feature_importances_ attribute (like Random Forest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        try:
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance')
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png')
            plt.close()
            print("\nFeature Importance visualization saved as 'visualizations/feature_importance.png'")
        except Exception as e:
            print(f"Error creating feature importance plot: {str(e)}")
        
        print("\nFeature Importance:")
        for i in indices:
            print(f"{feature_names[i]}: {importances[i]:.4f}")

# 7. Save Model
def save_model(model, scaler, filename='models/crop_recommendation_model.pkl'):
    """
    Save trained model and scaler for future use
    """
    print(f"\nSaving model as {filename}...")
    with open(filename, 'wb') as file:
        pickle.dump({'model': model, 'scaler': scaler}, file)
    print("Model saved successfully!")

# 8. Crop Recommendation Function
def recommend_crop(model, scaler, N, P, K, temperature, humidity, ph, rainfall):
    """
    Recommend a crop based on input parameters
    """
    # Prepare input data
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale the input
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    crop = model.predict(input_data_scaled)[0]
    
    return crop

# 9. Interactive Function
def get_user_input():
    """Get soil and climate parameters from user"""
    print("\n--- Crop Recommendation System ---\n")
    
    try:
        # Get soil parameters
        n = float(input("Enter Nitrogen content in soil (kg/ha): "))
        p = float(input("Enter Phosphorous content in soil (kg/ha): "))
        k = float(input("Enter Potassium content in soil (kg/ha): "))
        ph = float(input("Enter soil pH value (0-14): "))
        
        # Get climate parameters
        temp = float(input("Enter temperature (°C): "))
        humidity = float(input("Enter humidity (%): "))
        rainfall = float(input("Enter rainfall (mm): "))
        
        return n, p, k, temp, humidity, ph, rainfall
    except ValueError:
        print("Error: Please enter valid numerical values.")
        return get_user_input()

def interactive_recommendation(model, scaler):
    """Interactive mode for crop recommendation"""
    while True:
        n, p, k, temp, humidity, ph, rainfall = get_user_input()
        
        # Basic input validation
        if not (0 <= n <= 140 and 5 <= p <= 145 and 5 <= k <= 205 and 
                5 <= temp <= 45 and 0 <= humidity <= 100 and 
                3 <= ph <= 10 and 20 <= rainfall <= 300):
            print("Warning: Some values are outside the typical ranges. Results may be less reliable.")
        
        crop = recommend_crop(model, scaler, n, p, k, temp, humidity, ph, rainfall)
        print(f"\nRecommended crop: {crop}")
        
        again = input("\nWould you like to try another recommendation? (y/n): ")
        if again.lower() != 'y':
            break

# Main function to execute the pipeline
def main():
    """
    Execute the complete crop recommendation system pipeline
    """
    # Step 1: Load data
    data = load_data()
    
    # Step 2: Perform EDA
    perform_eda(data)
    
    # Step 3: Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Step 4: Train models
    best_model = train_models(X_train, y_train)
    
    # Step 5: Evaluate model
    evaluate_model(best_model, X_test, y_test)
    
    # Step 6: Analyze feature importance
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    analyze_feature_importance(best_model, feature_names)
    
    # Step 7: Save model
    save_model(best_model, scaler)
    
    # Step 8: Demo recommendation
    print("\nDemo Crop Recommendation:")
    sample_N, sample_P, sample_K = 90, 40, 40
    sample_temp, sample_humidity = 20, 80
    sample_ph, sample_rainfall = 6.5, 200
    
    recommended_crop = recommend_crop(
        best_model, scaler, 
        sample_N, sample_P, sample_K,
        sample_temp, sample_humidity, 
        sample_ph, sample_rainfall
    )
    
    print(f"For soil with N={sample_N}, P={sample_P}, K={sample_K}, " +
          f"temperature={sample_temp}°C, humidity={sample_humidity}%, " +
          f"pH={sample_ph}, and rainfall={sample_rainfall}mm, " +
          f"the recommended crop is: {recommended_crop}")
    
    # Step 9: Interactive mode
    try_interactive = input("\nWould you like to enter your own values for a recommendation? (y/n): ")
    if try_interactive.lower() == 'y':
        interactive_recommendation(best_model, scaler)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")