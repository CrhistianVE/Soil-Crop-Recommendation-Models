# Soil Intelligence Crop Recommendation Models

This repository contains pre-trained machine learning models for crop recommendation based on soil and environmental factors. The models are trained using the **Crop Recommendation Dataset**, which is included in this repository.

## Dataset
The dataset (`Crop_recommendation.csv`) consists of the following features:
- **N**: Nitrogen content in the soil
- **P**: Phosphorus content in the soil
- **K**: Potassium content in the soil
- **temperature**: Temperature in degrees Celsius
- **humidity**: Relative humidity percentage
- **ph**: pH value of the soil
- **rainfall**: Annual rainfall in mm
- **label**: Recommended crop (target variable)

## Models
The following machine learning models have been trained and saved in the `models/` directory:

| Model Name          | Description |
|--------------------|-------------|
| `DecisionTree.pkl`  | Decision Tree Classifier |
| `NBClassifier.pkl`  | Naïve Bayes Classifier |
| `RandomForest.pkl`  | Random Forest Classifier |
| `XGBoost.pkl`      | XGBoost Classifier |
| `label_map.pkl`    | Label encoding mapping |

## Usage
These models can be loaded and used for inference in Python using the `joblib` library:

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/RandomForest.pkl')

# Load label mapping
label_map = joblib.load('models/label_map.pkl')
reverse_label_map = {v: k for k, v in label_map.items()}

# Example input (replace with real data)
sample_input = pd.DataFrame([[90, 42, 43, 20.87, 82.02, 6.5, 202.93]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

# Predict
prediction = model.predict(sample_input)
predicted_crop = reverse_label_map[prediction[0]]
print("Recommended Crop:", predicted_crop)
```

## Repository Structure
```
Soil-Intel-Crop-Models/
│── models/
│   ├── DecisionTree.pkl
│   ├── NBClassifier.pkl
│   ├── RandomForest.pkl
│   ├── XGBoost.pkl
│   ├── label_map.pkl
│── Crop_recommendation.csv
│── README.md
```

## License
This repository is open for educational and research purposes. Feel free to use and modify it as needed.

## Author
Ben

## Acknowledgment
Dataset Source: **Crop Recommendation Dataset**

