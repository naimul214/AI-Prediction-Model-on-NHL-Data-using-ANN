# NHL Game Outcome Prediction

## Project Overview
This project uses a Neural Network model to predict NHL game outcomes based on various game statistics. The model analyzes historical NHL data and predicts whether a home team will win or lose a game using features like goals, shots, saves, and player statistics.

## Model Architecture
- **Model Type**: Multi-Layer Perceptron (MLP) Neural Network
- **Hidden Layers**: Two hidden layers with 10 and 5 neurons respectively
- **Activation Function**: ReLU
- **Optimizer**: Adam
- **Max Iterations**: 500

## Data Processing
The project uses multiple NHL datasets:
- Game data (basic game information)
- Team statistics
- Skater statistics
- Goalie statistics

These datasets are merged and processed to create meaningful features for the prediction model.

### Features Used
- Home and away goals
- Shots
- Saves
- Player goals
- Assists

### Target Variable
- Game outcome (home win or away win)

## Model Performance
The model achieved exceptional results on the test dataset:
- **Accuracy**: 100.00%
- **Precision**: 1.00
- **Recall**: 1.00
- **F1 Score**: 1.00

### Confusion Matrix
```
[[ 898033       9]
 [     17 1022432]]
```

## Technologies Used
- Python
- pandas: Data manipulation and analysis
- scikit-learn: Machine learning framework
- Neural Networks: MLP Classifier
- StandardScaler: Feature normalization

## Skills Demonstrated
- Data preprocessing and feature engineering
- Neural network implementation and configuration
- Model evaluation and performance metrics analysis
- Working with complex, multi-table datasets
- Sports analytics application

## Future Improvements
- Feature importance analysis to identify key factors in game outcomes
- Hyperparameter tuning to optimize model performance
- Adding additional features like team rankings, player injuries, etc.
- Time series analysis to account for team performance trends
- Expanding the model to predict score margins or total goals

## Contact Information
Naimul Hassan  
nhridoy214@gmail.com  
https://www.linkedin.com/in/naimul214

---
*This project was completed as part of "Artificial Neural Networks" course at Durham College of Applied Arts & Technology*