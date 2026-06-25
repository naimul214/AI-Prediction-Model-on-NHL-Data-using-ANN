import os
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Import local data loader
from data_loader import load_and_preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    """
    Main pipeline function that loads the preprocessed data, standardizes it, 
    trains the Neural Network (MLPClassifier), evaluates performance, and 
    serializes the trained model and scaler.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load and Preprocess Data
    try:
        X, y = load_and_preprocess_data()
    except Exception as e:
        logger.error(f"Failed loading data: {e}")
        return
        
    # 2. Split into training and test sets (80/20 split)
    logger.info("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Standardize Features
    logger.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Initialize and train MLP Classifier
    logger.info("Initializing MLPClassifier neural network...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(10, 5), 
        max_iter=500, 
        activation='relu', 
        solver='adam', 
        random_state=42
    )
    
    logger.info("Fitting Neural Network model...")
    mlp.fit(X_train_scaled, y_train)
    
    # 5. Evaluate the model
    logger.info("Evaluating model on test set...")
    y_pred = mlp.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info("=== Evaluation Results ===")
    logger.info(f"Accuracy:  {accuracy * 100:.2f}%")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # 6. Save model and scaler artifacts
    model_path = os.path.join(models_dir, 'mlp_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    logger.info(f"Saving model to {model_path}")
    joblib.dump(mlp, model_path)
    
    logger.info(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)
    
    logger.info("Training and serialization completed successfully.")

if __name__ == '__main__':
    train_model()
