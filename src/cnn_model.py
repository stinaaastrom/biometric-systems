import numpy as np

# TensorFlow/Keras imports
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50

AGE_CLASSES = [(0,12), (13,17), (18,25), (26,35), (36,45), (46,60), (61,74), (75, None)]

# Default path for model files (absolute path)
import os
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'age_model.keras')
# Import evaluation function (works whether running src as a package or plain scripts)
eval_fn = None
try:
    # If src is a package
    from .evaluation import evaluate_model_performance as eval_fn  # type: ignore
except Exception:
    try:
        # Plain module import in same folder
        from evaluation import evaluate_model_performance as eval_fn  # type: ignore
    except Exception:
        eval_fn = None

class CNNModel:
    
    def __init__(self, X_train, age_train, X_test, age_test):
        self.X_train = X_train
        self.age_train = age_train
        self.X_test = X_test
        self.age_test = age_test
        self.model = None
        self.history = None

    @staticmethod
    def find_age_class(predicted_age):
        """
        Returns the index of the class an age belongs to.
        Clamps ages to valid range: <0 goes to class 0, 75+ goes to last class.
        """
        # Handle negative predictions - assign to youngest class
        if predicted_age < AGE_CLASSES[0][0]:
            return 0
        
        # Find matching class
        for idx, (min_age, max_age) in enumerate(AGE_CLASSES):
            if max_age is None:
                # Last class catches everything >= min_age
                if predicted_age >= min_age:
                    return idx
            elif min_age <= predicted_age <= max_age:
                return idx
        
        # Fallback: if somehow missed, assign to last class
        return len(AGE_CLASSES) - 1


    def build_cnn_model(self):
        """Build CNN model using ResNet50 transfer learning."""
        print("\nLoading ResNet50 with transfer learning...")
        
        # Load pre-trained ResNet50 (without top classification layer)
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(64, 64, 3)
        )
        
        # Freeze only the first 100 layers, allow fine-tuning of later layers
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        print(f"Frozen first 100 layers, {len([l for l in base_model.layers if l.trainable])} layers trainable")
        
        # Build model on top of pre-trained base
        input_layer = Input(shape=(64, 64, 3))
        x = base_model(input_layer, training=True)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        age_output = Dense(1, name='age_output')(x)
        
        print("ResNet50 transfer learning model created")

        self.model = Model(inputs=input_layer, outputs=age_output)
        self.model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.001),
            metrics=['mae']
        )

        self.model.summary()

        print("\nStarting training...")
        self.history = self.model.fit(
            self.X_train,
            self.age_train,
            validation_data=(self.X_test, self.age_test),
            epochs=5,
            batch_size=128,
            verbose=1
        )
        print("Training complete!")

    def evaluate_model_performance(self, gen_test=None, etn_test=None):
        """Delegate evaluation to separate module to keep model clean."""
        if eval_fn is None:
            # Fallback attempt without package context
            try:
                from evaluation import evaluate_model_performance as eval_fn_local
            except Exception as e:
                raise ImportError(f"Could not import evaluation module: {e}")
            return eval_fn_local(self.model, self.X_test, self.age_test, self.history, AGE_CLASSES, self.find_age_class, gen_test, etn_test)
        return eval_fn(self.model, self.X_test, self.age_test, self.history, AGE_CLASSES, self.find_age_class, gen_test, etn_test)
    
    def save_model(self, filepath=None):
        """Save the trained model to a file"""
        if self.model is None:
            print("No model to save. Train the model first.")
            return
        
        if filepath is None:
            filepath = DEFAULT_MODEL_PATH
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load a trained model from a file"""
        if filepath is None:
            filepath = DEFAULT_MODEL_PATH
        
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model