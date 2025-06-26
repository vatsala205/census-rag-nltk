import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import matplotlib.pyplot as plt


def train_census_chatbot():
    """Train a neural network model for census data chatbot"""

    # Load the training data
    try:
        with open('training_data.pkl', 'rb') as f:
            words, classes, X_train, y_train = pickle.load(f)
    except FileNotFoundError:
        print("Error: training_data.pkl not found. Please run preprocess.py first.")
        return

    print(f"Training data loaded:")
    print(f"- Vocabulary size: {len(words)}")
    print(f"- Number of classes: {len(classes)}")
    print(f"- Training samples: {len(X_train)}")

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"Input shape: {X_train.shape}")
    print(f"Output shape: {y_train.shape}")

    # Build the model architecture optimized for census data
    model = Sequential()

    # Input layer
    model.add(Dense(256, input_shape=(len(X_train[0]),), activation='relu'))
    model.add(Dropout(0.4))

    # Hidden layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(len(y_train[0]), activation='softmax'))

    # Compile the model with Adam optimizer (often better than SGD)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    print("\nModel architecture:")
    model.summary()

    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=20,
        restore_best_weights=True
    )

    # Train the model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=8,
        verbose=1,
        callbacks=[early_stopping],
        validation_split=0.1  # Use 10% of data for validation
    )

    # Save the model
    model.save('chatbot_model.h5')
    print("\nModel trained and saved as chatbot_model.h5")

    # Plot training history
    try:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        print("Training history plots saved as 'training_history.png'")
    except ImportError:
        print("Matplotlib not available. Skipping plots.")

    # Evaluate the model
    final_loss, final_accuracy = model.evaluate(X_train, y_train, verbose=0)
    print(f"\nFinal training accuracy: {final_accuracy:.4f}")
    print(f"Final training loss: {final_loss:.4f}")

    # Test the model with some sample inputs
    print("\n=== Testing Model ===")
    test_samples = [
        "What is the population of India",
        "How many people live in Delhi",
        "Area of Maharashtra",
        "Households in Punjab",
        "Villages in Gujarat"
    ]

    for sample in test_samples:
        # Simple tokenization for testing
        words_lower = [w.lower() for w in words]
        sample_words = sample.lower().split()
        bag = [1 if w in sample_words else 0 for w in words_lower]

        prediction = model.predict(np.array([bag]), verbose=0)[0]
        predicted_class_index = np.argmax(prediction)
        confidence = prediction[predicted_class_index]
        predicted_tag = classes[predicted_class_index]

        print(f"Input: '{sample}'")
        print(f"Predicted: {predicted_tag} (confidence: {confidence:.3f})")
        print()

    return model, history


if __name__ == "__main__":
    # Train the model
    model, history = train_census_chatbot()

    print("\nTraining completed!")
    # print("Next steps:")
    # print("1. Run the chatbot with: python census_chat.py")
    # print("2. Test with queries like:")
    # print("   - 'What is the population of [location]?'")
    # print("   - 'How big is [location]?'")
    # print("   - 'Tell me about households in [location]'")