import os
import subprocess
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers

# Define paths
model_dir = "Models/05_sound_to_text/202503210353"  # Your actual directory
h5_model_path = os.path.join(model_dir, "model.h5").replace('\\', '/')
saved_model_dir = os.path.join(model_dir, "saved_model").replace('\\', '/')
output_file = os.path.join(model_dir, "model.onnx").replace('\\', '/')

print(f"Working with model at: {h5_model_path}")

# First, try to extract model configuration from h5 file
try:
    print("Attempting to read model configuration...")
    with tf.keras.utils.custom_object_scope({'tf': tf}):
        # Just read the architecture for inspection
        model_config = keras.models.load_model(h5_model_path, compile=False).get_config()
    
    # Extract input shape from the model config
    input_shape = model_config['layers'][0]['config']['batch_input_shape'][1:]
    print(f"Detected input shape: {input_shape}")
    
    # Extract output size from the last layer
    last_layer = model_config['layers'][-1]
    if 'units' in last_layer['config']:
        output_size = last_layer['config']['units']
        print(f"Detected output size: {output_size}")
    else:
        # Default if we can't find it
        output_size = 31  # 30 vocab + 1 for CTC blank
        print(f"Using default output size: {output_size}")
        
except Exception as e:
    print(f"Could not read model configuration: {e}")
    print("Using default values...")
    input_shape = [403, 193]  # Default from earlier errors
    output_size = 31  # 30 vocab + 1 for CTC blank

# Create a reconstructed model that avoids the Lambda layer issue
def create_reconstructed_model(input_shape, output_size):
    inputs = keras.Input(shape=input_shape, name="input")
    
    # Use built-in approach to add channel dimension
    # This replaces the problematic Lambda layer
    x = layers.Reshape((*input_shape, 1))(inputs)
    
    # Convolution layer 1
    x = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Convolution layer 2
    x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    # Reshape for RNN layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # RNN layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Dense layer
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(output_size, activation="softmax", name="output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create reconstructed model
print("Creating reconstructed model with same architecture...")
reconstructed_model = create_reconstructed_model(input_shape, output_size)

# Try to transfer weights from original model
try:
    print("Attempting to transfer weights from original model...")
    
    # Custom load that ignores the Lambda layer
    original_model = keras.models.load_model(h5_model_path, compile=False, custom_objects={
        'CTCloss': lambda: None  # Dummy for custom loss
    })
    
    # Skip the first layer (Input) and the Lambda layer
    start_idx = 2  # Skip Input and Lambda layers
    
    # Copy weights for each layer
    for i, layer in enumerate(reconstructed_model.layers[2:], start=2):
        if i < len(original_model.layers):
            try:
                original_weights = original_model.layers[i].get_weights()
                layer.set_weights(original_weights)
                print(f"Transferred weights for layer {i}: {layer.name}")
            except Exception as e:
                print(f"Could not transfer weights for layer {i}: {e}")
    
    print("Weight transfer completed!")
except Exception as e:
    print(f"Could not transfer weights: {e}")
    print("Continuing with uninitialized weights...")

# Save reconstructed model in SavedModel format
print("Saving reconstructed model in SavedModel format...")
os.makedirs(saved_model_dir, exist_ok=True)
tf.saved_model.save(reconstructed_model, saved_model_dir)

print(f"Model saved to {saved_model_dir}")

# Convert using the command line tool
print("Converting SavedModel to ONNX using command line...")
try:
    # Use subprocess to run the command-line converter
    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output", output_file,
        "--opset", "12"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Conversion successful!")
        print(f"Model saved to {output_file}")
    else:
        print("Conversion failed with the following error:")
        print(result.stderr)
        
        # Alternative suggestion
        print("\nAlternative approach:")
        print("Try running the following command directly in your terminal/command prompt:")
        print(f"python -m tf2onnx.convert --saved-model {saved_model_dir} --output {output_file} --opset 12")
        
except Exception as e:
    print(f"Error during conversion: {e}")
    print("\nAlternative approach:")
    print("Try running the following command directly in your terminal/command prompt:")
    print(f"python -m tf2onnx.convert --saved-model {saved_model_dir} --output {output_file} --opset 12")