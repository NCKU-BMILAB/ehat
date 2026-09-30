#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <env_name>"
    echo "  <env_name>: The name for the new Python environment (e.g., ehat_env)"
    exit 1
}

# Check if an environment name is provided
if [ -z "$1" ]; then
    usage
fi

ENV_NAME="$1"
PYTHON_VERSION="3.8"

echo "Setting up Python environment: $ENV_NAME with Python $PYTHON_VERSION"

# Check if conda is available and prefer it
if command -v conda &> /dev/null
then
    echo "Conda detected. Creating conda environment."
    # Create conda environment
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -c conda-forge -y

    # Check if environment creation was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create conda environment. Exiting."
        exit 1
    fi

    # Activate the environment (for the current shell session, but not for the script's subshell)
    # The user will need to activate it manually after the script finishes.
    echo "Conda environment '$ENV_NAME' created. Please activate it using: conda activate $ENV_NAME"
    echo "Installing packages..."
    conda run -n "$ENV_NAME" pip3 install \
        psutil==7.2.2 \
        numpy==1.20.2 \
        pyyaml==6.0.2 \
        pandas==1.2.5 \
        scipy==1.6.2 \
        tensorboard==2.5.0 \
        protobuf==3.6.0 \
        tqdm==4.59.0 \
        simpleitk==2.02 \
        scikit-learn==0.24.1 \
        torch==1.13.1 torchvision==0.14.1

    if [ $? -ne 0 ]; then
        echo "Error: Failed to install packages in conda environment. Exiting."
        exit 1
    fi
    echo "Conda environment setup complete. Remember to run 'conda activate $ENV_NAME' to use it."
else
    echo "Conda is not available on your device! Please install it."
fi

echo "All required packages installed."
echo "Note: PyTorch 1.13.1 with CUDA 11.8 was specified. If you have a different CUDA version or no GPU, you might need to adjust the PyTorch installation command."
