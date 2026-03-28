# Context-Aware-Chatbot-Using-LangChain-RAG
# Project Overview
This project features a Multimodal Deep Learning model developed with TensorFlow and Keras to enhance real estate price prediction. Unlike standard models limited to numerical data, this system integrates two distinct data streams:
Tabular Data: Structural specifications such as square footage, room counts, and property age.
Image Features: Visual data (represented as 512-dimensional vectors) that capture the property's aesthetic condition.
The system utilizes a Late Fusion technique, where two independent neural network branches are trained separately before being merged for a final valuation.
# Model Architecture
The architecture is divided into two primary input branches:
# Tabular Branch
Input: 5 numerical features (area, bedrooms, bathrooms, age, and garage).
Structure: Dense layers (128 → 64) utilizing Dropout (0.3) to ensure model generalization and prevent overfitting.
# Image Branch
Input: A 512-dimensional feature vector (representative of high-level features from models like VGG16).
Structure: Dense layers (256 → 128) with Dropout (0.3).
# Fusion & Final Output
Merge: A concatenation layer that combines insights from both branches.
Head: Final dense layers (128 → 64) terminating in a linear output for precise price regression.
# Getting Started
To set up the environment, install the required dependencies:
pip install numpy pandas tensorflow scikit-learn matplotlib pillow opencv-python joblib
