# IMDb Review Rating Prediction Project

This project is a machine learning model designed to predict IMDb review ratings based on text data. Using TensorFlow, Keras, and natural language processing techniques, the model preprocesses review text, converts it into embeddings, and applies three different neural network architectures (Simple Neural Network, Convolutional Neural Network, and Long Short-Term Memory) to classify reviews as positive or negative.

## Project Structure

The project consists of the following main steps:

1. **Data Preprocessing**:
   - Remove HTML tags and non-alphabetic characters.
   - Convert to lowercase and eliminate stop words.
   - Tokenize and pad sequences to a consistent length for model input.

2. **Word Embeddings**:
   - Use GloVe embeddings (100-dimensional) to convert words into vectors.
   - Create an embedding matrix for use in the models.

3. **Model Development**:
   - **Simple Neural Network (SNN)**
   - **Convolutional Neural Network (CNN)**
   - **Long Short-Term Memory (LSTM)**
   - Each model was trained for 6 epochs and evaluated for accuracy on both the training and test sets.

4. **Evaluation & Visualization**:
   - Plot model accuracy and loss during training.
   - Evaluate each model on test data and save predictions.
   
5. **Unseen Review Predictions**:
   - Preprocess new IMDb review data, predict ratings, and save predictions to a CSV file.

## Results

The project achieved the following test accuracies:
- **SNN Model**: 75.09%
- **CNN Model**: 84.93%
- **LSTM Model**: 85.79%

## Technologies

- Python
- TensorFlow
- Keras
- NLTK
- Pandas
- NumPy
- Seaborn
- Matplotlib
