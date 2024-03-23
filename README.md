# LSTM Time Series Prediction

This project demonstrates time series prediction using LSTM (Long Short-Term Memory) neural networks. It utilizes a Python script to preprocess the Jena climate dataset and train an LSTM model for temperature prediction.

## Usage

1. **Requirements**: Ensure you have the following libraries installed:
   - Pandas
   - TensorFlow
   - Seaborn
   - NumPy
   - Matplotlib
   - Keras

2. **Data Preparation**:
   - The Jena climate dataset should be downloaded and placed in the "datasets_ml" folder.
   - Hourly samples from the dataset are selected (every sixth hour).
   - The "Date Time" column is set as the index and the dataset is prepared.

3. **Data Splitting**:
   - The `data_to_split` function splits the dataset into input and output sequences with a specified window size.

4. **Training and Evaluation**:
   - The dataset is split into training, validation, and test sets.
   - An LSTM model is created using Keras with two LSTM layers followed by dense layers.
   - The best model is saved using ModelCheckpoint.
   - The model is compiled with Mean Squared Error loss and Adam optimizer.
   - The model is trained using the training data and evaluated on the validation set.

## Files and Folders

- `main.py`: Python script containing the code for creating and training the LSTM model.
- `datasets_ml/`: Folder containing the Jena climate dataset.
- `models/`: Folder where the best model is saved.

## Contributing

Contributions are welcome. If you want to improve the project or add new features, please create a pull request.

## License

This project is licensed under the MIT License. For more information, see the [LICENSE](LICENSE) file.

## Contact

If you have any questions or suggestions, feel free to contact us

----------------------------------------


