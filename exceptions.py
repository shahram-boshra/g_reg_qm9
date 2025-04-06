# --- exceptions.py ---

class ConfigLoadError(Exception):
    """
    Custom exception for configuration loading issues.

    This exception should be raised when there are problems loading or parsing
    the configuration file, such as:

    - File not found.
    - Invalid file format (e.g., incorrect JSON or YAML syntax).
    - Missing required configuration parameters.
    - Invalid data types for configuration parameters.

    Example:
        try:
            config = load_config("config.yaml")
        except ConfigLoadError as e:
            print(f"Error loading configuration: {e}")
            # Handle the error appropriately, e.g., exit the program or use default values.
    """
    pass

class DataSplitError(Exception):
    """
    Custom exception for data splitting issues.

    This exception should be raised when there are problems splitting the dataset
    into training, validation, or test sets, such as:

    - Invalid split ratios (e.g., sum of ratios not equal to 1).
    - Insufficient data for the requested split.
    - Errors during the splitting process (e.g., incorrect indexing).

    Example:
        try:
            train_set, val_set, test_set = split_data(data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        except DataSplitError as e:
            print(f"Error splitting data: {e}")
            # Handle the error appropriately, e.g., use a different splitting strategy or exit.
    """
    pass

class ModelInitializationError(Exception):
    """
    Custom exception for model initialization issues.

    This exception should be raised when there are problems creating or initializing
    the machine learning model, such as:

    - Invalid model architecture.
    - Missing required model parameters.
    - Errors during weight initialization.
    - Incompatible model parameters with the dataset.

    Example:
        try:
            model = create_model(input_dim=10, output_dim=5)
        except ModelInitializationError as e:
            print(f"Error initializing model: {e}")
            # Handle the error appropriately, e.g., use a different model or adjust parameters.
    """
    pass

class TrainingError(Exception):
    """
    Custom exception for training issues.

    This exception should be raised when there are problems during the model training
    process, such as:

    - Loss function errors.
    - Optimizer errors.
    - Gradient issues (e.g., NaN gradients).
    - Memory errors.
    - Unexpected interruptions.

    Example:
        try:
            train_model(model, train_loader, optimizer, loss_fn)
        except TrainingError as e:
            print(f"Error during training: {e}")
            # Handle the error appropriately, e.g., adjust hyperparameters or debug the training loop.
    """
    pass

class TestingError(Exception):
    """
    Custom exception for testing issues.

    This exception should be raised when there are problems during the model testing
    or evaluation process, such as:

    - Errors during prediction.
    - Incorrect evaluation metrics.
    - Invalid test data.
    - Device errors during testing.

    Example:
        try:
            evaluate_model(model, test_loader)
        except TestingError as e:
            print(f"Error during testing: {e}")
            # Handle the error appropriately, e.g., check the evaluation metrics or debug the testing loop.
    """
    pass

class DeviceInitializationError(Exception):
    """
    Custom exception for device initialization issues.

    This exception should be raised when there are problems initializing the device
    (e.g., CPU or GPU) for model training or inference, such as:

    - GPU not available.
    - CUDA errors.
    - Insufficient memory on the device.
    - Device driver issues.

    Example:
        try:
            device = initialize_device()
        except DeviceInitializationError as e:
            print(f"Error initializing device: {e}")
            # Handle the error appropriately, e.g., use CPU or exit the program.
    """
    pass

class DatasetLoadingError(Exception):
    """
    Custom exception for dataset loading issues.

    This exception should be raised when there are problems loading the dataset,
    such as:

    - File not found.
    - File format errors.
    - Data corruption.
    - Memory errors during loading.
    - Incorrect data transformations.

    Example:
        try:
            dataset = load_dataset("data.csv")
        except DatasetLoadingError as e:
            print(f"Error loading dataset: {e}")
            # Handle the error appropriately, e.g., check the file path or use a different dataset.
    """
    pass

class ModelLayerInitializationError(Exception):
    """
    Custom exception for model layer initialization issues.

    This exception should be raised when there are problems initializing individual
    layers within the model, such as:

    - Invalid layer parameters.
    - Incompatible layer dimensions.
    - Errors during custom layer creation.

    Example:
        try:
            layer = create_custom_layer(input_dim=10, output_dim=5)
        except ModelLayerInitializationError as e:
            print(f"Error initializing model layer: {e}")
            # Handle the error appropriately, e.g., adjust layer parameters or debug the layer creation.
    """
    pass

class EarlyStoppingError(Exception):
    """
    Custom exception for early stopping issues.

    This exception should be raised when there are problems related to the early
    stopping mechanism, such as:

    - Invalid early stopping parameters.
    - Errors during patience checks.
    - Unexpected termination of training due to early stopping logic.

    Example:
        try:
            early_stopper = EarlyStopper(patience=10, min_delta=0.001)
            # ... training loop ...
            if early_stopper.early_stop(val_loss):
                print("Early stopping triggered.")
                break
        except EarlyStoppingError as e:
            print(f"Error during early stopping: {e}")
            # Handle the error appropriately, e.g., adjust early stopping parameters or disable early stopping.
    """
    pass

class PlottingError(Exception):
    """
    Custom exception for plotting issues.

    This exception should be raised when there are problems generating plots or
    visualizations, such as:

    - Invalid plot parameters.
    - Errors during data visualization.
    - Missing plotting libraries.
    - File saving errors.

    Example:
        try:
            plot_results(history)
        except PlottingError as e:
            print(f"Error plotting results: {e}")
            # Handle the error appropriately, e.g., check plot parameters or install necessary libraries.
    """
    pass

class DataLoadingError(DatasetLoadingError):
    """
    Alias to DatasetLoadingError for backward compatibility.

    This alias is provided to ensure that existing code that uses `DataLoadingError`
    continues to function correctly. It maps directly to `DatasetLoadingError`.
    """
    pass
