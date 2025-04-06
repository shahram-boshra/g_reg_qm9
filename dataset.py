import torch
import torch_geometric.data
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
import logging
import os
from data_utils import SelectQM9Targets
from exceptions import DatasetLoadingError
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

class QM9Dataset(torch_geometric.data.Dataset):
    """
    A custom PyTorch Geometric Dataset for loading and processing the QM9 dataset.

    This class extends `torch_geometric.data.Dataset` to handle the QM9 dataset,
    allowing for target selection and subsetting. It loads pre-processed data
    from a specified file path and transforms it into `torch_geometric.data.Data`
    objects.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (Optional[callable], optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed version.
            Defaults to None.
        target_indices (Optional[List[int]], optional): List of indices specifying which
            targets from the QM9 dataset to include. If None, all targets are included.
            Defaults to None.
        subset_size (Optional[int], optional): The number of data points to include in the
            subset. If None, the entire dataset is used. Defaults to None.

    Attributes:
        root (str): Root directory of the dataset.
        target_indices (Optional[List[int]]): Indices of selected targets.
        subset_size (Optional[int]): Size of the dataset subset.
        data_list (List[Data]): List of `torch_geometric.data.Data` objects.

    Raises:
        DatasetLoadingError: If there is an error loading the dataset from the processed file.
    """

    def __init__(self, root: str, transform: Optional[callable] = None, target_indices: Optional[List[int]] = None, subset_size: Optional[int] = None):
        self.root = root
        self.target_indices = target_indices
        self.subset_size = subset_size
        try:
            self.data_list: List[Data] = self._load_data()
        except DatasetLoadingError as e:
            logger.error(f"Failed to load dataset: {e}")
            raise # Re-raise the exception to propagate it
        super().__init__(self.root, transform)
        logger.info(f"QM9Dataset initialized. Data loaded from: {os.path.join(self.processed_dir, self.processed_file_names[0])}")

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns the name of the processed data file.

        Returns:
            List[str]: List containing the filename of the processed data file.
        """
        return ['qm9_v3.pt']

    def process(self) -> None:
        """
        Processes the raw data and saves it to the processed directory.

        This method is intended to be overridden if custom processing is required.
        In this implementation, it does nothing as the data is expected to be
        pre-processed.
        """
        logger.info("Processing QM9 data...")
        # If the processed file exists, this function should not be executed.
        pass

    def _load_data(self) -> List[Data]:
        """
        Loads the pre-processed data from the specified file path.

        This method loads the data from the processed file, selects the specified
        targets, and optionally creates a subset of the data.

        Returns:
            List[Data]: List of `torch_geometric.data.Data` objects.

        Raises:
            DatasetLoadingError: If there is an error loading the data from the file.
        """
        processed_path = os.path.join(self.processed_dir, self.processed_file_names[0])
        logger.info(f"Attempting to load data from: {processed_path}")
        if os.path.exists(processed_path):
            logger.info(f"File exists: {processed_path}")
        else:
            logger.info(f"File does not exist: {processed_path}")

        try:
            loaded_data = torch.load(processed_path)
            data_list: List[Data] = []
            for data in loaded_data:
                y = data['y'][:, self.target_indices]
                graph_data = torch_geometric.data.Data(x=data['x'], edge_index=data['edge_index'], y=y, pos=data['pos'])
                data_list.append(graph_data)

            if self.subset_size is not None:
                data_list = data_list[:self.subset_size]

            return data_list
        except Exception as e:
            raise DatasetLoadingError(f"Error loading data from {processed_path}: {e}")

    def len(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        """
        Retrieves the `torch_geometric.data.Data` object at the specified index.

        Args:
            idx (int): Index of the data sample to retrieve.

        Returns:
            Data: The `torch_geometric.data.Data` object at the specified index.
        """
        return self.data_list[idx]

    
