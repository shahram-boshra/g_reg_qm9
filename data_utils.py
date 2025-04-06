import torch
from torch_geometric.datasets import QM9
from typing import List, Callable
import logging
from exceptions import DataLoadingError
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

class SelectQM9Targets(object):
    """
    Transforms QM9 graph data by selecting specific target properties.

    This class is designed to be used as a transform within a PyTorch Geometric
    dataset. It allows the user to specify a list of indices corresponding to
    the desired target properties from the QM9 dataset.

    Attributes:
        target_indices (List[int]): A list of integers representing the indices
                                     of the target properties to be selected.

    Methods:
        __init__(self, target_indices: List[int]): Initializes the transform with
                                                     the specified target indices.
        __call__(self, data: Data) -> Data: Applies the transform to a single
                                             data object, selecting the specified
                                             target properties.
    """
    def __init__(self, target_indices: List[int]):
        """
        Initializes the SelectQM9Targets transform.

        Args:
            target_indices (List[int]): A list of integers representing the indices
                                         of the target properties to be selected.
        """
        self.target_indices = target_indices

    def __call__(self, data: Data) -> Data:
        """
        Applies the transform to a single data object.

        Selects the target properties specified by `self.target_indices` from
        the `data.y` tensor.

        Args:
            data (Data): A PyTorch Geometric Data object representing a single
                         graph from the QM9 dataset.

        Returns:
            Data: The transformed Data object with only the selected target
                  properties.

        Raises:
            DataLoadingError: If an error occurs during the target selection
                              process.
        """
        try:
            data.y = data.y[:, self.target_indices]
            return data
        except Exception as e:
            logger.error(f"Error selecting QM9 targets: {e}")
            raise DataLoadingError(f"Failed to select QM9 targets: {e}")

def load_qm9_data(root: str, target_indices: List[int]) -> List[Data]:
    """
    Loads QM9 dataset and applies a target selection transform.

    This function loads the QM9 dataset from the specified root directory and
    applies the `SelectQM9Targets` transform to select specific target properties.

    Args:
        root (str): The root directory where the QM9 dataset is stored.
        target_indices (List[int]): A list of integers representing the indices
                                     of the target properties to be selected.

    Returns:
        List[Data]: A list of PyTorch Geometric Data objects, where each object
                    represents a graph from the QM9 dataset with only the
                    selected target properties.

    Raises:
        DataLoadingError: If an error occurs during the dataset loading or
                          transform application process.
    """
    try:
        dataset = QM9(root=root)
        transform = SelectQM9Targets(target_indices=target_indices)
        dataset.transform = transform
        data_list = [dataset[i] for i in range(len(dataset))]
        return data_list
    except Exception as e:
        logger.error(f"Error loading QM9 data: {e}")
        raise DataLoadingError(f"Failed to load QM9 data: {e}")
