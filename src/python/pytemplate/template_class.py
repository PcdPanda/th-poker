"""
A template module that defines a demo class for demonstration purposes.

Author: Chongdan Pan
Created: 2024-10-24
"""

from __future__ import annotations


class TemplateClass:
    """
    Example template class for demonstration.

    Args:
        num (int): The base integer value for the instance.

    Attributes:
        _num (int): The stored integer value.
    """

    def __init__(self, num: int):
        """
        Initialize the TemplateClass instance.

        Args:
            num (int): The base integer value for the instance.
        """
        self._num = num

    def get_data(self, num: int) -> list[int]:
        """
        Generate a list containing the stored integer repeated `num` times.

        Args:
            num (int): The number of times to repeat the stored integer.

        Returns:
            list[int]: A list of integers.
        """
        return [self._num] * num
