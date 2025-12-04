"""
Data preprocessing module for FER2013 emotion recognition.

This module provides utilities for one-time data preparation:
- Project structure creation
- Data splitting (train/val/test)
- File organization
"""

from .structure_manager import ProjectStructureManager
from .data_splitter import DataSplitter
from .file_organizer import FileOrganizer

__all__ = [
    'ProjectStructureManager',
    'DataSplitter',
    'FileOrganizer',
]