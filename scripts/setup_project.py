#!/usr/bin/env python3
"""
Setup script for the facial emotion recognition project.

This script:
1. Creates the project directory structure
2. Creates a default configuration file
3. Loads and splits the dataset
4. Organizes files into train/val/test directories
5. Verifies the setup

Usage:
    python scripts/setup_project.py --data-dir path/to/raw/data
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import ProjectStructureManager, DataSplitter, FileOrganizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Setup project structure and prepare data'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to raw data directory (containing emotion subdirectories)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data (default: data/processed)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--operation',
        type=str,
        choices=['copy', 'move'],
        default='copy',
        help='File operation: copy or move (default: copy)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--skip-structure',
        action='store_true',
        help='Skip project structure creation'
    )
    
    parser.add_argument(
        '--skip-split',
        action='store_true',
        help='Skip data splitting (use existing splits)'
    )
    
    parser.add_argument(
        '--skip-organize',
        action='store_true',
        help='Skip file organization'
    )
    
    return parser.parse_args()


def main():
    """Main setup function."""
    args = parse_args()
    
    print("="*70)
    print("FACIAL EMOTION RECOGNITION - PROJECT SETUP")
    print("="*70)
    
    # Step 1: Create project structure
    if not args.skip_structure:
        print("\n" + "="*70)
        print("STEP 1: Creating Project Structure")
        print("="*70)
        
        manager = ProjectStructureManager()
        
        # Define emotion classes
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Create structure
        manager.create_structure(emotions=emotions)
        
        # Create default config
        manager.create_default_config()
        
        # Verify
        manager.verify_structure()
    else:
        print("\n⏭️  Skipping project structure creation")
    
    # Step 2: Split data
    if not args.skip_split:
        print("\n" + "="*70)
        print("STEP 2: Splitting Dataset")
        print("="*70)
        
        splitter = DataSplitter(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )
        
        # Load data
        df = splitter.load_from_directory(args.data_dir)
        
        # Create splits
        train_df, val_df, test_df = splitter.split_data(df)
        
        # Verify splits
        verification = splitter.verify_splits()
        
        # Save splits
        splitter.save_splits('results/metrics')
    else:
        print("\n⏭️  Skipping data splitting")
        print("Loading existing splits from results/metrics/")
        
        splitter = DataSplitter(random_seed=args.random_seed)
        train_df, val_df, test_df = splitter.load_splits('results/metrics')
    
    # Step 3: Organize files
    if not args.skip_organize:
        print("\n" + "="*70)
        print("STEP 3: Organizing Files")
        print("="*70)
        
        organizer = FileOrganizer(
            output_dir=args.output_dir,
            operation=args.operation,
            create_dirs=True
        )
        
        # Organize files
        organizer.organize_splits(train_df, val_df, test_df)
        
        # Verify file counts
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        counts_df = organizer.verify_file_counts(emotions)
    else:
        print("\n⏭️  Skipping file organization")
    
    # Final summary
    print("\n" + "="*70)
    print("SETUP COMPLETE! 🎉")
    print("="*70)
    print("\n✅ Project is ready for training!")
    print("\n📁 Directory structure:")
    print("  ├── data/processed/")
    print("  │   ├── train/")
    print("  │   ├── val/")
    print("  │   └── test/")
    print("  ├── configs/config.yaml")
    print("  └── results/metrics/")
    print("      ├── train_split.csv")
    print("      ├── val_split.csv")
    print("      ├── test_split.csv")
    print("      └── split_info.yaml")
    print("\n🚀 Next steps:")
    print("  1. Review the configuration in configs/config.yaml")
    print("  2. Start training with scripts/train.py")
    print("  3. Check notebooks/ for exploratory analysis")


if __name__ == '__main__':
    main()
