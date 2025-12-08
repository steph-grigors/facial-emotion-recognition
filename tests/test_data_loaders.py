# test_dataloaders.py (run from project root)
from src.data_loading import create_dataloaders

print("Testing Baseline dataloaders...")
train_l, val_l, test_l, _ = create_dataloaders(
    data_dir='data/processed',
    model_type='baseline',
    batch_size=4
)
images, labels = next(iter(train_l))
print(f"Baseline images shape: {images.shape}")
assert images.shape[1:] == (3, 224, 224), "Wrong size!"
print("✅ Baseline dataloaders work!")

print("\nTesting Enhanced dataloaders...")
train_l, val_l, test_l, _ = create_dataloaders(
    data_dir='data/processed',
    model_type='enhanced',
    batch_size=4
)
images, labels = next(iter(train_l))
print(f"Enhanced images shape: {images.shape}")
assert images.shape[1:] == (3, 128, 128), "Wrong size!"
print("✅ Enhanced dataloaders work!")