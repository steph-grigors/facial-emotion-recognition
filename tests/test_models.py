# test_models.py (run from project root)
from src.models import create_model
import torch

print("Testing Baseline CNN...")
baseline = create_model('baseline', num_classes=7)
x1 = torch.randn(2, 3, 224, 224)
out1 = baseline(x1)
assert out1.shape == (2, 7), "Baseline shape wrong!"
print("✅ Baseline CNN works!")

print("\nTesting Enhanced CNN...")
enhanced = create_model('enhanced', num_classes=7, pretrained=False)
x2 = torch.randn(2, 3, 128, 128)
out2 = enhanced(x2)
assert out2.shape == (2, 7), "Enhanced shape wrong!"
print("✅ Enhanced CNN works!")