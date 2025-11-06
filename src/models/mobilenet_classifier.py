"""
MobileNetV3-Large model for driver distraction classification.

This module provides:
1. Model loading from torchvision pretrained weights
2. Classification head for 2-class problem (Attentive vs Distracted)
3. Fine-tuning utilities
4. Inference methods

MobileNetV3 is optimized for:
- Edge devices (Raspberry Pi, mobile)
- Real-time inference (20-30 FPS)
- Low memory footprint (~10 MB)
- Fast training (1-2 hours on Mac)
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from PIL import Image


class MobileNetDriverClassifier(nn.Module):
    """
    MobileNetV3-Large based driver distraction classifier.

    Architecture:
    - MobileNetV3-Large as backbone (pretrained on ImageNet)
    - Custom classification head for 2 classes
    - Option to freeze/unfreeze backbone
    - ~5M parameters (vs 1.6B for LFM2-VL)
    """

    def __init__(
        self,
        num_classes: int = 2,
        freeze_backbone: bool = False,
        pretrained: bool = True,
        dropout: float = 0.2,
        device: str = "auto"
    ):
        """
        Initialize the classifier.

        Args:
            num_classes: Number of output classes (2 for Attentive/Distracted)
            freeze_backbone: Whether to freeze backbone weights
            pretrained: Load ImageNet pretrained weights
            dropout: Dropout rate for classifier head
            device: Device to load model on ('auto', 'cuda', 'mps', 'cpu')
        """
        super().__init__()

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        print(f"Loading MobileNetV3-Large (pretrained={pretrained})...")

        # Load pretrained MobileNetV3-Large
        if pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            self.backbone = mobilenet_v3_large(weights=weights)
        else:
            self.backbone = mobilenet_v3_large(weights=None)

        # Get the number of features from the last layer
        # MobileNetV3 has a classifier with input features of 1280
        in_features = self.backbone.classifier[0].in_features

        # Replace the classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),  # MobileNetV3 uses Hardswish
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.Hardswish(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, num_classes)
        )

        # Initialize classifier weights
        self._init_classifier_weights()

        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

        # Determine and move to device
        self.device = self._get_device(device)
        self.to(self.device)

        print(f"✓ Model loaded on device: {self.device}")
        print(f"  Backbone frozen: {freeze_backbone}")
        print(f"  Pretrained: {pretrained}")

    def _get_device(self, device: str) -> torch.device:
        """Determine the device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)

    def _init_classifier_weights(self):
        """Initialize classifier head weights."""
        for module in self.backbone.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _freeze_backbone(self):
        """Freeze backbone weights (only train classifier head)."""
        # Freeze all feature extraction layers
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Freeze avgpool
        if hasattr(self.backbone, 'avgpool'):
            for param in self.backbone.avgpool.parameters():
                param.requires_grad = False

        print("  ✓ Backbone frozen (only classifier head will be trained)")

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  ✓ Backbone unfrozen (full model will be trained)")

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Image tensor [batch_size, 3, 224, 224]
            labels: Ground truth labels (optional, for computing loss)

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Forward through backbone
        logits = self.backbone(images)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {
            'logits': logits,
            'loss': loss
        }

    def predict(
        self,
        images: torch.Tensor,
        return_probs: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions on images.

        Args:
            images: Image tensor [batch_size, 3, 224, 224]
            return_probs: Return probabilities instead of logits

        Returns:
            Dictionary with predictions, probabilities, and class names
        """
        self.eval()

        # Move to device if needed
        if images.device != self.device:
            images = images.to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.forward(images)

        logits = outputs['logits']

        # Get predictions
        if return_probs:
            probs = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probs, dim=-1)

            return {
                'predictions': predicted_classes.cpu().numpy(),
                'probabilities': probs.cpu().numpy(),
                'class_names': ['Attentive' if p == 0 else 'Distracted'
                               for p in predicted_classes.cpu().numpy()]
            }
        else:
            predicted_classes = torch.argmax(logits, dim=-1)
            return {
                'predictions': predicted_classes.cpu().numpy(),
                'logits': logits.cpu().numpy(),
                'class_names': ['Attentive' if p == 0 else 'Distracted'
                               for p in predicted_classes.cpu().numpy()]
            }

    def save_pretrained(self, save_path: Path):
        """Save model and configuration."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save full model state dict
        torch.save(
            self.state_dict(),
            save_path / "model.pt"
        )

        # Save configuration
        config = {
            'num_classes': self.num_classes,
            'freeze_backbone': self.freeze_backbone,
            'architecture': 'MobileNetV3-Large'
        }

        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Model saved to {save_path}")

    @classmethod
    def load_pretrained(cls, load_path: Path, device: str = "auto"):
        """Load a saved model."""
        load_path = Path(load_path)

        # Load configuration
        with open(load_path / "config.json", 'r') as f:
            config = json.load(f)

        # Initialize model
        model = cls(
            num_classes=config['num_classes'],
            freeze_backbone=False,  # Don't freeze when loading
            pretrained=False,  # Don't load ImageNet weights
            device=device
        )

        # Load state dict
        state_dict = torch.load(
            load_path / "model.pt",
            map_location=model.device
        )
        model.load_state_dict(state_dict)

        print(f"✓ Model loaded from {load_path}")
        return model

    def get_trainable_parameters(self) -> int:
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """Get count of total parameters."""
        return sum(p.numel() for p in self.parameters())

    def print_parameter_stats(self):
        """Print parameter statistics."""
        trainable = self.get_trainable_parameters()
        total = self.get_total_parameters()
        percentage = (trainable / total) * 100

        print(f"\nModel Parameters:")
        print(f"  Total: {total:,}")
        print(f"  Trainable: {trainable:,} ({percentage:.2f}%)")
        print(f"  Frozen: {total - trainable:,}")

        # Estimate model size
        model_size_mb = (total * 4) / (1024 * 1024)  # 4 bytes per float32 parameter
        print(f"  Estimated size: {model_size_mb:.2f} MB")


if __name__ == "__main__":
    """Test the model."""
    print("Testing MobileNetDriverClassifier...\n")

    try:
        # Test model initialization
        model = MobileNetDriverClassifier(
            num_classes=2,
            freeze_backbone=True,
            pretrained=True,
            device="auto"
        )

        model.print_parameter_stats()

        # Test forward pass
        print("\nTesting forward pass...")
        dummy_input = torch.randn(2, 3, 224, 224).to(model.device)
        dummy_labels = torch.tensor([0, 1]).to(model.device)

        outputs = model(dummy_input, dummy_labels)
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Loss: {outputs['loss'].item():.4f}")

        # Test prediction
        print("\nTesting prediction...")
        predictions = model.predict(dummy_input)
        print(f"  Predictions: {predictions['predictions']}")
        print(f"  Probabilities shape: {predictions['probabilities'].shape}")
        print(f"  Class names: {predictions['class_names']}")

        print("\n✓ All tests passed!")
        print("\nModel is ready for training!")
        print("Expected training time: 1-2 hours on Mac M1/M2/M3")
        print("Expected inference speed: 20-30 FPS on Raspberry Pi 4")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
