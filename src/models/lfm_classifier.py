"""
LFM2-VL-1.6B model wrapper for driver distraction classification.

This module provides:
1. Model loading from HuggingFace
2. Classification head for 2-class problem (Attentive vs Distracted)
3. Fine-tuning utilities
4. Inference methods
"""

import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoConfig
)
from typing import Optional, Dict, Any, List
from pathlib import Path
import json


class LFMDriverClassifier(nn.Module):
    """
    LFM2-VL-1.6B based driver distraction classifier.

    Architecture:
    - LFM2-VL-1.6B as backbone (vision-language model)
    - Custom classification head for 2 classes
    - Option to freeze/unfreeze parts of the model
    """

    def __init__(
        self,
        model_name: str = "LiquidAI/LFM2-VL-1.6B",
        num_classes: int = 2,
        freeze_vision_encoder: bool = True,
        freeze_language_model: bool = False,
        use_prompt_based: bool = False,
        device: str = "auto"
    ):
        """
        Initialize the classifier.

        Args:
            model_name: HuggingFace model identifier
            num_classes: Number of output classes (2 for Attentive/Distracted)
            freeze_vision_encoder: Whether to freeze vision encoder weights
            freeze_language_model: Whether to freeze language model weights
            use_prompt_based: Use prompt-based classification instead of classifier head
            device: Device to load model on ('auto', 'cuda', 'mps', 'cpu')
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.use_prompt_based = use_prompt_based

        print(f"Loading {model_name}...")

        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name)

        # Load the base model
        # Use float32 on MPS due to compatibility issues with bfloat16
        model_dtype = torch.float32 if self._get_device(device).type == 'mps' else torch.bfloat16
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map=device if device != "auto" else None,
            dtype=model_dtype,
            low_cpu_mem_usage=True
        )

        # Get hidden size from model config
        self.hidden_size = self.config.text_config.hidden_size

        # Determine device first
        self.device = self._get_device(device)

        if not use_prompt_based:
            # Add classification head
            # We'll add a pooling layer + classifier on top of the model
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )

            # Initialize classifier weights
            self._init_classifier_weights()

            # Match classifier dtype to model dtype and move to device
            self.classifier = self.classifier.to(dtype=model_dtype, device=self.device)

        # Freeze parts of the model if specified
        if freeze_vision_encoder:
            self._freeze_vision_encoder()

        if freeze_language_model:
            self._freeze_language_model()

        print(f"✓ Model loaded on device: {self.device}")
        print(f"  Vision encoder frozen: {freeze_vision_encoder}")
        print(f"  Language model frozen: {freeze_language_model}")
        print(f"  Classifier mode: {'Prompt-based' if use_prompt_based else 'Head-based'}")

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
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _freeze_vision_encoder(self):
        """Freeze vision encoder weights."""
        # LFM2-VL has vision_tower (vision encoder) and language_model
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'vision_tower'):
            for param in self.base_model.model.vision_tower.parameters():
                param.requires_grad = False
            print("  ✓ Vision encoder frozen")
        else:
            print("  ⚠ Vision encoder not found - skipping freeze")

    def _freeze_language_model(self):
        """Freeze language model weights."""
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'language_model'):
            for param in self.base_model.model.language_model.parameters():
                param.requires_grad = False
            print("  ✓ Language model frozen")
        else:
            print("  ⚠ Language model not found - skipping freeze")

    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("  ✓ All parameters unfrozen")

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pixel_values: Image tensor from processor
            input_ids: Text token IDs
            attention_mask: Attention mask for text
            labels: Ground truth labels (optional, for computing loss)

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        if self.use_prompt_based:
            # Prompt-based classification
            outputs = self.base_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            # Parse generated text for classification
            # This requires more complex logic to extract answer from generated text
            # For now, we'll use the simpler classifier head approach
            raise NotImplementedError("Prompt-based classification not fully implemented yet")

        else:
            # Classifier head approach
            # Get model outputs
            outputs = self.base_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )

            # Get last hidden state
            # Pool the sequence (mean pooling or use CLS token)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            pooled = hidden_states.mean(dim=1)  # Mean pooling across sequence

            # Pass through classifier
            logits = self.classifier(pooled)

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
        images: List,
        processor: Any,
        prompt: str = "Is the driver attentive or distracted?",
        return_probs: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions on images.

        Args:
            images: List of PIL images
            processor: HuggingFace processor
            prompt: Text prompt for classification
            return_probs: Return probabilities instead of logits

        Returns:
            Dictionary with predictions, probabilities, and class names
        """
        self.eval()

        # Create conversations
        conversations = []
        for image in images:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            conversations.append(conversation)

        # Process inputs
        inputs = processor(conversations, return_tensors="pt", padding=True)

        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.forward(**inputs)

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

        # Save base model
        self.base_model.save_pretrained(save_path / "base_model")

        # Save classifier head if using head-based approach
        if not self.use_prompt_based:
            torch.save(
                self.classifier.state_dict(),
                save_path / "classifier_head.pt"
            )

        # Save configuration
        config = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'use_prompt_based': self.use_prompt_based,
            'hidden_size': self.hidden_size
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
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            use_prompt_based=config['use_prompt_based'],
            device=device
        )

        # Load base model
        # Use float32 on MPS due to compatibility issues with bfloat16
        model_dtype = torch.float32 if device == 'mps' else torch.bfloat16
        model.base_model = AutoModelForImageTextToText.from_pretrained(
            load_path / "base_model",
            device_map=device if device != "auto" else None,
            dtype=model_dtype
        )

        # Load classifier head if exists
        if not config['use_prompt_based']:
            classifier_path = load_path / "classifier_head.pt"
            if classifier_path.exists():
                model.classifier.load_state_dict(
                    torch.load(classifier_path, map_location=model.device)
                )

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


if __name__ == "__main__":
    """Test the model."""
    print("Testing LFMDriverClassifier...\n")

    # Test model initialization
    try:
        model = LFMDriverClassifier(
            num_classes=2,
            freeze_vision_encoder=True,
            freeze_language_model=False,
            device="auto"
        )

        model.print_parameter_stats()

        print("\n✓ Model initialization test passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nNote: Make sure you have:")
        print("  1. Installed transformers>=4.57.0")
        print("  2. Internet connection to download the model")
        print("  3. Sufficient disk space (~3-4 GB)")
