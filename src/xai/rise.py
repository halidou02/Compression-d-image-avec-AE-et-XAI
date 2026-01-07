"""
RISE: Randomized Input Sampling for Explanation.

Model-agnostic explanation method using random masks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class RISE:
    """
    RISE: Randomized Input Sampling for Explanation.
    
    Generates saliency maps by probing the model with randomly masked inputs.
    Model-agnostic alternative to Grad-CAM.
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int] = (256, 256),
        num_masks: int = 4000,
        mask_prob: float = 0.5,
        mask_size: int = 8
    ):
        """
        Args:
            model: Model to explain
            input_size: Expected input size (H, W)
            num_masks: Number of random masks to generate
            mask_prob: Probability of keeping a cell
            mask_size: Size of each mask cell (before upsampling)
        """
        self.model = model
        self.input_size = input_size
        self.num_masks = num_masks
        self.mask_prob = mask_prob
        self.mask_size = mask_size
        
        self.masks = None
        self._generate_masks()
    
    def _generate_masks(self):
        """Pre-generate random masks."""
        H, W = self.input_size
        cell_size = self.mask_size
        
        # Small mask dimensions
        h = H // cell_size + 1
        w = W // cell_size + 1
        
        masks = []
        for _ in range(self.num_masks):
            # Generate small binary mask
            mask_small = (torch.rand(1, 1, h, w) < self.mask_prob).float()
            
            # Upsample with random shift
            shift_h = np.random.randint(0, cell_size)
            shift_w = np.random.randint(0, cell_size)
            
            mask_up = F.interpolate(
                mask_small,
                size=(h * cell_size, w * cell_size),
                mode='bilinear',
                align_corners=False
            )
            
            # Crop to target size with shift
            mask = mask_up[:, :, shift_h:shift_h+H, shift_w:shift_w+W]
            masks.append(mask)
        
        self.masks = torch.cat(masks, dim=0)  # [N, 1, H, W]
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        batch_size: int = 128
    ) -> torch.Tensor:
        """
        Generate RISE saliency map.
        
        Args:
            input_tensor: Input image [1, 3, H, W]
            target_class: Target class (None uses predicted class)
            batch_size: Batch size for mask evaluation
            
        Returns:
            saliency: Saliency map [H, W]
        """
        self.model.eval()
        device = input_tensor.device
        
        # Get predicted class if not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Move masks to device
        masks = self.masks.to(device)
        
        # Collect predictions
        saliency = torch.zeros(self.input_size, device=device)
        norm = torch.zeros(self.input_size, device=device)
        
        num_batches = (self.num_masks + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, self.num_masks)
            
            batch_masks = masks[start:end]  # [B, 1, H, W]
            
            # Apply masks to input
            masked_inputs = input_tensor * batch_masks
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(masked_inputs)
                probs = F.softmax(outputs, dim=1)
                scores = probs[:, target_class]  # [B]
            
            # Accumulate weighted masks
            for j, (mask, score) in enumerate(zip(batch_masks, scores)):
                saliency += mask.squeeze() * score
                norm += mask.squeeze()
        
        # Normalize
        saliency = saliency / (norm + 1e-8)
        
        # Normalize to [0, 1]
        saliency = saliency - saliency.min()
        saliency = saliency / (saliency.max() + 1e-8)
        
        return saliency


class RISEBatch(RISE):
    """
    Batch-optimized RISE implementation.
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        num_masks: Optional[int] = None
    ) -> torch.Tensor:
        """
        Faster RISE using vectorized operations.
        """
        self.model.eval()
        device = input_tensor.device
        
        if num_masks is None:
            num_masks = self.num_masks
        
        # Limit masks if specified
        masks = self.masks[:num_masks].to(device)
        
        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Process all masks
        with torch.no_grad():
            # Expand input: [1, 3, H, W] -> [N, 3, H, W]
            inputs_masked = input_tensor.expand(num_masks, -1, -1, -1) * masks
            
            # Get predictions in batches
            scores = []
            batch_size = 64
            for i in range(0, num_masks, batch_size):
                batch = inputs_masked[i:i+batch_size]
                out = self.model(batch)
                probs = F.softmax(out, dim=1)
                scores.append(probs[:, target_class])
            
            scores = torch.cat(scores)  # [N]
        
        # Compute weighted sum
        saliency = (masks.squeeze(1) * scores.view(-1, 1, 1)).sum(dim=0)
        norm = masks.squeeze(1).sum(dim=0)
        
        saliency = saliency / (norm + 1e-8)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency


if __name__ == "__main__":
    import torchvision.models as models
    
    # Test RISE
    resnet = models.resnet18(pretrained=True)
    rise = RISE(resnet, input_size=(224, 224), num_masks=100)  # Small for testing
    
    x = torch.randn(1, 3, 224, 224)
    
    saliency = rise(x)
    print(f"Saliency shape: {saliency.shape}")
    print(f"Saliency range: [{saliency.min():.3f}, {saliency.max():.3f}]")
