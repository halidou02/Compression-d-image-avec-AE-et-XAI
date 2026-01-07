"""
Grad-CAM Implementation.

Gradient-weighted Class Activation Mapping for generating importance heatmaps.
Used to guide semantic importance scoring in SR-SC module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    
    Generates attention heatmaps highlighting important regions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module
    ):
        """
        Args:
            model: The neural network model
            target_layer: Layer to compute Grad-CAM for (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Grad-CAM.
        
        Args:
            input_tensor: Input image [1, 3, H, W]
            target_class: Target class index (None for predicted class)
            
        Returns:
            heatmap: Grad-CAM heatmap [H, W]
            output: Model output
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        return cam.squeeze(0).squeeze(0), output


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: Improved version with better localization.
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Grad-CAM++."""
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Grad-CAM++ weights
        grads = self.gradients
        acts = self.activations
        
        # Second and third order gradients
        grads_2 = grads ** 2
        grads_3 = grads ** 3
        
        # Alpha coefficients
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        alpha_num = grads_2
        alpha_denom = 2 * grads_2 + sum_acts * grads_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # Weights
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        
        # CAM
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Upsample
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        return cam.squeeze(0).squeeze(0), output


class AuxiliaryClassifier(nn.Module):
    """
    Simple auxiliary classifier for Grad-CAM.
    Used when main model doesn't have classification head.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 80,  # COCO has 80 classes
        pretrained: bool = True
    ):
        super().__init__()
        
        self.encoder = encoder
        
        # Simple classifier head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)  # Assuming 512-dim features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled = self.gap(features).squeeze(-1).squeeze(-1)
        return self.classifier(pooled)
    
    def get_target_layer(self) -> nn.Module:
        """Get target layer for Grad-CAM (last conv in encoder)."""
        # For ResNet-18, this is the last block
        if hasattr(self.encoder, 'backbone'):
            return self.encoder.backbone[-1]
        return self.encoder


def compute_importance_from_gradcam(
    heatmap: torch.Tensor,
    features: torch.Tensor
) -> torch.Tensor:
    """
    Convert Grad-CAM heatmap to channel importance scores.
    
    importance[c] = sum_{h,w}(heatmap[h,w] * |features[c,h,w]|) / sum(heatmap)
    
    Args:
        heatmap: Grad-CAM heatmap [H', W'] (already upsampled to feature size)
        features: Feature maps [C, H, W]
        
    Returns:
        importance: [C] importance score per channel
    """
    C, H, W = features.shape
    
    # Resize heatmap if needed
    if heatmap.shape != (H, W):
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
    
    # Normalize heatmap
    heatmap_sum = heatmap.sum() + 1e-8
    heatmap_normalized = heatmap / heatmap_sum
    
    # Compute weighted importance per channel
    importance = (heatmap_normalized.unsqueeze(0) * torch.abs(features)).sum(dim=(1, 2))
    
    return importance


if __name__ == "__main__":
    import torchvision.models as models
    
    # Test Grad-CAM
    resnet = models.resnet18(pretrained=True)
    target_layer = resnet.layer4[-1]
    
    gradcam = GradCAM(resnet, target_layer)
    
    # Dummy input
    x = torch.randn(1, 3, 224, 224)
    
    heatmap, output = gradcam(x)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Output shape: {output.shape}")
