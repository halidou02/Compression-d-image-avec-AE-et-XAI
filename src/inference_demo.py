"""
Inference Interface for Semantic Communication.

Features:
- Load trained model
- Adjustable rate and SNR
- Display compression ratio
- Show Grad-CAM heatmap
- Compare input vs reconstruction
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import io

from src.models.xai_pipeline import XAIGuidedSemanticComm
from src.utils.gradcam import GradCAMHook
from src.utils.metrics import compute_psnr, compute_ssim


# Global model
MODEL = None
GRADCAM_HOOK = None
DEVICE = None


def load_model(checkpoint_path: str = None):
    """Load trained model."""
    global MODEL, GRADCAM_HOOK, DEVICE
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL = XAIGuidedSemanticComm(num_channels=256, pretrained_encoder=False).to(DEVICE)
    
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent.parent / 'checkpoints' / 'best_mono.pt'
    
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        MODEL.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path}")
        psnr_val = ckpt.get('grid_psnr', ckpt.get('psnr', 0)); print(f'Grid PSNR: {psnr_val:.2f} dB')
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    MODEL.eval()
    
    # Grad-CAM hook on channel_reduce (256ch)
    GRADCAM_HOOK = GradCAMHook(MODEL.encoder.channel_reduce)
    
    return f"Model loaded on {DEVICE}"


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Convert numpy image to tensor [1, 3, 256, 256]."""
    if image is None:
        return None
    
    try:
        # Handle different input types
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        else:
            img = Image.fromarray(np.array(image).astype(np.uint8)).convert('RGB')
        
        img = img.resize((256, 256), Image.LANCZOS)
        
        # To tensor [0, 1]
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(DEVICE)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy image for display."""
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def create_heatmap_overlay(image: np.ndarray, cam: torch.Tensor, alpha: float = 0.5) -> np.ndarray:
    """Create CAM overlay on image."""
    # Resize CAM to image size
    cam_resized = F.interpolate(
        cam.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()
    
    # Normalize
    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
    
    # Create heatmap
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.imshow(cam_resized, cmap='jet', alpha=alpha)
    plt.axis('off')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    return np.array(Image.open(buf))


def inference(image: np.ndarray, rate: float, snr: float):
    """Run inference and return results."""
    global MODEL, GRADCAM_HOOK, DEVICE
    
    if MODEL is None:
        load_model()
    
    if image is None:
        return None, None, None, "Please upload an image"
    
    # Preprocess
    img_tensor = preprocess_image(image)
    
    # Forward with gradient for CAM
    MODEL.zero_grad()
    with torch.enable_grad():
        result = MODEL(img_tensor, snr_db=snr, rate=rate, return_intermediate=True)
        x_hat = result['output']
        features = result['features']
        
        # Backward for CAM
        loss = ((x_hat - img_tensor) ** 2).sum()
        loss.backward()
        
        # Compute CAM
        cam = GRADCAM_HOOK.compute_cam()
    
    GRADCAM_HOOK.clear()
    MODEL.zero_grad()
    
    # Metrics
    with torch.no_grad():
        psnr = compute_psnr(x_hat, img_tensor).item()
        ssim = compute_ssim(x_hat, img_tensor).item()
    
    # Compression info
    k = max(1, int(round(rate * 256)))
    total_symbols = 256 * 16 * 16  # All channels × spatial
    active_symbols = k * 16 * 16    # Active channels × spatial
    compression_ratio = total_symbols / active_symbols
    bandwidth_savings = (1 - rate) * 100
    
    # Create outputs
    recon_np = tensor_to_numpy(x_hat)
    heatmap_np = create_heatmap_overlay(tensor_to_numpy(img_tensor), cam[0])
    
    # Info text
    info = f"""
## Results
- **PSNR**: {psnr:.2f} dB
- **SSIM**: {ssim:.4f}

## Compression
- **Rate**: {rate:.2f} ({rate*100:.0f}%)
- **Active Channels**: {k} / 256
- **Active Symbols**: {active_symbols:,} / {total_symbols:,}
- **Compression Ratio**: {compression_ratio:.2f}x
- **Bandwidth Saved**: {bandwidth_savings:.1f}%

## Channel
- **SNR**: {snr:.1f} dB
"""
    
    return recon_np, heatmap_np, info, None


def create_interface():
    """Create Gradio interface."""
    
    with gr.Blocks(title="Semantic Communication Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🛰️ Semantic Communication - Inference Demo

Upload an image, adjust **rate** and **SNR**, and see the reconstruction with Grad-CAM heatmap.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Image", type="numpy")
                
                with gr.Row():
                    rate_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                        label="Rate (compression level)"
                    )
                    snr_slider = gr.Slider(
                        minimum=0, maximum=20, value=10, step=1,
                        label="SNR (dB)"
                    )
                
                run_btn = gr.Button("🚀 Run Inference", variant="primary")
            
            with gr.Column(scale=2):
                with gr.Row():
                    recon_image = gr.Image(label="Reconstruction")
                    heatmap_image = gr.Image(label="Grad-CAM Heatmap")
                
                info_output = gr.Markdown(label="Results")
                error_output = gr.Textbox(label="Status", visible=False)
        
        # Examples
        gr.Examples(
            examples=[
                ["examples/cat.jpg", 0.5, 10],
                ["examples/city.jpg", 0.25, 5],
                ["examples/portrait.jpg", 1.0, 20],
            ],
            inputs=[input_image, rate_slider, snr_slider],
            outputs=[recon_image, heatmap_image, info_output, error_output],
            fn=inference,
            cache_examples=False,
        )
        
        run_btn.click(
            fn=inference,
            inputs=[input_image, rate_slider, snr_slider],
            outputs=[recon_image, heatmap_image, info_output, error_output]
        )
    
    return demo


if __name__ == "__main__":
    # Load model
    status = load_model()
    print(status)
    
    # Create examples directory
    examples_dir = Path(__file__).parent.parent / 'examples'
    examples_dir.mkdir(exist_ok=True)
    
    # Launch
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861)
