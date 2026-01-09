"""
Inference Interface for Unified JSCC Model.

Uses the fused UnifiedJSCC model that combines:
- JSCCNoSkip (trained encoder/decoder)
- PreNetAdaptive (rate predictor)

Features:
- Upload image
- Configure network conditions (SNR, bandwidth, latency, BER)
- Automatic rate selection based on network conditions
- Manual rate override option
- Compare across different networks
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import io

# Import unified model
from scripts.fuse_models import UnifiedJSCC, NetworkConditions
from src.utils.metrics import compute_psnr, compute_ssim


# ============================================================================
# Global State
# ============================================================================

MODEL = None
DEVICE = None


def load_model():
    """Load the unified model."""
    global MODEL, DEVICE
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(__file__).parent.parent
    
    # Try unified checkpoint first
    unified_path = root / 'checkpoints' / 'unified_jscc.pt'
    jscc_path = root / 'checkpoints' / 'best_noskip.pt'
    prenet_path = root / 'results' / 'prenet_adaptive' / 'best_prenet_adaptive.pt'
    
    if unified_path.exists():
        # Load from unified checkpoint
        MODEL = UnifiedJSCC(num_channels=256).to(DEVICE)
        ckpt = torch.load(unified_path, map_location=DEVICE, weights_only=False)
        MODEL.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ Loaded unified model from {unified_path}")
    else:
        # Load from separate checkpoints
        MODEL = UnifiedJSCC.from_pretrained(
            jscc_path=str(jscc_path),
            prenet_path=str(prenet_path),
            device=DEVICE
        )
    
    MODEL.eval()
    return f"✅ Model loaded on {DEVICE}"


# ============================================================================
# Image Processing
# ============================================================================

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Convert numpy image to tensor."""
    if image is None:
        return None
    
    try:
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        else:
            img = Image.fromarray(np.array(image).astype(np.uint8)).convert('RGB')
        
        img = img.resize((256, 256), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(DEVICE)
    except Exception as e:
        print(f"Error: {e}")
        return None


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to display image."""
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ============================================================================
# Inference Functions
# ============================================================================

def inference_auto_rate(
    image: np.ndarray,
    snr_db: float,
    bandwidth_mhz: float,
    latency_ms: float,
    ber_exp: int
):
    """Inference with automatic rate selection."""
    global MODEL, DEVICE
    
    if MODEL is None:
        load_model()
    
    if image is None:
        return None, None, "Please upload an image"
    
    img_tensor = preprocess_image(image)
    if img_tensor is None:
        return None, None, "Error processing image"
    
    try:
        network = NetworkConditions(
            snr_db=snr_db,
            bandwidth_mhz=bandwidth_mhz,
            latency_ms=latency_ms,
            ber_exp=float(ber_exp)
        )
        
        with torch.no_grad():
            result = MODEL(img_tensor, network=network)
        
        x_hat = result['output']
        rate = result['rate']
        k = result['k']
        
        psnr = compute_psnr(x_hat, img_tensor).item()
        ssim = compute_ssim(x_hat, img_tensor).item()
        bandwidth_saved = (1 - rate) * 100
        
        recon_np = tensor_to_numpy(x_hat)
        
        info = f"""
## 📡 Network Conditions
| Parameter | Value |
|-----------|-------|
| SNR | {snr_db:.1f} dB |
| Bandwidth | {bandwidth_mhz:.1f} MHz |
| Latency | {latency_ms:.0f} ms |
| BER | 10^-{ber_exp} |

## 🎯 Auto Rate Selection
| Metric | Value |
|--------|-------|
| **Rate** | **{rate:.2f}** |
| Channels | {k} / 256 |

## 📊 Quality
| Metric | Value |
|--------|-------|
| **PSNR** | **{psnr:.2f} dB** |
| **SSIM** | **{ssim:.4f}** |
| BW Saved | {bandwidth_saved:.1f}% |
"""
        return recon_np, info, None
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def inference_manual_rate(
    image: np.ndarray,
    rate: float,
    snr_db: float
):
    """Inference with manual rate."""
    global MODEL, DEVICE
    
    if MODEL is None:
        load_model()
    
    if image is None:
        return None, None, "Please upload an image"
    
    img_tensor = preprocess_image(image)
    if img_tensor is None:
        return None, None, "Error processing image"
    
    try:
        with torch.no_grad():
            result = MODEL(img_tensor, snr_db=snr_db, rate_override=rate)
        
        x_hat = result['output']
        k = result['k']
        
        psnr = compute_psnr(x_hat, img_tensor).item()
        ssim = compute_ssim(x_hat, img_tensor).item()
        
        recon_np = tensor_to_numpy(x_hat)
        
        info = f"""
## 🎛️ Manual Settings
| Parameter | Value |
|-----------|-------|
| Rate | {rate:.2f} |
| SNR | {snr_db:.1f} dB |
| Channels | {k} / 256 |

## 📊 Quality
| Metric | Value |
|--------|-------|
| **PSNR** | **{psnr:.2f} dB** |
| **SSIM** | **{ssim:.4f}** |
"""
        return recon_np, info, None
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def compare_networks(image: np.ndarray):
    """Compare reconstruction across different network conditions."""
    global MODEL, DEVICE
    
    if MODEL is None:
        load_model()
    
    if image is None:
        return None, "Please upload an image"
    
    img_tensor = preprocess_image(image)
    if img_tensor is None:
        return None, "Error processing image"
    
    scenarios = [
        ("5G", NetworkConditions(snr_db=25, bandwidth_mhz=100, latency_ms=5, ber_exp=6)),
        ("WiFi", NetworkConditions(snr_db=15, bandwidth_mhz=20, latency_ms=30, ber_exp=5)),
        ("LTE", NetworkConditions(snr_db=10, bandwidth_mhz=10, latency_ms=50, ber_exp=4)),
        ("Satellite", NetworkConditions(snr_db=5, bandwidth_mhz=5, latency_ms=500, ber_exp=3)),
    ]
    
    results = []
    
    try:
        with torch.no_grad():
            for name, network in scenarios:
                result = MODEL(img_tensor, network=network)
                psnr = compute_psnr(result['output'], img_tensor).item()
                results.append({
                    'name': name,
                    'rate': result['rate'],
                    'psnr': psnr,
                    'k': result['k'],
                    'output': tensor_to_numpy(result['output'])
                })
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].imshow(tensor_to_numpy(img_tensor))
        axes[0, 0].set_title("Original", fontsize=12)
        axes[0, 0].axis('off')
        
        # Reconstructions
        for i, res in enumerate(results):
            ax = axes[(i + 1) // 3, (i + 1) % 3]
            ax.imshow(res['output'])
            ax.set_title(f"{res['name']}\nRate={res['rate']:.2f}, PSNR={res['psnr']:.1f}dB", fontsize=10)
            ax.axis('off')
        
        # Bar chart
        ax = axes[1, 2]
        names = [r['name'] for r in results]
        rates = [r['rate'] for r in results]
        psnrs = [r['psnr'] for r in results]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, rates, width, label='Rate', color='steelblue')
        ax2 = ax.twinx()
        ax2.bar(x + width/2, psnrs, width, label='PSNR', color='coral')
        
        ax.set_xlabel('Network')
        ax.set_ylabel('Rate', color='steelblue')
        ax2.set_ylabel('PSNR (dB)', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1)
        ax2.set_ylim(20, 35)
        ax.set_title('Rate & Quality')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        comparison_img = np.array(Image.open(buf))
        
        summary = "## 📊 Network Comparison\n\n"
        summary += "| Network | Rate | Channels | PSNR |\n"
        summary += "|---------|------|----------|------|\n"
        for res in results:
            summary += f"| {res['name']} | {res['rate']:.2f} | {res['k']}/256 | {res['psnr']:.1f} dB |\n"
        
        return comparison_img, summary
        
    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create Gradio interface."""
    
    with gr.Blocks(title="UnifiedJSCC - Semantic Communication", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🛰️ Unified JSCC - Semantic Communication

**Single model** for automatic rate-adaptive image transmission over wireless channels.
        """)
        
        with gr.Tabs():
            # Tab 1: Auto Rate
            with gr.TabItem("📡 Auto Rate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_auto = gr.Image(label="📷 Input", type="numpy")
                        
                        gr.Markdown("### 🌐 Network")
                        snr = gr.Slider(0, 30, 15, step=1, label="SNR (dB)")
                        bw = gr.Slider(1, 100, 20, step=1, label="Bandwidth (MHz)")
                        lat = gr.Slider(5, 500, 50, step=5, label="Latency (ms)")
                        ber = gr.Slider(2, 6, 4, step=1, label="BER (10^-x)")
                        
                        gr.Markdown("### 🚀 Presets")
                        with gr.Row():
                            btn_5g = gr.Button("5G", size="sm")
                            btn_wifi = gr.Button("WiFi", size="sm")
                            btn_lte = gr.Button("LTE", size="sm")
                            btn_sat = gr.Button("Satellite", size="sm")
                        
                        run_auto = gr.Button("🔄 Run", variant="primary")
                    
                    with gr.Column(scale=2):
                        out_auto = gr.Image(label="🖼️ Reconstruction")
                        info_auto = gr.Markdown()
                        err_auto = gr.Textbox(visible=False)
                
                btn_5g.click(fn=lambda: (25, 100, 5, 6), outputs=[snr, bw, lat, ber])
                btn_wifi.click(fn=lambda: (15, 20, 30, 5), outputs=[snr, bw, lat, ber])
                btn_lte.click(fn=lambda: (10, 10, 50, 4), outputs=[snr, bw, lat, ber])
                btn_sat.click(fn=lambda: (5, 5, 400, 3), outputs=[snr, bw, lat, ber])
                
                run_auto.click(
                    fn=inference_auto_rate,
                    inputs=[input_auto, snr, bw, lat, ber],
                    outputs=[out_auto, info_auto, err_auto]
                )
            
            # Tab 2: Manual
            with gr.TabItem("🎛️ Manual Rate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_manual = gr.Image(label="📷 Input", type="numpy")
                        rate = gr.Slider(0.1, 1.0, 0.5, step=0.05, label="Rate")
                        snr_m = gr.Slider(0, 20, 10, step=1, label="SNR (dB)")
                        run_manual = gr.Button("🔄 Run", variant="primary")
                    
                    with gr.Column(scale=2):
                        out_manual = gr.Image(label="🖼️ Reconstruction")
                        info_manual = gr.Markdown()
                        err_manual = gr.Textbox(visible=False)
                
                run_manual.click(
                    fn=inference_manual_rate,
                    inputs=[input_manual, rate, snr_m],
                    outputs=[out_manual, info_manual, err_manual]
                )
            
            # Tab 3: Compare
            with gr.TabItem("📊 Compare"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_compare = gr.Image(label="📷 Input", type="numpy")
                        run_compare = gr.Button("🔄 Compare Networks", variant="primary")
                    
                    with gr.Column(scale=2):
                        out_compare = gr.Image(label="📊 Comparison")
                        summary = gr.Markdown()
                
                run_compare.click(
                    fn=compare_networks,
                    inputs=[input_compare],
                    outputs=[out_compare, summary]
                )
        
        gr.Markdown("""
---
**Model**: UnifiedJSCC (12.6M params) | **Resolution**: 256×256
        """)
    
    return demo


if __name__ == "__main__":
    print("Loading model...")
    status = load_model()
    print(status)
    
    print("Starting interface...")
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7862)
