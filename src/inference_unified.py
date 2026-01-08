"""
Unified Inference Interface for AdaptiveJSCC.

Features:
- Upload image
- Configure network conditions (SNR, bandwidth, latency, BER)
- Automatic rate selection based on network conditions
- Manual rate override option
- Display reconstruction + metrics
- Visualize rate selection across different network scenarios
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

# Models
from src.models.adaptive_jscc import AdaptiveJSCC, NetworkConditions
from src.utils.metrics import compute_psnr, compute_ssim


# ============================================================================
# Global State
# ============================================================================

MODEL = None
DEVICE = None


def load_model(jscc_path: str = None, prenet_path: str = None):
    """Load unified AdaptiveJSCC model."""
    global MODEL, DEVICE
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Default paths
    root = Path(__file__).parent.parent
    if jscc_path is None:
        jscc_path = root / 'checkpoints' / 'best_noskip.pt'
    if prenet_path is None:
        prenet_path = root / 'results' / 'prenet_adaptive' / 'best_prenet_adaptive.pt'
    
    # Load model
    MODEL = AdaptiveJSCC.from_pretrained(
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
        status = load_model()
    
    if image is None:
        return None, None, "Please upload an image"
    
    # Preprocess
    img_tensor = preprocess_image(image)
    if img_tensor is None:
        return None, None, "Error processing image"
    
    # Create network conditions
    network = NetworkConditions(
        snr_db=snr_db,
        bandwidth_mhz=bandwidth_mhz,
        latency_ms=latency_ms,
        ber_exp=float(ber_exp)
    )
    
    # Forward
    MODEL.eval()
    with torch.no_grad():
        result = MODEL(img_tensor, network=network)
    
    x_hat = result['output']
    rate = result['rate']
    k = result['k']
    
    # Metrics
    psnr = compute_psnr(x_hat, img_tensor).item()
    ssim = compute_ssim(x_hat, img_tensor).item()
    
    # Compression stats
    total_symbols = 256 * 16 * 16
    active_symbols = k * 16 * 16
    compression_ratio = total_symbols / active_symbols
    bandwidth_saved = (1 - rate) * 100
    
    # Output
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
| **Rate (auto)** | **{rate:.2f}** |
| Active Channels | {k} / 256 |

## 📊 Quality
| Metric | Value |
|--------|-------|
| **PSNR** | **{psnr:.2f} dB** |
| **SSIM** | **{ssim:.4f}** |

## 📦 Compression
| Metric | Value |
|--------|-------|
| Ratio | {compression_ratio:.2f}x |
| Bandwidth Saved | {bandwidth_saved:.1f}% |
"""
    
    return recon_np, info, None


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
    
    MODEL.eval()
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
| Active Channels | {k} / 256 |

## 📊 Quality
| Metric | Value |
|--------|-------|
| **PSNR** | **{psnr:.2f} dB** |
| **SSIM** | **{ssim:.4f}** |
"""
    
    return recon_np, info, None


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
    
    # Network scenarios
    scenarios = [
        ("5G", NetworkConditions(snr_db=25, bandwidth_mhz=100, latency_ms=5, ber_exp=6)),
        ("WiFi", NetworkConditions(snr_db=15, bandwidth_mhz=20, latency_ms=30, ber_exp=5)),
        ("LTE", NetworkConditions(snr_db=10, bandwidth_mhz=10, latency_ms=50, ber_exp=4)),
        ("Satellite", NetworkConditions(snr_db=5, bandwidth_mhz=5, latency_ms=500, ber_exp=3)),
    ]
    
    # Collect results
    results = []
    MODEL.eval()
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
    
    # Create comparison figure
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
    
    # Rate comparison bar chart
    ax = axes[1, 2]
    names = [r['name'] for r in results]
    rates = [r['rate'] for r in results]
    psnrs = [r['psnr'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rates, width, label='Rate', color='steelblue')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, psnrs, width, label='PSNR', color='coral')
    
    ax.set_xlabel('Network')
    ax.set_ylabel('Rate', color='steelblue')
    ax2.set_ylabel('PSNR (dB)', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax2.set_ylim(20, 35)
    ax.set_title('Rate & Quality Comparison')
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    comparison_img = np.array(Image.open(buf))
    
    # Summary table
    summary = "## 📊 Network Comparison\n\n"
    summary += "| Network | Rate | Channels | PSNR |\n"
    summary += "|---------|------|----------|------|\n"
    for res in results:
        summary += f"| {res['name']} | {res['rate']:.2f} | {res['k']}/256 | {res['psnr']:.1f} dB |\n"
    
    return comparison_img, summary


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create Gradio interface."""
    
    with gr.Blocks(title="AdaptiveJSCC - Semantic Communication", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🛰️ AdaptiveJSCC - Semantic Communication Demo

**Unified model** that automatically selects optimal compression rate based on network conditions.
        """)
        
        with gr.Tabs():
            # Tab 1: Auto Rate
            with gr.TabItem("📡 Auto Rate (Network-Aware)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image_auto = gr.Image(label="📷 Input Image", type="numpy")
                        
                        gr.Markdown("### 🌐 Network Conditions")
                        
                        snr_slider = gr.Slider(
                            minimum=0, maximum=30, value=15, step=1,
                            label="SNR (dB)"
                        )
                        bw_slider = gr.Slider(
                            minimum=1, maximum=100, value=20, step=1,
                            label="Bandwidth (MHz)"
                        )
                        lat_slider = gr.Slider(
                            minimum=5, maximum=500, value=50, step=5,
                            label="Latency (ms)"
                        )
                        ber_slider = gr.Slider(
                            minimum=2, maximum=6, value=4, step=1,
                            label="BER Exponent (10^-x)"
                        )
                        
                        # Preset buttons
                        gr.Markdown("### 🚀 Quick Presets")
                        with gr.Row():
                            preset_5g = gr.Button("5G", size="sm")
                            preset_wifi = gr.Button("WiFi", size="sm")
                            preset_lte = gr.Button("LTE", size="sm")
                            preset_sat = gr.Button("Satellite", size="sm")
                        
                        run_btn_auto = gr.Button("🔄 Run Inference", variant="primary")
                    
                    with gr.Column(scale=2):
                        recon_auto = gr.Image(label="🖼️ Reconstruction")
                        info_auto = gr.Markdown(label="Results")
                        error_auto = gr.Textbox(label="Error", visible=False)
                
                # Preset handlers
                preset_5g.click(
                    fn=lambda: (25, 100, 5, 6),
                    outputs=[snr_slider, bw_slider, lat_slider, ber_slider]
                )
                preset_wifi.click(
                    fn=lambda: (15, 20, 30, 5),
                    outputs=[snr_slider, bw_slider, lat_slider, ber_slider]
                )
                preset_lte.click(
                    fn=lambda: (10, 10, 50, 4),
                    outputs=[snr_slider, bw_slider, lat_slider, ber_slider]
                )
                preset_sat.click(
                    fn=lambda: (5, 5, 400, 3),
                    outputs=[snr_slider, bw_slider, lat_slider, ber_slider]
                )
                
                run_btn_auto.click(
                    fn=inference_auto_rate,
                    inputs=[input_image_auto, snr_slider, bw_slider, lat_slider, ber_slider],
                    outputs=[recon_auto, info_auto, error_auto]
                )
            
            # Tab 2: Manual Rate
            with gr.TabItem("🎛️ Manual Rate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image_manual = gr.Image(label="📷 Input Image", type="numpy")
                        
                        rate_slider = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                            label="Rate (compression level)"
                        )
                        snr_manual = gr.Slider(
                            minimum=0, maximum=20, value=10, step=1,
                            label="SNR (dB)"
                        )
                        
                        run_btn_manual = gr.Button("🔄 Run Inference", variant="primary")
                    
                    with gr.Column(scale=2):
                        recon_manual = gr.Image(label="🖼️ Reconstruction")
                        info_manual = gr.Markdown(label="Results")
                        error_manual = gr.Textbox(label="Error", visible=False)
                
                run_btn_manual.click(
                    fn=inference_manual_rate,
                    inputs=[input_image_manual, rate_slider, snr_manual],
                    outputs=[recon_manual, info_manual, error_manual]
                )
            
            # Tab 3: Network Comparison
            with gr.TabItem("📊 Compare Networks"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image_compare = gr.Image(label="📷 Input Image", type="numpy")
                        run_btn_compare = gr.Button("🔄 Compare Networks", variant="primary")
                    
                    with gr.Column(scale=2):
                        compare_output = gr.Image(label="📊 Comparison")
                        compare_summary = gr.Markdown(label="Summary")
                
                run_btn_compare.click(
                    fn=compare_networks,
                    inputs=[input_image_compare],
                    outputs=[compare_output, compare_summary]
                )
        
        gr.Markdown("""
---
### ℹ️ About
- **Model**: AdaptiveJSCC (~12.5M parameters)
- **Rate Selection**: Automatic based on SNR, bandwidth, latency, BER
- **Resolution**: 256×256 pixels
        """)
    
    return demo


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Loading model...")
    status = load_model()
    print(status)
    
    print("Starting interface...")
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7862)
