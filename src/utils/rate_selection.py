"""
Adaptive Rate Selection based on Network Constraints.

Calculates a channel quality score and determines optimal compression rate.
"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChannelConditions:
    """Network channel characteristics."""
    snr_db: float  # Signal-to-Noise Ratio (dB), typically 0-30
    bandwidth_hz: float = 10e6  # Bandwidth in Hz, default 10 MHz
    latency_ms: float = 50.0  # Round-trip latency in ms
    ber: float = 1e-4  # Bit Error Rate, typically 1e-3 to 1e-6
    fading_type: str = 'none'  # 'none', 'slow', 'fast'


def compute_channel_score(channel: ChannelConditions) -> float:
    """
    Compute a normalized channel quality score [0, 1].
    
    Higher score = better channel = can use lower rate (more compression).
    Lower score = worse channel = need higher rate (more redundancy).
    
    Args:
        channel: ChannelConditions dataclass
        
    Returns:
        score: float in [0, 1], where 1 = excellent, 0 = very poor
    """
    # 1. SNR Score (0-1): higher SNR = better
    # Normalize: 0 dB → 0.0, 20 dB → 1.0
    snr_score = min(1.0, max(0.0, channel.snr_db / 20.0))
    
    # 2. Bandwidth Score (0-1): higher BW = more capacity
    # Normalize: 1 MHz → 0.2, 50 MHz → 1.0
    bw_mhz = channel.bandwidth_hz / 1e6
    bw_score = min(1.0, max(0.2, bw_mhz / 50.0))
    
    # 3. Latency Score (0-1): lower latency = better
    # Normalize: 10 ms → 1.0, 500 ms → 0.2
    latency_score = max(0.2, 1.0 - (channel.latency_ms - 10) / 500)
    
    # 4. BER Score (0-1): lower BER = better
    # Normalize: 1e-6 → 1.0, 1e-2 → 0.0
    ber_log = -math.log10(max(channel.ber, 1e-10))  # 6 for 1e-6, 2 for 1e-2
    ber_score = min(1.0, max(0.0, (ber_log - 2) / 4))
    
    # 5. Fading penalty
    fading_penalty = {
        'none': 1.0,
        'slow': 0.9,
        'fast': 0.7
    }.get(channel.fading_type, 1.0)
    
    # Weighted combination
    # SNR is most important for JSCC
    weights = {
        'snr': 0.50,
        'bw': 0.20,
        'latency': 0.10,
        'ber': 0.15,
        'fading': 0.05
    }
    
    score = (
        weights['snr'] * snr_score +
        weights['bw'] * bw_score +
        weights['latency'] * latency_score +
        weights['ber'] * ber_score +
        weights['fading'] * fading_penalty
    )
    
    return score


def select_optimal_rate(
    channel: ChannelConditions,
    target_psnr: Optional[float] = None,
    min_rate: float = 0.1,
    max_rate: float = 1.0
) -> dict:
    """
    Select optimal compression rate based on channel conditions.
    
    Strategy:
    - Good channel (high score) → HIGH rate (send more, better quality)
    - Bad channel (low score) → LOW rate (compress more to fit bandwidth)
    
    Args:
        channel: Network conditions
        target_psnr: Optional minimum PSNR target (dB)
        min_rate: Minimum allowed rate
        max_rate: Maximum allowed rate
        
    Returns:
        dict with rate, score, and explanation
    """
    score = compute_channel_score(channel)
    
    # Direct relationship: good channel → can send more → high rate
    # rate = min_rate + (max_rate - min_rate) * score
    base_rate = min_rate + (max_rate - min_rate) * score
    
    # Adjust for target PSNR if specified
    # Higher target PSNR → need higher rate
    if target_psnr is not None:
        if target_psnr >= 28:
            psnr_factor = 1.2  # Need more info for high quality
        elif target_psnr >= 25:
            psnr_factor = 1.0
        else:
            psnr_factor = 0.8  # Can afford lower rate
        base_rate *= psnr_factor
    
    # Clamp to valid range
    optimal_rate = max(min_rate, min(max_rate, base_rate))
    
    # Round to nice values: 0.25, 0.5, 0.75, 1.0
    nice_rates = [0.1, 0.25, 0.5, 0.75, 1.0]
    optimal_rate = min(nice_rates, key=lambda x: abs(x - optimal_rate))
    
    return {
        'rate': optimal_rate,
        'channel_score': score,
        'snr_db': channel.snr_db,
        'explanation': _get_explanation(score, optimal_rate, channel)
    }


def _get_explanation(score: float, rate: float, channel: ChannelConditions) -> str:
    """Generate human-readable explanation."""
    if score >= 0.8:
        quality = "excellent"
    elif score >= 0.6:
        quality = "good"
    elif score >= 0.4:
        quality = "moderate"
    elif score >= 0.2:
        quality = "poor"
    else:
        quality = "very poor"
    
    return (
        f"Channel quality: {quality} (score={score:.2f}). "
        f"SNR={channel.snr_db:.1f}dB. "
        f"Recommended rate={rate} for optimal quality/bandwidth trade-off."
    )


# Convenience functions for common scenarios
def select_rate_for_wifi(rssi_dbm: float = -60) -> dict:
    """Select rate for WiFi based on RSSI."""
    # Convert RSSI to approximate SNR
    # RSSI -30 dBm → ~25 dB SNR, RSSI -80 dBm → ~5 dB SNR
    snr = max(0, (rssi_dbm + 90) / 2)
    channel = ChannelConditions(
        snr_db=snr,
        bandwidth_hz=20e6,  # WiFi 20 MHz
        latency_ms=10,
        ber=1e-5,
        fading_type='slow'
    )
    return select_optimal_rate(channel)


def select_rate_for_5g(snr_db: float = 15) -> dict:
    """Select rate for 5G network."""
    channel = ChannelConditions(
        snr_db=snr_db,
        bandwidth_hz=100e6,  # 5G 100 MHz
        latency_ms=5,
        ber=1e-6,
        fading_type='fast'
    )
    return select_optimal_rate(channel)


def select_rate_for_satellite(snr_db: float = 8) -> dict:
    """Select rate for satellite link."""
    channel = ChannelConditions(
        snr_db=snr_db,
        bandwidth_hz=5e6,  # Satellite 5 MHz
        latency_ms=600,  # High latency
        ber=1e-3,
        fading_type='slow'
    )
    return select_optimal_rate(channel)


if __name__ == '__main__':
    print("=" * 60)
    print("Adaptive Rate Selection - Demo")
    print("=" * 60)
    
    # Test different scenarios
    scenarios = [
        ("Excellent (5G, high SNR)", ChannelConditions(snr_db=25, bandwidth_hz=100e6, latency_ms=5, ber=1e-6)),
        ("Good (WiFi, medium SNR)", ChannelConditions(snr_db=15, bandwidth_hz=20e6, latency_ms=20, ber=1e-5)),
        ("Moderate (LTE, low SNR)", ChannelConditions(snr_db=10, bandwidth_hz=10e6, latency_ms=50, ber=1e-4)),
        ("Poor (Satellite)", ChannelConditions(snr_db=5, bandwidth_hz=5e6, latency_ms=500, ber=1e-3)),
        ("Very Poor (Edge)", ChannelConditions(snr_db=2, bandwidth_hz=1e6, latency_ms=200, ber=5e-3)),
    ]
    
    print("\n{:<25} {:<8} {:<8} {}".format("Scenario", "Score", "Rate", "Explanation"))
    print("-" * 80)
    
    for name, channel in scenarios:
        result = select_optimal_rate(channel)
        print(f"{name:<25} {result['channel_score']:.2f}     {result['rate']:<8} SNR={channel.snr_db}dB")
    
    print("\n" + "=" * 60)
    print("Convenience Functions")
    print("=" * 60)
    
    print(f"\nWiFi (RSSI=-50dBm): {select_rate_for_wifi(-50)['rate']}")
    print(f"WiFi (RSSI=-70dBm): {select_rate_for_wifi(-70)['rate']}")
    print(f"5G (SNR=20dB): {select_rate_for_5g(20)['rate']}")
    print(f"Satellite: {select_rate_for_satellite()['rate']}")
