"""
Network Channel Simulator.

Simulates various network conditions for training and testing:
- SNR variations (AWGN, fading)
- Bandwidth constraints
- Latency profiles
- Bit error rates
- Real-world network profiles (5G, WiFi, LTE, Satellite)
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum
import random


class NetworkType(Enum):
    """Predefined network types."""
    FIBER = "fiber"
    WIFI_5G = "wifi_5g"
    WIFI_24G = "wifi_24g"
    LTE = "lte"
    MOBILE_5G = "5g"
    SATELLITE = "satellite"
    IOT_LORA = "lora"
    EDGE = "edge"
    CUSTOM = "custom"


@dataclass
class ChannelState:
    """Current state of the simulated channel."""
    snr_db: float
    bandwidth_hz: float
    latency_ms: float
    ber: float
    packet_loss_rate: float = 0.0
    jitter_ms: float = 0.0
    network_type: NetworkType = NetworkType.CUSTOM
    
    def to_tensor(self, device='cpu') -> torch.Tensor:
        """Convert to tensor for model input."""
        return torch.tensor([
            self.snr_db,
            self.bandwidth_hz / 1e6,  # MHz
            self.latency_ms,
            -np.log10(max(self.ber, 1e-10))  # BER exponent
        ], device=device, dtype=torch.float32)
    
    def to_dict(self) -> dict:
        return {
            'snr_db': self.snr_db,
            'bandwidth_mhz': self.bandwidth_hz / 1e6,
            'latency_ms': self.latency_ms,
            'ber': self.ber,
            'packet_loss': self.packet_loss_rate,
            'jitter_ms': self.jitter_ms,
            'network_type': self.network_type.value
        }


@dataclass
class NetworkProfile:
    """Profile defining network characteristics and variations."""
    name: str
    network_type: NetworkType
    
    # Mean values
    snr_mean: float
    bandwidth_hz: float
    latency_mean_ms: float
    ber_mean: float
    
    # Variations (std dev or range)
    snr_std: float = 2.0
    latency_std_ms: float = 5.0
    ber_variation: float = 0.5  # log scale
    
    # Fading parameters
    fading_type: str = 'none'  # 'none', 'slow', 'fast', 'rayleigh'
    fading_strength: float = 0.0
    
    # Additional characteristics
    packet_loss_rate: float = 0.0
    jitter_ms: float = 0.0


# Predefined network profiles
NETWORK_PROFILES = {
    NetworkType.FIBER: NetworkProfile(
        name="Fiber Optic",
        network_type=NetworkType.FIBER,
        snr_mean=35.0, snr_std=1.0,
        bandwidth_hz=1000e6,
        latency_mean_ms=2.0, latency_std_ms=0.5,
        ber_mean=1e-12, ber_variation=0.2,
        fading_type='none'
    ),
    
    NetworkType.MOBILE_5G: NetworkProfile(
        name="5G Mobile",
        network_type=NetworkType.MOBILE_5G,
        snr_mean=22.0, snr_std=5.0,
        bandwidth_hz=100e6,
        latency_mean_ms=8.0, latency_std_ms=3.0,
        ber_mean=1e-6, ber_variation=0.5,
        fading_type='fast',
        fading_strength=0.3,
        jitter_ms=2.0
    ),
    
    NetworkType.WIFI_5G: NetworkProfile(
        name="WiFi 5GHz",
        network_type=NetworkType.WIFI_5G,
        snr_mean=18.0, snr_std=4.0,
        bandwidth_hz=40e6,
        latency_mean_ms=15.0, latency_std_ms=5.0,
        ber_mean=1e-5, ber_variation=0.5,
        fading_type='slow',
        fading_strength=0.2,
        packet_loss_rate=0.001
    ),
    
    NetworkType.WIFI_24G: NetworkProfile(
        name="WiFi 2.4GHz",
        network_type=NetworkType.WIFI_24G,
        snr_mean=15.0, snr_std=5.0,
        bandwidth_hz=20e6,
        latency_mean_ms=25.0, latency_std_ms=10.0,
        ber_mean=1e-4, ber_variation=0.6,
        fading_type='slow',
        fading_strength=0.3,
        packet_loss_rate=0.005,
        jitter_ms=5.0
    ),
    
    NetworkType.LTE: NetworkProfile(
        name="LTE 4G",
        network_type=NetworkType.LTE,
        snr_mean=12.0, snr_std=4.0,
        bandwidth_hz=20e6,
        latency_mean_ms=40.0, latency_std_ms=15.0,
        ber_mean=1e-4, ber_variation=0.5,
        fading_type='fast',
        fading_strength=0.4,
        jitter_ms=10.0
    ),
    
    NetworkType.SATELLITE: NetworkProfile(
        name="Satellite (GEO)",
        network_type=NetworkType.SATELLITE,
        snr_mean=8.0, snr_std=3.0,
        bandwidth_hz=10e6,
        latency_mean_ms=600.0, latency_std_ms=50.0,
        ber_mean=1e-3, ber_variation=0.5,
        fading_type='slow',
        fading_strength=0.5,
        packet_loss_rate=0.01
    ),
    
    NetworkType.IOT_LORA: NetworkProfile(
        name="LoRa IoT",
        network_type=NetworkType.IOT_LORA,
        snr_mean=5.0, snr_std=3.0,
        bandwidth_hz=0.125e6,  # 125 kHz
        latency_mean_ms=200.0, latency_std_ms=100.0,
        ber_mean=1e-3, ber_variation=0.8,
        fading_type='slow',
        fading_strength=0.6
    ),
    
    NetworkType.EDGE: NetworkProfile(
        name="Edge Network (Poor)",
        network_type=NetworkType.EDGE,
        snr_mean=3.0, snr_std=2.0,
        bandwidth_hz=1e6,
        latency_mean_ms=150.0, latency_std_ms=50.0,
        ber_mean=1e-2, ber_variation=0.5,
        fading_type='fast',
        fading_strength=0.7,
        packet_loss_rate=0.05,
        jitter_ms=30.0
    ),
}


class ChannelSimulator:
    """
    Simulates realistic network channel conditions.
    
    Features:
    - Predefined network profiles (5G, WiFi, LTE, etc.)
    - Time-varying conditions (fading)
    - Random sampling for training
    - Scenario generation
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
    
    def get_profile(self, network_type: NetworkType) -> NetworkProfile:
        """Get predefined network profile."""
        return NETWORK_PROFILES.get(network_type)
    
    def sample_from_profile(self, profile: NetworkProfile) -> ChannelState:
        """Sample channel state from profile with random variations."""
        # SNR with Gaussian variation
        snr = np.random.normal(profile.snr_mean, profile.snr_std)
        snr = np.clip(snr, 0, 40)
        
        # Apply fading if enabled
        if profile.fading_type != 'none':
            if profile.fading_type == 'rayleigh':
                fading = np.random.rayleigh(profile.fading_strength)
            else:
                fading = np.random.uniform(-profile.fading_strength, profile.fading_strength)
            snr = snr * (1 + fading)
            snr = np.clip(snr, 0, 40)
        
        # Latency with variation
        latency = np.random.normal(profile.latency_mean_ms, profile.latency_std_ms)
        latency = max(1.0, latency)
        
        # BER with log-scale variation
        ber_log = np.log10(profile.ber_mean)
        ber_log += np.random.normal(0, profile.ber_variation)
        ber = 10 ** ber_log
        ber = np.clip(ber, 1e-12, 1e-1)
        
        return ChannelState(
            snr_db=float(snr),
            bandwidth_hz=profile.bandwidth_hz,
            latency_ms=float(latency),
            ber=float(ber),
            packet_loss_rate=profile.packet_loss_rate,
            jitter_ms=profile.jitter_ms,
            network_type=profile.network_type
        )
    
    def sample_random(
        self,
        snr_range: Tuple[float, float] = (2, 30),
        bw_range: Tuple[float, float] = (1e6, 100e6),
        latency_range: Tuple[float, float] = (5, 500),
        ber_range: Tuple[float, float] = (1e-6, 1e-2)
    ) -> ChannelState:
        """Sample completely random channel conditions."""
        snr = np.random.uniform(*snr_range)
        bandwidth = np.random.uniform(*bw_range)
        latency = np.random.uniform(*latency_range)
        ber = 10 ** np.random.uniform(np.log10(ber_range[0]), np.log10(ber_range[1]))
        
        return ChannelState(
            snr_db=float(snr),
            bandwidth_hz=float(bandwidth),
            latency_ms=float(latency),
            ber=float(ber),
            network_type=NetworkType.CUSTOM
        )
    
    def sample_batch(
        self,
        batch_size: int,
        network_type: Optional[NetworkType] = None,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, List[ChannelState]]:
        """
        Sample a batch of channel conditions for training.
        
        Returns:
            tensor: [B, 4] tensor of (SNR, BW_MHz, latency_ms, BER_exp)
            states: List of ChannelState objects
        """
        states = []
        
        for _ in range(batch_size):
            if network_type is not None:
                profile = self.get_profile(network_type)
                state = self.sample_from_profile(profile)
            else:
                # Random profile selection
                if np.random.rand() < 0.7:
                    # 70% from predefined profiles
                    profile = random.choice(list(NETWORK_PROFILES.values()))
                    state = self.sample_from_profile(profile)
                else:
                    # 30% completely random
                    state = self.sample_random()
            
            states.append(state)
        
        # Build tensor
        tensors = [s.to_tensor(device) for s in states]
        batch_tensor = torch.stack(tensors)
        
        return batch_tensor, states
    
    def generate_time_varying(
        self,
        profile: NetworkProfile,
        duration_seconds: float,
        sample_rate_hz: float = 10.0
    ) -> List[ChannelState]:
        """
        Generate time-varying channel conditions.
        
        Simulates realistic SNR variations over time.
        """
        num_samples = int(duration_seconds * sample_rate_hz)
        states = []
        
        # Start with base state
        current_snr = profile.snr_mean
        
        for _ in range(num_samples):
            # Random walk for SNR
            snr_change = np.random.normal(0, profile.snr_std * 0.1)
            current_snr += snr_change
            
            # Apply mean reversion
            current_snr += 0.1 * (profile.snr_mean - current_snr)
            current_snr = np.clip(current_snr, 0, 40)
            
            # Apply fading
            if profile.fading_type == 'fast':
                fading = np.random.rayleigh(profile.fading_strength)
                snr_with_fading = current_snr * (1 - fading * 0.5)
            else:
                snr_with_fading = current_snr
            
            state = ChannelState(
                snr_db=float(snr_with_fading),
                bandwidth_hz=profile.bandwidth_hz,
                latency_ms=profile.latency_mean_ms,
                ber=profile.ber_mean,
                network_type=profile.network_type
            )
            states.append(state)
        
        return states


def demo():
    """Demonstrate channel simulator."""
    print("=" * 60)
    print("Channel Simulator Demo")
    print("=" * 60)
    
    sim = ChannelSimulator(seed=42)
    
    # Show all profiles
    print("\nPredefined Network Profiles:")
    print("-" * 50)
    for net_type, profile in NETWORK_PROFILES.items():
        state = sim.sample_from_profile(profile)
        print(f"{profile.name:<20} SNR={state.snr_db:5.1f}dB, BW={state.bandwidth_hz/1e6:6.1f}MHz, Latency={state.latency_ms:6.1f}ms")
    
    # Sample batch
    print("\n" + "-" * 50)
    print("Random batch sampling (10 samples):")
    tensor, states = sim.sample_batch(10, device='cpu')
    print(f"Tensor shape: {tensor.shape}")
    print(f"Sample: SNR={tensor[0, 0]:.1f}, BW={tensor[0, 1]:.1f}MHz, Lat={tensor[0, 2]:.1f}ms")
    
    # Time-varying
    print("\n" + "-" * 50)
    print("Time-varying channel (5G, 1 second):")
    profile = NETWORK_PROFILES[NetworkType.MOBILE_5G]
    time_states = sim.generate_time_varying(profile, duration_seconds=1.0, sample_rate_hz=10)
    snrs = [s.snr_db for s in time_states]
    print(f"SNR range: {min(snrs):.1f} - {max(snrs):.1f} dB")
    print(f"SNR samples: {[f'{s:.1f}' for s in snrs[:5]]}...")


if __name__ == '__main__':
    demo()
