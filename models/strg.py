"""
Spectro-Topographic Relational Graph (STRG) construction.

Nodes are electrode--frequency pairs (or electrodes, for the electrode-only
ablation). Edges are the union of a fixed spatial k-NN prior and a
window/band-specific Top-k functional-dependency proposal. The two relation
values are kept separate as e_ij = [A^S_ij, A^F_ij].
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from utils.graph_utils import (
    build_spatial_adjacency_from_positions,
    expand_channel_graph_to_frequency_blocks,
    get_standard_10_20_positions,
)


class STRG(nn.Module):
    """Construct per-window Spectro-Topographic Relational Graphs."""

    def __init__(
        self,
        num_channels: int,
        num_frequency_bands: int = 5,
        k_spatial: int = 6,
        k_functional: int = 6,
        use_spatial_topology: bool = True,
        use_functional_connectivity: bool = True,
        use_frequency_nodes: bool = True,
        dynamic_functional: bool = True,
        device: str = 'cuda',
        electrode_positions: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        """
        Args:
            num_channels: Number of EEG electrodes
            num_frequency_bands: Number of canonical bands (must be 5)
            k_spatial: k_s, spatial nearest neighbors per electrode
            k_functional: k_f, strongest non-self correlations kept per electrode
            use_spatial_topology: Include the spatial k-NN prior
            use_functional_connectivity: Include correlation-derived relations
            use_frequency_nodes: If True, nodes are (electrode, band) pairs;
                if False, nodes are electrodes with bandpowers as features
            dynamic_functional: If True, A^F is computed per window; if False,
                one sentence-level A^F is reused across windows
            electrode_positions: Optional (C, 3) coordinates from ZuCo chanlocs
            alpha, beta: Ignored (kept so older configs/scripts still construct)
        """
        super().__init__()
        del alpha, beta
        self.num_channels = num_channels
        self.num_frequency_bands = num_frequency_bands
        self.k_spatial = k_spatial
        self.k_functional = k_functional
        self.use_spatial_topology = use_spatial_topology
        self.use_functional_connectivity = use_functional_connectivity
        self.use_frequency_nodes = use_frequency_nodes
        self.dynamic_functional = dynamic_functional
        self.device = device
        self.electrode_positions = electrode_positions

        assert num_frequency_bands == 5, (
            f"num_frequency_bands must be 5 (got {num_frequency_bands}). "
            "Standard bands: delta, theta, alpha, beta, gamma"
        )
        self.frequency_band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        self.num_nodes = (
            num_channels * num_frequency_bands if use_frequency_nodes else num_channels
        )
        self.node_dim = 1 if use_frequency_nodes else num_frequency_bands
        self.edge_dim = 2

        self.register_buffer('A_spatial_channels', self._build_spatial_channels())
        self.register_buffer('A_spatial', self._expand_spatial(self.A_spatial_channels))

    def _build_spatial_channels(self) -> torch.Tensor:
        if self.electrode_positions is not None:
            if isinstance(self.electrode_positions, torch.Tensor):
                positions = self.electrode_positions.detach().cpu().numpy()
            else:
                positions = np.asarray(self.electrode_positions)
            if positions.shape[0] != self.num_channels:
                raise ValueError(
                    f"Electrode positions shape {positions.shape} doesn't match "
                    f"num_channels {self.num_channels}"
                )
        else:
            positions = get_standard_10_20_positions(self.num_channels)

        A = build_spatial_adjacency_from_positions(
            positions,
            k_nearest=self.k_spatial
        )
        A = torch.from_numpy(A).float()
        A = (A + A.T) / 2
        A.fill_diagonal_(1.0)
        return A

    def _expand_spatial(self, A_channels: torch.Tensor) -> torch.Tensor:
        if self.use_frequency_nodes:
            return expand_channel_graph_to_frequency_blocks(
                A_channels, self.num_frequency_bands
            )
        return A_channels

    def set_electrode_positions(self, electrode_positions: torch.Tensor):
        self.electrode_positions = electrode_positions
        A_ch = self._build_spatial_channels()
        self.register_buffer('A_spatial_channels', A_ch)
        self.register_buffer('A_spatial', self._expand_spatial(A_ch))

    def _ensure_windowed(self, eeg_bands: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        windowed = {}
        for name, band in eeg_bands.items():
            if band.dim() == 2:
                windowed[name] = band.unsqueeze(0).unsqueeze(0)
            elif band.dim() == 3:
                windowed[name] = band.unsqueeze(1)
            elif band.dim() == 4:
                windowed[name] = band
            else:
                raise ValueError(
                    f"Band {name} has rank {band.dim()}; expected 2, 3, or 4"
                )
        return windowed

    def _compute_bandpower(self, band_data: torch.Tensor) -> torch.Tensor:
        """Log-variance over time. Raw variance is too small for GAT after z-scored EEG."""
        power = torch.var(band_data, dim=-1, keepdim=False).clamp_min(1e-8)
        return torch.log(power)

    def _topk_correlation(self, band_data: torch.Tensor) -> torch.Tensor:
        """
        Per-row Top-k absolute Pearson correlation, excluding self.

        Args:
            band_data: (..., C, T)

        Returns:
            A_functional: (..., C, C) with A_cc' = |R_cc'| if c' in TopK of c
        """
        num_channels = band_data.shape[-2]
        time_steps = band_data.shape[-1]
        denom = max(time_steps - 1, 1)

        centered = band_data - band_data.mean(dim=-1, keepdim=True)
        std = band_data.std(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized = centered / std
        corr = torch.matmul(normalized, normalized.transpose(-1, -2)) / denom
        corr = corr.abs()

        eye = torch.eye(num_channels, device=band_data.device, dtype=band_data.dtype)
        corr = corr * (1.0 - eye)

        k = min(self.k_functional, max(num_channels - 1, 1))
        topk_vals, topk_idx = torch.topk(corr, k, dim=-1)
        A = torch.zeros_like(corr)
        A.scatter_(-1, topk_idx, topk_vals)
        return A

    def _functional_per_band(self, eeg_bands: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            eeg_bands: each (B, W, C, T) or, for static full-sentence, (B, 1, C, T)

        Returns:
            A_func_band: (B, W, F, C, C)
        """
        per_band = []
        for name in self.frequency_band_names:
            if name not in eeg_bands:
                raise ValueError(f"Missing frequency band: {name}")
            per_band.append(self._topk_correlation(eeg_bands[name]))
        return torch.stack(per_band, dim=2)

    def _node_features(self, eeg_bands: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns:
            node_features: (B, W, N, node_dim)
            bandpowers: (B, W, C, F)
        """
        powers = []
        for name in self.frequency_band_names:
            powers.append(self._compute_bandpower(eeg_bands[name]))
        bandpowers = torch.stack(powers, dim=-1)  # (B, W, C, F)

        if self.use_frequency_nodes:
            # Node order: (ch0, delta), (ch1, delta), ..., (ch0, theta), ...
            node_features = bandpowers.permute(0, 1, 3, 2).reshape(
                bandpowers.shape[0], bandpowers.shape[1], -1, 1
            )
        else:
            node_features = bandpowers  # (B, W, C, F)
        # Per-window z-score so spatial/spectral pattern is not washed out by scale.
        mu = node_features.mean(dim=2, keepdim=True)
        sd = node_features.std(dim=2, keepdim=True).clamp_min(1e-6)
        node_features = (node_features - mu) / sd
        return node_features, bandpowers

    def _expand_functional(self, A_func_band: torch.Tensor) -> torch.Tensor:
        """(B, W, F, C, C) -> (B, W, N, N)."""
        if self.use_frequency_nodes:
            return expand_channel_graph_to_frequency_blocks(
                A_func_band, self.num_frequency_bands
            )
        return A_func_band.mean(dim=2)

    def _relation_tensors(
        self,
        A_spatial: torch.Tensor,
        A_functional: torch.Tensor,
        batch_size: int,
        num_windows: int,
        device: torch.device,
    ):
        if self.use_spatial_topology:
            A_S = A_spatial.to(device).unsqueeze(0).unsqueeze(0)
            A_S = A_S.expand(batch_size, num_windows, -1, -1)
        else:
            A_S = torch.zeros_like(A_functional)

        if not self.use_functional_connectivity:
            A_F = torch.zeros_like(A_S if self.use_spatial_topology else A_functional)
            if not self.use_spatial_topology:
                A_S = torch.zeros(
                    batch_size, num_windows, self.num_nodes, self.num_nodes, device=device
                )
                A_F = A_S.clone()
        else:
            A_F = A_functional

        if not self.use_spatial_topology and self.use_functional_connectivity:
            A_S = torch.zeros_like(A_F)

        # Self-loops so every node remains in its own neighborhood
        A_S = A_S.clone()
        node_index = torch.arange(self.num_nodes, device=device)
        A_S[:, :, node_index, node_index] = 1.0

        edge_attr = torch.stack([A_S, A_F], dim=-1)
        edge_mask = ((A_S > 0) | (A_F > 0)).float()
        return edge_attr, edge_mask, A_S, A_F

    def forward(
        self,
        eeg_bands: Dict[str, torch.Tensor],
        eeg_bands_full: Optional[Dict[str, torch.Tensor]] = None,
        eeg_windows: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            eeg_bands: Dict of windowed (or unwindowed) band tensors.
                Accepted shapes per band: (C, T), (B, C, T), or (B, W, C, T)
            eeg_bands_full: Optional full-sentence bands (B, C, T) used when
                dynamic_functional is False
            eeg_windows: Optional broadband windows (B, W, C, T) used for
                electrode-only functional graphs

        Returns:
            dict with edge_attr, edge_mask, node_features, bandpowers,
            A_spatial, A_functional
        """
        eeg_bands = self._ensure_windowed(eeg_bands)
        first = eeg_bands[self.frequency_band_names[0]]
        batch_size, num_windows, num_channels, _ = first.shape
        if num_channels != self.num_channels:
            raise ValueError(
                f"Input has {num_channels} channels but model expects {self.num_channels}"
            )
        device = first.device

        node_features, bandpowers = self._node_features(eeg_bands)

        if self.use_functional_connectivity:
            if not self.dynamic_functional:
                if eeg_bands_full is not None:
                    full = self._ensure_windowed(eeg_bands_full)
                    # Collapse any accidental W>1 by concatenating time
                    full_collapsed = {}
                    for name, band in full.items():
                        B, W, C, T = band.shape
                        full_collapsed[name] = band.permute(0, 2, 1, 3).reshape(B, 1, C, W * T)
                    A_func_band = self._functional_per_band(full_collapsed)
                    A_func_band = A_func_band.expand(-1, num_windows, -1, -1, -1)
                else:
                    A_func_band = self._functional_per_band(eeg_bands)
                    A_func_mean = A_func_band.mean(dim=1, keepdim=True)
                    A_func_band = A_func_mean.expand(-1, num_windows, -1, -1, -1).contiguous()
            elif (not self.use_frequency_nodes) and eeg_windows is not None:
                windows = eeg_windows
                if windows.dim() == 3:
                    windows = windows.unsqueeze(1)
                A_func_ch = self._topk_correlation(windows)  # (B, W, C, C)
                A_func_band = A_func_ch.unsqueeze(2).expand(
                    -1, -1, self.num_frequency_bands, -1, -1
                )
            else:
                A_func_band = self._functional_per_band(eeg_bands)
            A_functional = self._expand_functional(A_func_band)
        else:
            A_functional = torch.zeros(
                batch_size, num_windows, self.num_nodes, self.num_nodes, device=device
            )

        edge_attr, edge_mask, A_S, A_F = self._relation_tensors(
            self.A_spatial, A_functional, batch_size, num_windows, device
        )

        return {
            'edge_attr': edge_attr,
            'edge_mask': edge_mask,
            'node_features': node_features,
            'bandpowers': bandpowers,
            'A': edge_mask,
        }

    def extract_strg_components(
        self,
        eeg_bands: Dict[str, torch.Tensor],
        return_separate: bool = False,
        **kwargs
    ):
        """Visualization helper. Uses the first window of the first sample."""
        outputs = self.forward(eeg_bands, **kwargs)
        if not return_separate:
            return outputs['edge_mask'], outputs['node_features'], outputs['bandpowers']

        A_S = outputs['edge_attr'][0, 0, ..., 0]
        A_F = outputs['edge_attr'][0, 0, ..., 1]
        A_union = outputs['edge_mask'][0, 0]
        bandpowers = outputs['bandpowers'][:, 0]
        node_features = outputs['node_features'][:, 0]

        A_functional_per_band = {}
        windowed = self._ensure_windowed(eeg_bands)
        first_window = {k: v[:, :1] for k, v in windowed.items()}
        if self.use_functional_connectivity:
            A_func_band = self._functional_per_band(first_window)[0, 0]
            for f_idx, name in enumerate(self.frequency_band_names):
                A_functional_per_band[name] = A_func_band[f_idx].detach().cpu().numpy()

        return {
            'A_combined': A_union.detach().unsqueeze(0).cpu().numpy(),
            'A_spatial': A_S.detach().cpu().numpy(),
            'A_functional_per_band': A_functional_per_band,
            'A_functional_full': A_F.detach().cpu().numpy(),
            'node_features': node_features.detach().cpu().numpy(),
            'bandpowers': bandpowers.detach().cpu().numpy(),
            'edge_attr': outputs['edge_attr'][:, 0].detach().cpu().numpy(),
        }
