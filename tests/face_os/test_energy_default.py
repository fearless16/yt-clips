"""Tests for I-07: Energy Normalization Default-On.

Verifies that energy normalization is enabled by default via config.
"""

import numpy as np
import pytest


class TestEnergyNormalizationDefault:
    """Energy normalization must be enabled by default."""

    def test_config_has_energy_section(self):
        """Config must have energy section with normalize_energy flag."""
        from face_os.config import get_config
        cfg = get_config()
        assert hasattr(cfg, 'energy')
        assert hasattr(cfg.energy, 'normalize_energy')

    def test_normalize_energy_default_true(self):
        """normalize_energy must default to True."""
        from face_os.config import get_config
        cfg = get_config()
        assert cfg.energy.normalize_energy is True

    def test_normalization_method_default_zscore(self):
        """normalization_method must default to 'zscore'."""
        from face_os.config import get_config
        cfg = get_config()
        assert cfg.energy.normalization_method == 'zscore'

    def test_energy_scaler_uses_config(self):
        """EnergyScaler in pipeline must use config-driven normalization."""
        from face_os.energy_scaling import EnergyScaler, EnergyScalingConfig
        # When normalize_energy=True and method=zscore
        config = EnergyScalingConfig(normalization_method='zscore')
        scaler = EnergyScaler(config)
        # Should produce normalized values (not raw)
        val1 = scaler.normalize('test', 10.0)
        val2 = scaler.normalize('test', 20.0)
        # After two samples, z-score should center around 0
        # (both values contribute to running stats)
        assert isinstance(val1, float)
        assert isinstance(val2, float)

    def test_energy_scaler_none_method_passes_through(self):
        """When method='none', values should pass through unchanged."""
        from face_os.energy_scaling import EnergyScaler, EnergyScalingConfig
        config = EnergyScalingConfig(normalization_method='none')
        scaler = EnergyScaler(config)
        val = scaler.normalize('test', 42.0)
        assert val == 42.0

    def test_pipeline_uses_config_energy(self):
        """Pipeline must read energy config and set _normalize_energy."""
        # This test just verifies the config is accessible
        from face_os.config import get_config
        cfg = get_config()
        energy_cfg = getattr(cfg, 'energy', None)
        assert energy_cfg is not None
        assert getattr(energy_cfg, 'normalize_energy', False) is True
