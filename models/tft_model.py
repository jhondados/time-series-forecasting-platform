"""Simplified Temporal Fusion Transformer implementation."""
import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

def create_tft_model(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """Create production-ready TFT model."""
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=3e-4,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=[0.1, 0.5, 0.9],  # P10, P50, P90
        loss=QuantileLoss(),
        reduce_on_plateau_patience=5,
        log_interval=50,
    )
