"""Shared Tiny MLP serialization, quantization and NumPy reference math."""
from pathlib import Path
import numpy as np
LABELS = ['background','cough','glass_break','baby_cry','dog_bark']
LABEL_NAMES = LABELS

def quantize_dequantize(weights: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    result = []
    for layer in weights:
        quantized_layer = {}
        for name, values in layer.items():
            maximum = max(float(np.max(np.abs(values))), 1e-10)
            scale = maximum / 127.0
            quantized = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
            quantized_layer[name] = (quantized.astype(np.float32) * scale).astype(np.float32)
        result.append(quantized_layer)
    return result

def save_model(
    path: Path,
    weights: list[dict[str, np.ndarray]],
    mean: np.ndarray,
    std: np.ndarray,
    extra: dict[str, np.ndarray],
) -> None:
    if len(weights) != 2:
        raise ValueError("deployment exporter currently expects one hidden layer")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        W1=weights[0]["W"].astype(np.float32),
        B1=weights[0]["B"].astype(np.float32),
        W2=weights[1]["W"].astype(np.float32),
        B2=weights[1]["B"].astype(np.float32),
        FEATURE_MEAN=mean.astype(np.float32),
        FEATURE_STD=std.astype(np.float32),
        LABEL_NAMES=LABEL_NAMES,
        **extra,
    )

def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)

def forward(x: np.ndarray, weights: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    activations = [x]
    preactivations: list[np.ndarray] = []
    value = x
    for layer_index, layer in enumerate(weights):
        pre = value @ layer["W"] + layer["B"]
        preactivations.append(pre)
        value = np.maximum(0.0, pre) if layer_index < len(weights) - 1 else pre
        activations.append(value)
    return softmax(value), activations, preactivations

def metrics(truth,pred):
    matrix=np.zeros((5,5),dtype=int)
    for t,p in zip(truth,pred):matrix[int(t),int(p)]+=1
    recall={k:float(matrix[i,i]/sum(matrix[i])) if sum(matrix[i]) else None for i,k in enumerate(LABELS)}
    return {'correct':int(np.trace(matrix)),'total':int(matrix.sum()),'accuracy':float(np.trace(matrix)/matrix.sum()),
            'balanced_accuracy_present_classes':float(np.mean([v for v in recall.values() if v is not None])),
            'per_class_recall':recall,'confusion_matrix':matrix.tolist()}
