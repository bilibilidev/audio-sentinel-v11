"""C array formatting for Tiny MLP INT8 weight storage and float scales."""
from __future__ import annotations
import numpy as np

def fmt_float(v: float) -> str:
    return f"{float(v):.9e}f"

def c_float_array(name: str, arr: np.ndarray, dims: str = "") -> str:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        rows = ["    { " + ", ".join(fmt_float(v) for v in row) + " }" for row in arr]
        return f"static const float {name}{dims} = {{\n" + ",\n".join(rows) + "\n};\n"
    values = ", ".join(fmt_float(v) for v in arr.ravel())
    return f"static const float {name}{dims} = {{ {values} }};\n"

def c_int8_array(name: str, arr: np.ndarray, dims: str = "") -> str:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        rows = ["    { " + ", ".join(f"{int(v):d}" for v in row) + " }" for row in arr]
        return f"static const int8_t {name}{dims} = {{\n" + ",\n".join(rows) + "\n};\n"
    values = ", ".join(f"{int(v):d}" for v in arr.ravel())
    return f"static const int8_t {name}{dims} = {{ {values} }};\n"

def c_scalar(name: str, value: float) -> str:
    return f"static const float {name} = {{ {float(value):.9e}f }};\n"

def quantize_symmetric(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor int8 quantization: float = int8 * scale."""
    maximum = max(float(np.max(np.abs(values))), 1e-10)
    scale = maximum / 127.0
    quantized = np.clip(np.round(values / scale), -127.0, 127.0).astype(np.int8)
    return quantized, scale

def label_block(names: list[str], unused_attribute: bool) -> str:
    labels = ", ".join(f'"{n}"' for n in names)
    attribute = " __attribute__((unused))" if unused_attribute else ""
    return f"static const char *LABEL_NAMES[NUM_CLASSES]{attribute} = {{ {labels} }};\n"

def int8_header(
    model: np.lib.npyio.NpzFile,
    names: list[str],
    guard: str,
    unused_attribute: bool = True,
) -> str:
    feature_dim = int(model["W1"].shape[0])
    hidden_dim = int(model["W1"].shape[1])
    num_classes = len(names)
    text = (
        f"#ifndef {guard}\n"
        f"#define {guard}\n"
        "\n"
        "#include <stdint.h>\n"
        "\n"
        f"#define FEATURE_DIM {feature_dim}\n"
        f"#define HIDDEN_DIM {hidden_dim}\n"
        f"#define NUM_CLASSES {num_classes}\n"
        "\n"
    )
    for name, dims in (
        ("W1", "[FEATURE_DIM][HIDDEN_DIM]"),
        ("B1", "[HIDDEN_DIM]"),
        ("W2", "[HIDDEN_DIM][NUM_CLASSES]"),
        ("B2", "[NUM_CLASSES]"),
    ):
        quantized, scale = quantize_symmetric(model[name])
        text += c_int8_array(f"{name}_INT8", quantized, dims)
        text += c_scalar(f"{name}_SCALE", scale)
    text += c_float_array("FEATURE_MEAN", model["FEATURE_MEAN"], "[FEATURE_DIM]")
    text += c_float_array("FEATURE_STD", model["FEATURE_STD"], "[FEATURE_DIM]")
    text += label_block(names, unused_attribute=unused_attribute)
    text += "\n#endif\n"
    return text
