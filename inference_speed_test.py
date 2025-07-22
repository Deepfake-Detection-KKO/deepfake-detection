# run_inference_speed_test.py
# load libraries
import matplotlib
matplotlib.use('Agg') 
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.data import DataLoader
import pandas as pd

from deepfake_utils.models import MyModel


# Model Weight Paths 
MODELS_TO_TEST = {
    "ViT-b32-pretrained-clip": {"weights": "model_weights/experiment_44.pth", "transform_type": "ViT-CLIP", "dropout_rate": 0.0, "freeze_layers": False},
    "ConvNeXt-base-pretrained-clip": {"weights": "model_weights/experiment_49.pth", "transform_type": "ConvNeXt-CLIP", "dropout_rate": 0.0, "freeze_layers": False},
    "ResNet-50-pretrained-clip": {"weights": "model_weights/experiment_28.pth", "transform_type": "ResNet-CLIP", "dropout_rate": 0.0, "freeze_layers": False}
}

# Inference Test Parameters
NUM_CLASSES = 2
BATCH_SIZE_FOR_INFERENCE_TEST = 1
NUM_WARMUP_RUNS = 50
NUM_TIMING_RUNS = 500

# Plotting Configuration
FIGURE_WIDTH = 3.25
FIGURE_HEIGHT = FIGURE_WIDTH
DPI = 300
FONTSIZE_TITLE = 10
FONTSIZE_LABELS = 9
FONTSIZE_TICKS = 8
FONTSIZE_TEXT_ON_BARS = 7

# Inference Speed Measurement
def measure_inference_speed(model, input_tensor, device, num_warmup_runs, num_timing_runs):
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)

    print(f"  Warming up ({num_warmup_runs} runs)...")
    with torch.no_grad():
        for _ in range(num_warmup_runs):
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()

    print(f"  Timing inference ({num_timing_runs} runs)...")
    timings = []
    with torch.no_grad():
        for _ in range(num_timing_runs):
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(input_tensor)
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))
            elif device.type == 'mps':
                torch.mps.synchronize()
                start_time = time.perf_counter()
                _ = model(input_tensor)
                torch.mps.synchronize()
                end_time = time.perf_counter()
                timings.append((end_time - start_time) * 1000)
            else:
                start_time = time.perf_counter()
                _ = model(input_tensor)
                end_time = time.perf_counter()
                timings.append((end_time - start_time) * 1000)

    avg_time_ms = np.mean(timings)
    std_dev_ms = np.std(timings)
    return avg_time_ms, std_dev_ms

# Main Execution 
if __name__ == "__main__":
    OUTPUT_DIR = "inference_speed_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Running on Apple Silicon (MPS).")
    elif device.type == 'cpu':
        print("WARNING: Running on CPU. Inference speed will be significantly slower.")
    else:
        print(f"Running on {device.type}.")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print("TensorFloat32 precision set to 'high' for CUDA.")

    # Create a dummy input tensor explicitly
    # Assuming common input shape for pre-trained vision models (3 channels, 224x224 pixels)
    dummy_input_for_inference = torch.randn(BATCH_SIZE_FOR_INFERENCE_TEST, 3, 224, 224)
    print(f"Using dummy input shape for inference speed test: {dummy_input_for_inference.shape}")

    inference_results_data = {}

    print("\n--- Running Inference Speed Tests ---")
    for model_name, model_config in MODELS_TO_TEST.items():
        weights_path = model_config["weights"]
        transform_type = model_config["transform_type"] 
        dropout_rate = model_config["dropout_rate"]
        freeze_layers = model_config["freeze_layers"]

        print(f"\nMeasuring inference speed for {model_name} (Batch Size: {BATCH_SIZE_FOR_INFERENCE_TEST})...")

        current_model = MyModel(
            model_type=model_name,
            device=device,
            num_classes=NUM_CLASSES,
            freeze_layers=freeze_layers,
            dropout_rate=dropout_rate
        )

        try:
            current_model_weights = torch.load(weights_path, weights_only=True, map_location=device)
            cleaned_weights = type(current_model_weights)([
                (k.replace("_orig_mod.", ""), v) for k, v in current_model_weights.items()
            ])
            current_model.load_state_dict(cleaned_weights)
            print(f"Loaded weights for {model_name} from {weights_path}.")
        except FileNotFoundError:
            print(f"Error: Weights file '{weights_path}' not found. Skipping {model_name}.")
            continue
        except Exception as e:
            print(f"Error loading state_dict for {model_name} from {weights_path}: {e}. Skipping {model_name}.")
            continue

        current_model.to(device)
        current_model.eval()

        avg_ms, std_ms = measure_inference_speed(
            current_model,
            dummy_input_for_inference,
            device,
            NUM_WARMUP_RUNS,
            NUM_TIMING_RUNS
        )
        print(f"  Result: {avg_ms:.3f} ms ± {std_ms:.3f} ms")

        inference_results_data[model_name] = {
            'avg_inference_ms': avg_ms,
            'std_inference_ms': std_ms
        }

    #  Plotting and LaTeX Table Generation (unchanged) ---
    print("\nVisualizing Inference Times")
    model_names_for_plot = []
    avg_inference_times = []
    std_inference_times = []
    # Plotting Configuration
    FIGURE_WIDTH = 6.25
    FIGURE_HEIGHT = FIGURE_WIDTH * 1.0  # Or try 1.2 for even more vertical space
    DPI = 300
    FONTSIZE_TITLE = 10
    FONTSIZE_LABELS = 9
    FONTSIZE_TICKS = 8
    FONTSIZE_TEXT_ON_BARS = 7

    for name, data in inference_results_data.items():
        model_names_for_plot.append(name)
        avg_inference_times.append(data['avg_inference_ms'])
        std_inference_times.append(data['std_inference_ms'])

    if not model_names_for_plot:
        print("No inference time data available to plot.")
    else:
        sorted_indices = np.argsort(avg_inference_times)
        model_names_for_plot_sorted = [model_names_for_plot[i] for i in sorted_indices]
        avg_inference_times_sorted = [avg_inference_times[i] for i in sorted_indices]
        std_inference_times_sorted = [std_inference_times[i] for i in sorted_indices]

        plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)
        bars = plt.bar(model_names_for_plot_sorted, avg_inference_times_sorted,
                       yerr=std_inference_times_sorted, capsize=5, color='teal')

        plt.xlabel('Model', fontsize=FONTSIZE_LABELS)
        plt.ylabel(f'Inference Time (ms)', fontsize=FONTSIZE_LABELS)
        plt.title(f'Average Inference Time per Model (Batch Size={BATCH_SIZE_FOR_INFERENCE_TEST})', fontsize=FONTSIZE_TITLE)
        plt.xticks(rotation=45, ha='right', fontsize=FONTSIZE_TICKS)
        plt.yticks(fontsize=FONTSIZE_TICKS)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.05),
                     f'{yval:.3f}', ha='center', va='bottom', fontsize=FONTSIZE_TEXT_ON_BARS)

        plt.tight_layout()
        inference_time_plot_path = os.path.join(OUTPUT_DIR, 'inference_time_bar_chart.pdf')
        plt.savefig(inference_time_plot_path, bbox_inches='tight', dpi=DPI)
        plt.close()
        print(f"Saved Inference Time plot to: {inference_time_plot_path}")

    print("\n\n LaTeX Table with Inference Times")
    table_data = []
    for model_name, results in inference_results_data.items():
        inference_time_str = f"{results['avg_inference_ms']:.3f} $\\pm$ {results['std_inference_ms']:.3f}"
        table_data.append({
            "Model": model_name,
            "Inference Time (ms)": inference_time_str
        })

    df_table = pd.DataFrame(table_data)

    bold_col_inference_time = "Inference Time (ms)"

    def bold_best_inference_time(series):
        avg_times = []
        original_strings = []
        for s in series:
            original_strings.append(s)
            try:
                avg_times.append(float(s.split(' ')[0]))
            except ValueError:
                avg_times.append(float('inf'))
        
        valid_avg_times = [t for t in avg_times if t != float('inf')]
        if not valid_avg_times:
            return original_strings

        best_val = min(valid_avg_times)
        tolerance = 1e-6

        bolded_values = []
        for i, s in enumerate(original_strings):
            if abs(avg_times[i] - best_val) < tolerance:
                bolded_values.append(f"\\textbf{{{s}}}")
            else:
                bolded_values.append(s)
        return bolded_values

    df_table[bold_col_inference_time] = bold_best_inference_time(df_table[bold_col_inference_time])

    latex_table = df_table.to_latex(
        index=False,
        escape=False,
        column_format="l|c",
        caption=f"Average Inference Time per Model (Batch Size={BATCH_SIZE_FOR_INFERENCE_TEST})",
        label="tab:inference_time_performance"
    )
    latex_table_path = os.path.join(OUTPUT_DIR, 'inference_time_table.tex')
    with open(latex_table_path, 'w') as f:
        f.write(latex_table)

    print(latex_table)
    print(f"\nSaved LaTeX table to: {latex_table_path}")

    print("\nInference Speed Test Complete!")