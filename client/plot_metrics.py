import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Reading metrics from {file_path}...")
    try:
        df = pd.read_json(file_path, lines=True)
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
        return

    plt.figure(figsize=(10, 6))

    # Plot training loss if available
    if 'train_mean_nll' in df.columns:
        plt.plot(df['train_mean_nll'], label='train_mean_nll')
    elif 'loss' in df.columns:
        plt.plot(df['loss'], label='loss')
    
    # Plot test/validation loss if available
    # The user mentioned 'test/nll', let's check for it or similar keys
    test_keys = [key for key in df.columns if 'test' in key or 'val' in key]
    
    if test_keys:
        for key in test_keys:
            # Drop NA to handle if test metrics are less frequent
            plt.plot(df[key].dropna(), label=key)
    else:
        print("No test/validation metrics found.")

    plt.xlabel('Steps')
    plt.ylabel('Loss / NLL')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    output_file = 'metrics_plot.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    # Attempt to show the plot (will work if run in a GUI environment)
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics from a JSONL file.')
    parser.add_argument('file_path', nargs='?', default='/tmp/tinker-examples/sl-loop/metrics.jsonl', 
                        help='Path to the metrics.jsonl file')
    
    args = parser.parse_args()
    plot_metrics(args.file_path)
