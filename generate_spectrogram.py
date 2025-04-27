#!/usr/bin/env python3
"""
Generate spectrograms from input audio samples.

Usage:
    python generate-spectrogram.py --input INPUT_FILE \\
        --output OUTPUT_DIR [--spectrogram-size SIZE]
"""

import os
import argparse

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.mp4'}

def generate_spectrogram(audio_file, output_dir, size):
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np

    y, sr = librosa.load(audio_file, sr=None)
    D = np.abs(librosa.stft(y))
    DB = librosa.amplitude_to_db(D, ref=np.max)

    # Create a square figure of given size (width and height in inches)
    plt.figure(figsize=(size, size))
    librosa.display.specshow(DB, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.tight_layout(pad=0)
    basename, _ = os.path.splitext(os.path.basename(audio_file))
    spec_path = os.path.join(output_dir, f"{basename}.png")
    plt.savefig(spec_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved spectrogram: {spec_path}")
    return spec_path

def main():
    parser = argparse.ArgumentParser(
        description="Split audio into segments and optionally generate spectrograms."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input audio file or directory"
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory where audio segments will be saved"
    )
    parser.add_argument(
        "--spectrogram-size", "-z", type=float, default=16,
        help="Size (in inches) for both width and height of spectrogram figures"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.input):
        for dirpath, _, filenames in os.walk(args.input):
            rel_dir = os.path.relpath(dirpath, args.input)
            if rel_dir == '.':
                out_dir = args.output_dir
            else:
                out_dir = os.path.join(args.output_dir, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            for filename in sorted(filenames):
                ext = os.path.splitext(filename)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    input_path = os.path.join(dirpath, filename)
                    generate_spectrogram(input_path, out_dir, args.spectrogram_size)
    elif os.path.isfile(args.input):
        generate_spectrogram(args.input, args.output_dir, args.spectrogram_size)
    else:
        parser.error(f"Input path '{args.input}' is not a file or directory")

if __name__ == "__main__":
    main()
