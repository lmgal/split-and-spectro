#!/usr/bin/env python3
"""
Split an input audio file into n-second samples and optionally generate spectrograms.

Usage:
    python split_audio.py --input INPUT_FILE --segment-length LENGTH_IN_SECONDS \\
        --output-dir OUTPUT_DIR [--generate-spectrogram] [--spectrogram-dir SPECTROGRAM_DIR] [--spectrogram-size SIZE]
"""

import os
import argparse
from pydub import AudioSegment
from pydub.utils import which

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.mp4'}

def split_audio(input_file, segment_length, output_dir):
    # Check for ffmpeg (required by pydub)
    if which('ffmpeg') is None and which('ffmpeg.exe') is None:
        print("Warning: ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.")
    # Determine input file extension and map to ffmpeg format if needed
    fname = os.path.basename(input_file)
    basename, ext_with_dot = os.path.splitext(fname)
    input_ext = ext_with_dot.lstrip('.').lower()
    output_ext = 'wav'
    output_fmt = 'wav'
    # Load audio with explicit format when necessary
    audio = AudioSegment.from_file(input_file)
    duration_ms = len(audio)
    segment_ms = segment_length * 1000
    segments = []
    for i, start in enumerate(range(0, duration_ms, segment_ms)):
        end = min(start + segment_ms, duration_ms)
        segment = audio[start:end]
        # set output filename with appropriate extension
        out_name = f"{basename}_part{i:04d}.{output_ext}"
        out_path = os.path.join(output_dir, out_name)
        # export segment with desired format
        export_kwargs = {"format": output_fmt}
        # optionally, specify mp3 codec if output is mp3
        if output_fmt == 'mp3':
            export_kwargs["codec"] = "libmp3lame"
        segment.export(out_path, **export_kwargs)
        print(f"Exported audio segment: {out_path}")
        segments.append(out_path)
    return segments

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
        "--segment-length", "-l", type=int, required=True,
        help="Length of each segment in seconds"
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory where audio segments will be saved"
    )
    parser.add_argument(
        "--generate-spectrogram", "-s", action="store_true",
        help="Generate spectrograms for each segment"
    )
    parser.add_argument(
        "--spectrogram-dir", "-p",
        help="Directory where spectrogram images will be saved (required if --generate-spectrogram)"
    )
    parser.add_argument(
        "--spectrogram-size", "-z", type=float, default=16,
        help="Size (in inches) for both width and height of spectrogram figures"
    )

    args = parser.parse_args()

    if args.generate_spectrogram and not args.spectrogram_dir:
        parser.error("--spectrogram-dir is required when --generate-spectrogram is set")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.generate_spectrogram:
        os.makedirs(args.spectrogram_dir, exist_ok=True)

    if os.path.isdir(args.input):
        for dirpath, _, filenames in os.walk(args.input):
            rel_dir = os.path.relpath(dirpath, args.input)
            if rel_dir == '.':
                out_dir = args.output_dir
                spec_dir = args.spectrogram_dir if args.generate_spectrogram else None
            else:
                out_dir = os.path.join(args.output_dir, rel_dir)
                spec_dir = (os.path.join(args.spectrogram_dir, rel_dir)
                            if args.generate_spectrogram else None)
            os.makedirs(out_dir, exist_ok=True)
            if args.generate_spectrogram:
                os.makedirs(spec_dir, exist_ok=True)
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    input_path = os.path.join(dirpath, filename)
                    print(f"Processing file: {input_path}")
                    segments = split_audio(input_path, args.segment_length, out_dir)
                    if args.generate_spectrogram:
                        for seg in segments:
                            generate_spectrogram(seg, spec_dir, args.spectrogram_size)
    elif os.path.isfile(args.input):
        segments = split_audio(args.input, args.segment_length, args.output_dir)
        if args.generate_spectrogram:
            for seg in segments:
                generate_spectrogram(seg, args.spectrogram_dir, args.spectrogram_size)
    else:
        parser.error(f"Input path '{args.input}' is not a file or directory")

if __name__ == "__main__":
    main()
