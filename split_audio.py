#!/usr/bin/env python3
"""
Split an input audio file into n-second samples and optionally compute FFTs for each segment.

Usage:
    python split_audio.py --input INPUT_FILE --segment-length LENGTH_IN_SECONDS \
        --output-dir OUTPUT_DIR [--fft] [--fft-length FFT_LENGTH] [--fft-dir FFT_OUTPUT_DIR]
"""

import os
import argparse
import numpy as np
from pydub import AudioSegment
from pydub.utils import which

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.mp4'}

def split_audio(input_file, segment_length, output_dir,
                compute_fft=False, fft_length=None, fft_dir=None):
    # Check for ffmpeg (required by pydub)
    if which('ffmpeg') is None and which('ffmpeg.exe') is None:
        print("Warning: ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.")
    # Determine input file extension and map to ffmpeg format if needed
    fname = os.path.basename(input_file)
    basename, ext_with_dot = os.path.splitext(fname)
    input_ext = ext_with_dot.lstrip('.').lower()
    output_ext = 'wav'
    output_fmt = 'wav'
    # Load audio and apply low-pass filter if requested
    audio = AudioSegment.from_file(input_file)
    duration_ms = len(audio)
    segment_ms = segment_length * 1000
    segments = []
    for i, start in enumerate(range(0, duration_ms, segment_ms)):
        end = min(start + segment_ms, duration_ms)
        segment = audio[start:end]
        # Exclude segment if it is less than n seconds
        if len(segment) < segment_length * 1000:
            continue
        # set output filename with appropriate extension
        out_name = f"{basename}_part{i:04d}.{output_ext}"
        out_path = os.path.join(output_dir, out_name)
        # export segment with desired format
        segment.set_frame_rate(16000).set_channels(1).export(out_path, format='WAV')
        print(f"Exported audio segment: {out_path}")
        segments.append(out_path)
        # compute FFT if requested
        if compute_fft:
            from scipy.fft import rfft

            # prepare FFT output directory and file path
            if fft_dir is None:
                raise ValueError("fft_dir must be provided when compute_fft=True")
            # ensure mono for FFT
            seg_mono = segment if segment.channels == 1 else segment.set_channels(1)
            # get sample rate and samples
            sr = seg_mono.frame_rate
            samples = np.array(seg_mono.get_array_of_samples())
            # apply Hamming window
            window = np.hamming(len(samples))
            windowed_samples = samples * window
            # compute real FFT with specified length
            fft_n = fft_length or len(windowed_samples)
            # compute FFT 
            fft_result = rfft(windowed_samples, n=fft_n)
            complex_magnitude = np.round(np.abs(fft_result), decimals=6)
            # prepare CSV output
            base = os.path.splitext(os.path.basename(out_path))[0]
            csv_name = f"{base}_fft.csv"
            csv_path = os.path.join(fft_dir, csv_name)
            # save frequency vs amplitude to CSV
            # write CSV with header
            np.savetxt(csv_path, complex_magnitude, delimiter=',', fmt='%.6f', comments='')
            print(f"Saved FFT CSV: {csv_path}")
    return segments

def main():
    parser = argparse.ArgumentParser(
        description="Split audio into segments and optionally compute FFT for each segment."
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
        "--fft", action="store_true",
        help="Compute FFT for each output segment"
    )
    parser.add_argument(
        "--fft-length", type=int, default=1023,
        help="Length of FFT (number of points)"
    )
    parser.add_argument(
        "--fft-dir", help="Directory where FFT CSV files will be saved"
    )

    args = parser.parse_args()
    # validate FFT options
    os.makedirs(args.output_dir, exist_ok=True)
    if args.fft:
        if not args.fft_dir:
            parser.error("--fft-dir is required when --fft is specified")
        if args.fft_length <= 0:
            parser.error("--fft-length must be a positive integer")
        os.makedirs(args.fft_dir, exist_ok=True)

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.input):
        for dirpath, _, filenames in os.walk(args.input):
            rel_dir = os.path.relpath(dirpath, args.input)
            # prepare output directory for audio segments
            if rel_dir == '.':
                out_dir = args.output_dir
            else:
                out_dir = os.path.join(args.output_dir, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            # prepare FFT output directory if needed (mirror structure)
            if args.fft:
                if rel_dir == '.':
                    fft_subdir = args.fft_dir
                else:
                    fft_subdir = os.path.join(args.fft_dir, rel_dir)
                os.makedirs(fft_subdir, exist_ok=True)
            else:
                fft_subdir = None
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    input_path = os.path.join(dirpath, filename)
                    print(f"Processing file: {input_path}")
                    # split audio and compute FFT if requested
                    segments = split_audio(
                        input_path,
                        args.segment_length,
                        out_dir,
                        compute_fft=args.fft,
                        fft_length=args.fft_length,
                        fft_dir=fft_subdir
                    )
    elif os.path.isfile(args.input):
        # split audio and compute FFT if requested
        # prepare FFT output directory for single file
        fft_subdir = args.fft_dir if args.fft else None
        segments = split_audio(
            args.input,
            args.segment_length,
            args.output_dir,
            compute_fft=args.fft,
            fft_length=args.fft_length,
            fft_dir=fft_subdir
        )
    else:
        parser.error(f"Input path '{args.input}' is not a file or directory")

if __name__ == "__main__":
    main()
