import glob
import torch
import gc
from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
from os.path import join, dirname, splitext, exists
from pathlib import Path
from argparse import ArgumentParser
from librosa import resample
import torch.nn.functional as F

# Set CUDA architecture list
from sgmse.sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()

from sgmse.sgmse.model import ScoreModel
from sgmse.sgmse.util.other import pad_spec

def process_chunk(model, chunk, args, pad_mode):
    """Helper to process a single audio chunk through the model."""
    T_chunk = chunk.size(1)
    Y = torch.unsqueeze(model._forward_transform(model._stft(chunk.to(args.device))), 0)
    
    # Handle reflection padding errors on very short chunks
    try:
        Y = pad_spec(Y, mode=pad_mode)
    except RuntimeError as e:
        if "Padding size should be less than the corresponding input dimension" in str(e) and pad_mode == "reflection":
            Y = pad_spec(Y, mode="zero_pad")
        else:
            raise e
    
    if model.sde.__class__.__name__ == 'OUVESDE':
        if args.sampler_type == 'pc':
            sampler = model.get_pc_sampler('reverse_diffusion', args.corrector, Y, N=args.N, 
                corrector_steps=args.corrector_steps, snr=args.snr)
        elif args.sampler_type == 'ode':
            sampler = model.get_ode_sampler(Y, N=args.N)
        else:
            raise ValueError(f"Sampler type {args.sampler_type} not supported")
    elif model.sde.__class__.__name__ == 'SBVESDE':
        sampler_type = 'ode' if args.sampler_type == 'pc' else args.sampler_type
        sampler = model.get_sb_sampler(sde=model.sde, y=Y, sampler_type=sampler_type)
    else:
        raise ValueError(f"SDE {model.sde.__class__.__name__} not supported")
        
    with torch.inference_mode(), torch.autocast(device_type=args.device, dtype=torch.float16):
        sample, _ = sampler()
    
    x_hat = model.to_audio(sample.squeeze(), T_chunk).cpu()
    
    # Cleanup memory
    del Y, sampler, sample
    gc.collect()
    torch.cuda.empty_cache()
    
    return x_hat

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="split_files/Amplify", help='Base directory containing the input data (3 levels deep)')
    parser.add_argument("--output_dir", type=str, default="split_files/Amplify", help='Base directory for the enhanced data')
    parser.add_argument("--ckpt", type=str, default="scripts/sgmse/sgmse_artificial_rir_350k_2.ckpt", help='Path to model checkpoint')
    parser.add_argument("--sampler_type", type=str, default="pc", help="Sampler type for the PC sampler.")
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
    parser.add_argument("--force", action="store_true", help="Force computation even if the enhanced file already exists")
    args = parser.parse_args()

    # Enable cuDNN benchmarking for faster dynamic shapes if available
    torch.backends.cudnn.benchmark = True

    print(f"Loading model from {args.ckpt}...")
    model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=args.device, weights_only=False)
    model.t_eps = args.t_eps
    model.eval()

    # Check if the model is trained on 48 kHz data
    if model.backbone == 'ncsnpp_48k':
        target_sr = 48000
        pad_mode = "reflection"
    elif model.backbone == 'ncsnpp_v2':
        target_sr = 16000
        pad_mode = "reflection"
    else:
        target_sr = 16000
        pad_mode = "zero_pad"

    max_length_seconds = 20
    chunk_length_seconds = 10
    overlap_seconds = 5
    
    max_length_samples = max_length_seconds * target_sr
    chunk_samples = chunk_length_seconds * target_sr
    overlap_samples = overlap_seconds * target_sr
    stride = chunk_samples - overlap_samples

    input_base = Path(args.input_dir)
    output_base = Path(args.output_dir)

    # Search 3 levels deep: InputPath/Language/Gender/SongDir/
    song_dirs = [d for d in input_base.glob('*/*') if d.is_dir()]

    if not song_dirs:
        print(f"No subdirectories found 2 levels deep in {args.input_dir}.")
        print(f"Expected structure: {args.input_dir}<Gender>/<SongDir>/")
        exit(1)

    for song_dir in song_dirs:
        # Reconstruct paths
        parts = song_dir.relative_to(input_base).parts
        if len(parts) != 2:
            continue
            
        gender, song_id = parts
        current_out_dir = output_base / gender / song_id
        print(current_out_dir)
        
        # Get audio files (flac and wav)
        noisy_files = list(song_dir.glob('*.flac')) + list(song_dir.glob('*.wav'))
        
        if not noisy_files:
            continue

        # Skip directory check
        skip_dir = True
        if not args.force:
            for f in noisy_files:
                if not f.name.endswith('-dereverberized.flac'):  # Ignore already processed files if any
                    out_name = f"{f.stem}-dereverberized.flac"
                    if not (current_out_dir / out_name).exists():
                        skip_dir = False
                        break
        else:
            skip_dir = False

        if skip_dir:
            print(f"Skipping: {current_out_dir} (All files already dereverberized)")
            continue

        print(f"Processing: {current_out_dir}")
        current_out_dir.mkdir(parents=True, exist_ok=True)

        # Process files in this directory
        for noisy_file in tqdm(noisy_files, desc=f"{song_id}"):
            if noisy_file.name.endswith('-dereverberized.flac'):
                continue
                
            out_filename = f"{noisy_file.stem}-dereverberized.flac"
            out_path = current_out_dir / out_filename

            if not args.force and out_path.exists():
                continue

            y, sr = load(str(noisy_file))
            if sr != target_sr:
                y = torch.tensor(resample(y.numpy(), orig_sr=sr, target_sr=target_sr))

            T_orig = y.size(1)   
            norm_factor = y.abs().max()
            y = y / norm_factor
            
            # Chunking logic for long files
            if T_orig <= max_length_samples:
                x_hat = process_chunk(model, y, args, pad_mode)
            else:
                window = torch.hann_window(chunk_samples).cpu()
                
                pad_len = stride - ((T_orig - chunk_samples) % stride)
                if pad_len == stride: 
                    pad_len = 0
                
                y_padded = F.pad(y, (0, pad_len))
                out_wav_padded = torch.zeros(1, y_padded.size(1))
                window_sum_padded = torch.zeros(1, y_padded.size(1))
                
                num_chunks = (y_padded.size(1) - chunk_samples) // stride + 1
                
                for i in range(num_chunks):
                    start = i * stride
                    end = start + chunk_samples
                    chunk = y_padded[:, start:end]
                    
                    x_hat_chunk = process_chunk(model, chunk, args, pad_mode)
                    
                    out_wav_padded[:, start:end] += x_hat_chunk.unsqueeze(0) * window
                    window_sum_padded[:, start:end] += window
                    
                window_sum_padded[window_sum_padded < 1e-8] = 1.0
                out_wav_padded = out_wav_padded / window_sum_padded
                x_hat = out_wav_padded[:, :T_orig].squeeze(0)

            # Renormalize and write
            x_hat = x_hat * norm_factor
            write(str(out_path), x_hat.numpy(), target_sr)
            print(str(out_path))

            del y, x_hat
            gc.collect()