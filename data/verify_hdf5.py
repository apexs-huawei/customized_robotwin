"""
Verify HDF5 data by reading and plotting RGB (JPEG-encoded) + depth (PNG uint16-encoded) images.

Usage:
    python verify_hdf5.py <path_to_episode.hdf5>
    python verify_hdf5.py <path_to_episode.hdf5> --frame 5
    python verify_hdf5.py <path_to_episode.hdf5> --frame 0 --output verify.png
"""
import matplotlib
matplotlib.use("Agg")

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


def decode_rgb(buf):
    """Decode a JPEG-encoded byte string to an RGB numpy array."""
    if isinstance(buf, (bytes, bytearray)):
        raw = buf.rstrip(b'\x00')
        arr = np.frombuffer(raw, dtype=np.uint8)
    else:
        raw = buf.tobytes().rstrip(b'\x00')
        arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG RGB image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_depth(buf):
    """Decode a PNG-encoded uint16 byte string to a depth numpy array."""
    if isinstance(buf, (bytes, bytearray)):
        raw = buf.rstrip(b'\x00')
        arr = np.frombuffer(raw, dtype=np.uint8)
    else:
        raw = buf.tobytes().rstrip(b'\x00')
        arr = np.frombuffer(raw, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("Failed to decode PNG depth image")
    return depth


def main():
    parser = argparse.ArgumentParser(description="Verify RGB and depth data in an HDF5 file")
    parser.add_argument("hdf5_path", type=str, help="Path to the episode HDF5 file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize (default: 0)")
    parser.add_argument("--output", type=str, default="verify_hdf5.png", help="Output image path (default: verify_hdf5.png)")
    args = parser.parse_args()

    with h5py.File(args.hdf5_path, "r") as f:
        obs = f["observation"]
        cam_names = sorted([
            k for k in obs.keys()
            if isinstance(obs[k], h5py.Group) and ("rgb" in obs[k] or "depth" in obs[k])
        ])

        if not cam_names:
            print("No cameras with rgb/depth found in observation group.")
            print(f"Available keys: {list(obs.keys())}")
            return

        t = args.frame
        n_cams = len(cam_names)

        has_rgb = any("rgb" in obs[c] for c in cam_names)
        has_depth = any("depth" in obs[c] for c in cam_names)
        n_rows = int(has_rgb) + int(has_depth)
        if n_rows == 0:
            print("No rgb or depth data found.")
            return

        print(f"\n{'='*60}")
        print(f"HDF5: {args.hdf5_path}")
        print(f"Frame: {t}")
        print(f"Cameras: {cam_names}")
        print(f"{'='*60}")

        fig, axes = plt.subplots(n_rows, n_cams, figsize=(5 * n_cams, 5 * n_rows), squeeze=False)
        row = 0

        # RGB row
        if has_rgb:
            for col, cam in enumerate(cam_names):
                ax = axes[row, col]
                cam_group = obs[cam]
                if "rgb" in cam_group:
                    n_frames = len(cam_group["rgb"])
                    if t >= n_frames:
                        ax.set_title(f"{cam} RGB\nframe {t} out of range ({n_frames})")
                        print(f"\n[{cam}] RGB: frame {t} out of range ({n_frames} total)")
                    else:
                        rgb = decode_rgb(cam_group["rgb"][t])
                        ax.imshow(rgb)
                        ax.set_title(f"{cam} RGB\nframe {t}/{n_frames}, shape={rgb.shape}")
                        print(f"\n[{cam}] RGB:")
                        print(f"  HDF5 dataset dtype: {cam_group['rgb'].dtype}")
                        print(f"  HDF5 dataset shape: {cam_group['rgb'].shape}")
                        print(f"  Decoded shape:      {rgb.shape}")
                        print(f"  Decoded dtype:      {rgb.dtype}")
                        print(f"  Value range:        [{rgb.min()}, {rgb.max()}]")
                else:
                    ax.set_title(f"{cam} RGB — N/A")
                    print(f"\n[{cam}] RGB: N/A")
                ax.axis("off")
            row += 1

        # Depth row
        if has_depth:
            for col, cam in enumerate(cam_names):
                ax = axes[row, col]
                cam_group = obs[cam]
                if "depth" in cam_group:
                    n_frames = len(cam_group["depth"])
                    if t >= n_frames:
                        ax.set_title(f"{cam} depth\nframe {t} out of range ({n_frames})")
                        print(f"\n[{cam}] Depth: frame {t} out of range ({n_frames} total)")
                    else:
                        raw = cam_group["depth"][t]
                        if isinstance(raw, (bytes, np.bytes_)):
                            depth_raw = decode_depth(raw)
                            depth = depth_raw.astype(np.float32)
                            storage_fmt = "PNG-encoded bytes"
                            raw_dtype = depth_raw.dtype
                        else:
                            depth = np.array(raw, dtype=np.float32)
                            storage_fmt = "raw numeric array"
                            raw_dtype = cam_group["depth"].dtype

                        im = ax.imshow(depth, cmap="viridis")
                        ax.set_title(
                            f"{cam} depth\n"
                            f"frame {t}/{n_frames}, shape={depth.shape}\n"
                            f"min={depth.min():.1f} max={depth.max():.1f} mm"
                        )
                        plt.colorbar(im, ax=ax, fraction=0.046)

                        print(f"\n[{cam}] Depth:")
                        print(f"  Storage format:     {storage_fmt}")
                        print(f"  HDF5 dataset dtype: {cam_group['depth'].dtype}")
                        print(f"  HDF5 dataset shape: {cam_group['depth'].shape}")
                        print(f"  Decoded dtype:      {raw_dtype}")
                        print(f"  Decoded shape:      {depth.shape}")
                        print(f"  Value range:        [{depth.min():.2f}, {depth.max():.2f}] mm")
                else:
                    ax.set_title(f"{cam} depth — N/A")
                    print(f"\n[{cam}] Depth: N/A")
                ax.axis("off")

        plt.suptitle(f"HDF5: {args.hdf5_path}\nFrame {t}", fontsize=12)
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n{'='*60}")
        print(f"Saved verification plot to {args.output}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
