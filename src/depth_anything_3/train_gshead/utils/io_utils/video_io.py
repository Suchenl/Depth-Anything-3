import av
import torch
import numpy as np
import cv2
from diffusers.utils import export_to_video

# ===========  imageio =========== #
import imageio
def write_video_imageio(video_tensor: torch.Tensor, save_path: str, fps: int = 30, quality: int = 10):
    """
    Save a video tensor (N, C, H, W) as a high-quality mp4 using imageio.
    This is an improved version that bypasses the low-quality defaults of 
    diffusers.export_to_video.

    Args:
        video_tensor (torch.Tensor): Shape (N, C, H, W), pixel range [0, 1].
                                     C must be 1 (grayscale) or 3 (RGB).
        save_path (str): Output video path (e.g., 'output.mp4').
        fps (int): Frames per second.
        quality (int): Video quality for codecs like libx264 (H.264). 
                       Range is ~0-10, where lower is better. 
                       ~5-10 is generally very high quality. 
                       Default is 8.
    """
    video_tensor = video_tensor.clamp(0, 1)
    N, C, H, W = video_tensor.shape
    assert C in [1, 3], f"Unsupported channel count {C}, must be 1 or 3."
    video_np = (video_tensor.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    if C == 1:
        video_np = np.repeat(video_np, 3, axis=-1)

    # --- Core changes are here ---
    # 'libx264' is the H.264 encoder, which is very compatible!
    # The 'quality' parameter is what controls the quality of the video!
    # 'pixelformat' ensures maximum player compatibility
    writer = imageio.get_writer(save_path, fps=fps, codec='libx264', quality=quality, pixelformat='yuv420p')
    for frame in video_np:
        writer.append_data(frame)
    writer.close()
    
# ===========  pyav =========== #
def read_video_pyav(video_path: str, get_fps: bool = False) -> torch.Tensor:
    """
    Reads an mp4 file and returns the Tensor for (N, C, H, W).
    Pixel range [0,1], float32.
    """
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        frames.append(torch.from_numpy(img).permute(2, 0, 1))
    container.close()
    video_tensor = torch.stack(frames).float() / 255.0
    if get_fps:
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate)
        return video_tensor, fps
    return video_tensor

def write_video_pyav(video_tensor: torch.Tensor, save_path: str, fps: int = 30):
    """
    Save video_tensor (N, C, H, W) to mp4 using PyAV.
    - Supports grayscale (C=1) and RGB (C=3)
    - Automatically ensures contiguous memory
    - Input range: [0, 1] (float)
    """
    # ---- Sanity check ----
    assert video_tensor.ndim == 4, f"Expected 4D tensor (N, C, H, W), got {video_tensor.shape}"
    N, C, H, W = video_tensor.shape
    assert C in [1, 3], f"Unsupported channel count: {C}, only 1 (grayscale) or 3 (RGB) allowed"
    # ---- Normalize + to CPU + contiguous ----
    video_tensor = video_tensor.clamp(0, 1).detach().cpu().contiguous()
    # ---- Create output container ----
    container = av.open(save_path, mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = W
    stream.height = H
    stream.pix_fmt = 'yuv420p'  # Required by libx264 for playback compatibility
    for i in range(N):
        frame = video_tensor[i]
        if C == 1:
            # (1, H, W) → (H, W)
            frame_np = (frame.squeeze(0).numpy() * 255).astype(np.uint8)
            frame_av = av.VideoFrame.from_ndarray(frame_np, format='gray')
        else:
            # (3, H, W) → (H, W, 3)
            frame_np = (frame.permute(1, 2, 0)
                            .contiguous()
                            .numpy() * 255).astype(np.uint8)
            frame_av = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
        for packet in stream.encode(frame_av):
            container.mux(packet)
    # ---- Flush encoder ----
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    # print(f"[✓] Saved video to {save_path} ({N} frames, {fps} FPS, size {W}x{H})")

# ===========  cv2 =========== #
def read_video_cv2(video_path: str, get_fps: bool = False) -> torch.Tensor:
    """
    Reads an mp4 file and returns the Tensor for (N, C, H, W).
    Pixel range [0,1], float32.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # (H, W, C) → (C, H, W)
        frames.append(frame_tensor)
    cap.release()
    if not frames:
        raise ValueError(f"Video {video_path} is empty or failed to read.")
    video_tensor = torch.stack(frames, dim=0).float() / 255.0  # (N, C, H, W)
    if get_fps:
        return video_tensor, fps
    return video_tensor

def write_video_cv2(video_tensor: torch.Tensor, save_path: str, fps: int = 30, codec: str = 'mp4v'):
    """
    Save the Tensor (N, C, H, W) as an mp4 file using OpenCV.
    
    Args:
        video_tensor (torch.Tensor): The video tensor with shape (N, C, H, W).
                                     Supports C=1 (grayscale) or C=3 (RGB).
                                     Input pixel range should be [0, 1].
        save_path (str): The path to save the output video file.
        fps (int): The desired frames per second for the output video.
        codec (str): The FourCC code for the desired video codec. 
                     'avc1'/'X264' (H.264) is highly recommended for compatibility.
                     'hevc' (H.265) provides better compression.
                     'mp4v' is an older, less compatible option.
    """
    assert len(codec) == 4, f"Codec must be a 4-character string (FourCC), but got: {codec}"
    video_tensor = video_tensor.clamp(0, 1)
    N, C, H, W = video_tensor.shape
    assert C in [1, 3], f"Unsupported channel count {C}, must be 1 or 3."
    fourcc = cv2.VideoWriter_fourcc(*codec)

    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H), isColor=True)
    if not out.isOpened():
        raise IOError(f"Could not open video writer for path: {save_path}")
    for i in range(N):
        frame = (video_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if C == 1:
            frame_gray = frame.squeeze(-1)
            frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()