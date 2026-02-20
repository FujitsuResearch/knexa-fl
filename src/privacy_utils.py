import re, torch, logging
from src.globals import PATTERNS, CLIP_NORM, GAUSS_NOISE_SIG, SIER_THRESH
 
logger = logging.getLogger(__name__)
PATTERN_RE = re.compile("|".join(PATTERNS), re.I)
 
def compute_sier(decoded_strings):
    flagged = sum(len(PATTERN_RE.findall(s)) for s in decoded_strings)
    total = sum(len(s.split()) for s in decoded_strings)
    sier = flagged / total if total else 0.0
    if sier > SIER_THRESH:
        logger.warning(f"SIER {sier} > threshold; skipping")
    return sier
 
def dp_clip_noise(t: torch.Tensor):
    # L2 norm clipping per sample
    norm = torch.norm(t, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    t_clipped = t * (CLIP_NORM / norm).clamp(max=1.0)
    
    # Add Gaussian noise calibrated to clipping bound
    noise = torch.normal(0, GAUSS_NOISE_SIG * CLIP_NORM, size=t.shape, device=t.device)
    t_noisy = t_clipped + noise
    
    # Optional: Re-clip after noise to maintain strict bound (uncomment if needed)
    # final_norm = torch.norm(t_noisy, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    # t_noisy = t_noisy * (CLIP_NORM / final_norm).clamp(max=1.0)
    
    return t_noisy