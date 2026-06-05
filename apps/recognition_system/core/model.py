import numpy as np
import torch
import torch.nn.functional as F

from backbones.iresnet import iresnet18, iresnet34, iresnet50, iresnet100

MODEL_FACTORY = {
    "iresnet18": iresnet18,
    "iresnet34": iresnet34,
    "iresnet50": iresnet50,
    "iresnet100": iresnet100,
}


def _is_cuda_available():
    """Safely check if CUDA is available and usable."""
    if not torch.cuda.is_available():
        return False
    try:
        # Try to actually use CUDA to verify it's compiled
        torch.empty(1, device='cuda')
        return True
    except RuntimeError:
        return False


def _select_state_dict(raw):
    """Pick backbone state_dict from different checkpoint formats."""
    if not isinstance(raw, dict):
        return raw

    # Common training checkpoint formats.
    for k in ("state_dict_backbone", "model", "state_dict"):
        if k in raw and isinstance(raw[k], dict):
            return raw[k]

    # model_best.pt often is already a plain state_dict.
    if any(isinstance(v, torch.Tensor) for v in raw.values()):
        return raw

    return raw


def _strip_prefixes(state_dict):
    """Remove wrappers introduced by DDP / module nesting."""
    cleaned = {}
    prefixes = ("module.", "backbone.", "net.", "model.")

    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if new_key.startswith(p):
                    new_key = new_key[len(p):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


class FaceEmbeddingModel:
    def __init__(self, weights_path: str, model_name: str, img_size: int = 112, device: str = "auto"):
        if model_name not in MODEL_FACTORY:
            supported = ", ".join(sorted(MODEL_FACTORY.keys()))
            raise ValueError(f"Unsupported model_name={model_name}. Supported: {supported}")

        if device == "auto":
            device = "cuda" if _is_cuda_available() else "cpu"
        elif device == "cuda" and not _is_cuda_available():
            print("⚠️  CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

        self.device = torch.device(device)
        self.img_size = img_size

        net = MODEL_FACTORY[model_name](pretrained=False, fp16=False, num_features=512)
        # Always load to CPU first to handle models trained on different devices
        raw = torch.load(weights_path, map_location='cpu')
        state = _select_state_dict(raw)
        state = _strip_prefixes(state)
        net.load_state_dict(state, strict=True)
        net.eval()
        net.to(self.device)
        self.net = net

    def preprocess(self, face_rgb: np.ndarray) -> torch.Tensor:
        face = face_rgb.astype(np.float32)
        face = (face - 127.5) / 127.5
        face = np.transpose(face, (2, 0, 1))
        tensor = torch.from_numpy(face).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        x = self.preprocess(face_rgb)
        feat = self.net(x)
        feat = F.normalize(feat, dim=1)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

