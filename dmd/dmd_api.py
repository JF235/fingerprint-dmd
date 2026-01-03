import os
from .dmd_utils import get_model, get_template, get_templates_batch, match, identify

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class DmdExtractor:
    def __init__(self, model_path, device='cpu'):
        """
        Inicializa o extrator DMD.
        
        Args:
            model_path: Caminho absoluto para o arquivo do modelo (.pth.tar)
            device: Dispositivo para executar o modelo ('cpu' ou 'cuda')
        """
        self.device = device
        self.model = get_model(model_path, device=device)
        self._warmup()

    def _warmup(self):
        """Warm up the model with a dummy inference to initialize CUDA kernels."""
        import torch
        import numpy as np
        dummy_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        dummy_mnt = np.array([[256, 256, 0]], dtype=np.float32)
        try:
            with torch.no_grad():
                _ = self.extract(dummy_img, dummy_mnt)
        except:
            pass  # Silently fail if warmup fails

    def extract(self, img, mnt):
        """Extract template from a single image."""
        return get_template(img, mnt, self.model, device=self.device)
    
    def extract_batch(self, images, mnts, use_gpu_patches=True, max_batch_size=64):
        """
        Extract templates from multiple images in batch mode.
        
        Args:
            images: List of numpy arrays [H, W]
            mnts: List of minutiae arrays, each [N_i, 3]
            use_gpu_patches: Use GPU-accelerated patch extraction (default: True)
            max_batch_size: Maximum batch size for inference (default: 64)
            
        Returns:
            List of template dicts
        """
        return get_templates_batch(
            images, mnts, self.model, 
            device=self.device,
            use_gpu_patches=use_gpu_patches,
            max_batch_size=max_batch_size
        )

class DmdMatcher:
    def __init__(self):
        pass

    def match(self, template_q, template_g, details=False):
        return match(template_q, template_g, details=details)

    def identify(self, queries, gallery, device='cpu', batch_size=256):
        return identify(queries, gallery, device, batch_size)

def get_model_path(which="dmd++"):
    if which == "dmd++":
        path = os.path.abspath(os.path.join(_CURRENT_DIR, '../logs/DMD++/best_model.pth.tar'))
    elif which == "dmd":
        path = os.path.abspath(os.path.join(_CURRENT_DIR, '../logs/DMD/best_model.pth.tar'))

    return path