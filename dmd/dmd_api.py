from .dmd_utils import get_model, get_template, match, identify


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

    def extract(self, img, mnt):
        return get_template(img, mnt, self.model, device=self.device)

class DmdMatcher:
    def __init__(self):
        pass

    def match(self, template_q, template_g, details=False):
        return match(template_q, template_g, details=details)

    def identify(self, queries, gallery, device='cpu', batch_size=256):
        return identify(queries, gallery, device, batch_size)