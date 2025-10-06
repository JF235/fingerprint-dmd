from dmd_utils import get_model, get_template, match, identify


class DmdExtractor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = get_model(model_path, device=device)

    def extract(self, img, mnt):
        return get_template(img, mnt, self.model, device=self.device)

class DmdMatcher:
    def __init__(self):
        pass

    def match(self, template_q, template_g):
        return match(template_q, template_g)

    def identify(self, queries, gallery, device='cpu', batch_size=256):
        return identify(queries, gallery, device, batch_size)