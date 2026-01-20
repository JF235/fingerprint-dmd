import pickle as pkl
import os
import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from .dmd_model import DMD
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_linear_assignment import batch_linear_assignment

DATASET = {}
LOADED_DATASET = ''
FNAMES = []

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.abspath(os.path.join(_CURRENT_DIR, '../datasets/test_data'))
MODEL_PATH = os.path.abspath(os.path.join(_CURRENT_DIR, '../logs/DMD++/best_model.pth.tar'))

def load_dmd_dataset(dataset_path):
    global DATASET, FNAMES, LOADED_DATASET
    if dataset_path == LOADED_DATASET:
        return
    
    with open(dataset_path, 'rb') as f:
        dict_mnt = pkl.load(f)
    LOADED_DATASET = dataset_path

    # 1. Filter based on file paths: {file_path1: [mnt1, mnt2, ...], file_path2: [...], ...}
    file_path_dict = {}
    for item in dict_mnt:
        file_path = item['img']
        if file_path not in file_path_dict:
            file_path_dict[file_path] = []
        file_path_dict[file_path].append(item['pose_2d'])
    
    
    # 2. Sort based on filename
    DATASET = dict(sorted(file_path_dict.items(), key=lambda x: osp.basename(x[0])))

    # 3. Filenames
    FNAMES = list(DATASET.keys())

def load_dmd_format(file_path, item_id):
    if file_path != LOADED_DATASET:
        load_dmd_dataset(file_path)
    
    img_path = osp.join(IMG_PATH, FNAMES[item_id])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mnt = DATASET[FNAMES[item_id]]

    return img, mnt

def plot_mnt(img, mnt, text=False):
    plt.imshow(img, cmap='gray')
    for m in mnt:
        x, y, a = m[0], m[1], m[2]
        plt.plot(x, y, 'ro', markerfacecolor='none')
        a = np.deg2rad(a)
        dx = 10 * np.cos(a)
        dy = 10 * np.sin(a)
        plt.plot([x, x+dx], [y, y+dy], 'r-')

        if text:
            plt.text(x + 10, y - 10, f'{np.rad2deg(a):.0f}', color='yellow', fontsize=5)

    plt.axis('off')

def extract_patches(img, mnt, patch_size=(128,128), img_ppi=500):
    tar_shape = np.array(patch_size)
    middle_shape = tar_shape.copy()

    # Helper function to extract a patch around a minutia point using affine transformation
    def _warp_affine_to_patch(src_img, pose_2d, tar_shape, middle_shape, img_ppi):
        # Compute the center of the target patch (height, width order)
        center = tar_shape[::-1] / 2.0

        # Scale factor to normalize image resolution to 500 PPI
        scale = img_ppi * 1.0 / 500 * float(tar_shape[0]) / float(middle_shape[0])

        # Extract minutia coordinates and angle
        # Get affine rotation matrix centered at the minutia point
        # Shift the patch so the minutia is at the center
        px, py, ang = float(pose_2d[0]), float(pose_2d[1]), float(pose_2d[2])
        M = cv2.getRotationMatrix2D((px, py), ang, scale)
        M[:, 2] += (center - np.array([px, py]))

        # Warp the image to extract the patch, fill borders with gray (127.5)
        patch = cv2.warpAffine(src_img, M, (int(tar_shape[1]), int(tar_shape[0])), flags=cv2.INTER_LINEAR, borderValue=127.5)
        patch = (patch - 127.5) / 127.5
        return patch.astype(np.float32)[None]

    # handle empty minutiae
    if mnt is None or len(mnt) == 0:
        return np.zeros((0, 1, int(tar_shape[0]), int(tar_shape[1])), dtype=np.float32)

    patches = []
    for pose in mnt:
        p = _warp_affine_to_patch(img, pose, tar_shape, middle_shape, img_ppi)
        patches.append(p)

    if len(patches) == 0:
        return np.zeros((0, 1, int(tar_shape[0]), int(tar_shape[1])), dtype=np.float32)
    patches = np.concatenate(patches, axis=0)
    return patches


def extract_patches_batch_gpu(images, mnts, patch_size=(128, 128), img_ppi=500, device='cuda'):
    """
    GPU-accelerated batch patch extraction using PyTorch.
    
    Args:
        images: List of numpy arrays or single batch tensor [B, H, W]
        mnts: List of minutiae arrays, each [N_i, 3] (x, y, angle)
        patch_size: Tuple (height, width) for output patches
        img_ppi: Image PPI for normalization
        device: 'cuda' or 'cpu'
        
    Returns:
        patches: Tensor [Total_N, 1, H, W] - all patches concatenated
        patch_counts: List of patch counts per image
    """
    tar_shape = np.array(patch_size)
    middle_shape = tar_shape.copy()
    scale = img_ppi * 1.0 / 500 * float(tar_shape[0]) / float(middle_shape[0])
    
    # Convert images to tensor if needed
    if isinstance(images, list):
        images = [torch.from_numpy(img).float() if isinstance(img, np.ndarray) else img for img in images]
    
    all_patches = []
    patch_counts = []
    
    device = torch.device(device)
    
    for img, mnt in zip(images, mnts):
        if mnt is None or len(mnt) == 0:
            patch_counts.append(0)
            continue
            
        # Move image to GPU
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
        img = img.to(device)
        
        # Normalize image
        img = (img - 127.5) / 127.5
        
        # Convert minutiae to tensor
        if isinstance(mnt, np.ndarray):
            mnt = torch.from_numpy(mnt).float()
        mnt = mnt.to(device)
        
        N = len(mnt)
        patch_counts.append(N)
        
        # Create affine transformation matrices for all minutiae
        # [N, 2, 3] transformation matrices
        angles_rad = torch.deg2rad(mnt[:, 2])
        cos_a = torch.cos(angles_rad)
        sin_a = torch.sin(angles_rad)
        
        # Center of output patch
        center = torch.tensor([tar_shape[1] / 2.0, tar_shape[0] / 2.0], device=device)
        
        # Build affine matrices: [cos -sin tx; sin cos ty]
        theta = torch.zeros(N, 2, 3, device=device)
        theta[:, 0, 0] = cos_a * scale
        theta[:, 0, 1] = sin_a * scale
        theta[:, 1, 0] = -sin_a * scale
        theta[:, 1, 1] = cos_a * scale
        
        # Translation to center the minutiae
        theta[:, 0, 2] = (center[0] - (mnt[:, 0] * cos_a * scale + mnt[:, 1] * sin_a * scale))
        theta[:, 1, 2] = (center[1] - (-mnt[:, 0] * sin_a * scale + mnt[:, 1] * cos_a * scale))
        
        # Normalize theta for grid_sample (expects [-1, 1] range)
        theta[:, 0, 2] = 2.0 * theta[:, 0, 2] / tar_shape[1]
        theta[:, 1, 2] = 2.0 * theta[:, 1, 2] / tar_shape[0]
        theta[:, 0, :2] = theta[:, 0, :2] * 2.0 / tar_shape[1]
        theta[:, 1, :2] = theta[:, 1, :2] * 2.0 / tar_shape[0]
        
        # Create grid and sample patches
        grid = F.affine_grid(theta, [N, 1, tar_shape[0], tar_shape[1]], align_corners=False)
        
        # Expand image for batch sampling
        img_batch = img.unsqueeze(0).unsqueeze(0).expand(N, 1, -1, -1)
        
        # Sample all patches at once
        patches = F.grid_sample(img_batch, grid, mode='bilinear', 
                               padding_mode='border', align_corners=False)
        
        all_patches.append(patches)
    
    if len(all_patches) == 0:
        return torch.zeros((0, 1, tar_shape[0], tar_shape[1]), device=device), patch_counts
    
    # Concatenate all patches
    all_patches = torch.cat(all_patches, dim=0)
    
    return all_patches, patch_counts


def get_model(model_path, device = 'cpu'):

    ndim_feat = 6
    tar_shape = (128, 128)
    pos_embed = True
    input_norm = True

    model = DMD(
        ndim_feat=ndim_feat, pos_embed=pos_embed, tar_shape=tar_shape, input_norm=input_norm,
    )

    ckp = torch.load(model_path, map_location=device)
    if 'model' in ckp.keys():
        ckp = ckp['model']
    model.load_state_dict(ckp)
    model.to(device)

    return model

def get_embeddings(model: DMD, patches, device='cpu'):
    model.eval()
    with torch.no_grad():
        patches_tensor = torch.from_numpy(patches).to(device)
        if patches_tensor.dim() == 3:
            patches_tensor = patches_tensor.unsqueeze(1)
        elif patches_tensor.dim() == 5:
            patches_tensor = patches_tensor.squeeze(2)

        if patches_tensor.dim() != 4:
            raise RuntimeError(f"get_embeddings: unexpected patches tensor shape {patches_tensor.shape}")

        embeddings = model.get_embedding(patches_tensor) # (N, ndim_feat, 16, 16)
    return embeddings


def get_embeddings_batch(model: DMD, patches_tensor, device='cuda', max_batch_size=64):
    """
    Process patches in batches to maximize GPU utilization.
    
    Args:
        model: DMD model
        patches_tensor: Tensor [N, 1, H, W] or numpy array
        device: Device to use
        max_batch_size: Maximum batch size for inference
        
    Returns:
        embeddings: Dict with 'feature' and 'mask' tensors
    """
    model.eval()
    
    # Convert to tensor if needed
    if isinstance(patches_tensor, np.ndarray):
        patches_tensor = torch.from_numpy(patches_tensor)
    
    # Ensure correct dimensions
    if patches_tensor.dim() == 3:
        patches_tensor = patches_tensor.unsqueeze(1)
    elif patches_tensor.dim() == 5:
        patches_tensor = patches_tensor.squeeze(2)
    
    if patches_tensor.dim() != 4:
        raise RuntimeError(f"get_embeddings_batch: unexpected shape {patches_tensor.shape}")
    
    N = patches_tensor.shape[0]
    
    if N == 0:
        # Return empty embeddings
        return {
            'feature': torch.zeros((1, 0, model.ndim_feat*2, 16, 16), device=device),
            'mask': torch.zeros((1, 0, 16, 16), device=device)
        }
    
    all_features = []
    all_masks = []
    
    with torch.no_grad():
        for i in range(0, N, max_batch_size):
            batch = patches_tensor[i:i+max_batch_size].to(device)
            embeddings = model.get_embedding(batch)
            
            all_features.append(embeddings['feature'].cpu())
            all_masks.append(embeddings['mask'].cpu())
    
    # Concatenate results
    result = {
        'feature': torch.cat(all_features, dim=1),  # [1, N, C, H, W]
        'mask': torch.cat(all_masks, dim=1)  # [1, N, H, W]
    }
    
    return result


def get_template(img, mnt, model, device='cpu'):
    patches = extract_patches(img, mnt)
    embeddings = get_embeddings(model, patches, device=device)
    mnt = np.array(mnt)
    mnt = torch.from_numpy(mnt).unsqueeze(0).float()
    embeddings['mnt'] = mnt.to(device)
    return embeddings


def get_templates_batch(images, mnts, model, device='cuda', use_gpu_patches=True, max_batch_size=64):
    """
    Extract templates from multiple images in batch mode for maximum GPU utilization.
    
    Args:
        images: List of numpy arrays [H, W]
        mnts: List of minutiae arrays, each [N_i, 3]
        model: DMD model
        device: Device to use
        use_gpu_patches: Use GPU-accelerated patch extraction
        max_batch_size: Maximum batch size for model inference
        
    Returns:
        templates: List of template dicts, each with 'feature', 'mask', 'mnt'
    """
    if len(images) == 0:
        return []
    
    device = torch.device(device)
    model.to(device)
    model.eval()
    
    # Step 1: Extract all patches (GPU-accelerated if enabled)
    if use_gpu_patches and device.type == 'cuda':
        all_patches, patch_counts = extract_patches_batch_gpu(images, mnts, device=device)
    else:
        # Fallback to CPU extraction
        all_patches_list = []
        patch_counts = []
        for img, mnt in zip(images, mnts):
            patches = extract_patches(img, mnt)
            all_patches_list.append(patches)
            patch_counts.append(len(patches))
        
        if sum(patch_counts) == 0:
            return [{'feature': torch.zeros((1, 0, model.ndim_feat*2, 16, 16)),
                     'mask': torch.zeros((1, 0, 16, 16)),
                     'mnt': torch.zeros((1, 0, 3))} for _ in range(len(images))]
        
        all_patches = np.concatenate(all_patches_list, axis=0)
        all_patches = torch.from_numpy(all_patches).to(device)
    
    # Step 2: Batch inference on all patches at once
    if all_patches.shape[0] == 0:
        return [{'feature': torch.zeros((1, 0, model.ndim_feat*2, 16, 16)),
                 'mask': torch.zeros((1, 0, 16, 16)),
                 'mnt': torch.zeros((1, 0, 3))} for _ in range(len(images))]
    
    embeddings = get_embeddings_batch(model, all_patches, device=device, max_batch_size=max_batch_size)
    
    # Step 3: Split embeddings back to individual templates
    templates = []
    start_idx = 0
    
    for i, (mnt, count) in enumerate(zip(mnts, patch_counts)):
        if count == 0:
            template = {
                'feature': torch.zeros((1, 0, model.ndim_feat*2, 16, 16), device=device),
                'mask': torch.zeros((1, 0, 16, 16), device=device),
                'mnt': torch.zeros((1, 0, 3), device=device)
            }
        else:
            end_idx = start_idx + count
            
            template = {
                'feature': embeddings['feature'][:, start_idx:end_idx].to(device),
                'mask': embeddings['mask'][:, start_idx:end_idx].to(device),
                'mnt': torch.from_numpy(np.array(mnt)).unsqueeze(0).float().to(device)
            }
            
            start_idx = end_idx
        
        templates.append(template)
    
    return templates


def calculate_score_torchB(feat1, feat2, mask1, mask2, ndim_feat=6, N_mean=1327, Normalize=False, binary=False, f2f_type=(2, 1)):
    '''
    The function to calculate the score between two images or two set images
    '''
    THRESHS = {0: 0.2, 1: 0.002, 2: 0.5} # 0 for plain, 1 for rolled, 2 for latent

    feat1_dense = feat1
    feat1_mask = mask1.repeat(1, 1, ndim_feat)

    feat2_dense = feat2
    feat2_mask = mask2.repeat(1, 1, ndim_feat) 

    if binary:
        feat1_dense = (feat1_dense > 0).float()
        feat2_dense = (feat2_dense > 0).float()
        feat1_mask = (feat1_mask > THRESHS[f2f_type[0]]).float()
        feat2_mask = (feat2_mask > THRESHS[f2f_type[1]]).float()
        n12 = torch.bmm(feat1_mask, feat2_mask.transpose(1, 2))
        d12 = (
            n12
            - torch.bmm((feat1_mask * feat1_dense), (feat2_mask * feat2_dense).transpose(1, 2))
            - torch.bmm((feat1_mask * (1 - feat1_dense)), (feat2_mask * (1 - feat2_dense)).transpose(1, 2))
        )
        score = 1 - 2 * torch.where(n12 > 0, d12 / n12.clamp(min=1e-3), torch.tensor(0.5, dtype=torch.float32))
    else:
        x1 = torch.sqrt(torch.bmm(feat1_mask * feat1_dense**2, feat2_mask.transpose(1, 2)))
        x2 = torch.sqrt(torch.bmm(feat1_mask, (feat2_dense**2 * feat2_mask).transpose(1, 2)))
        x12 = torch.bmm(feat1_mask * feat1_dense, (feat2_mask * feat2_dense).transpose(1, 2))

        score = x12 / (x1 * x2).clamp(min=1e-3)

        n12 = torch.bmm(feat1_mask, feat2_mask.transpose(1, 2))
    
    if Normalize:
        score = score * torch.sqrt(n12 / N_mean)

    return score

def lsa_score_torchB(S, min_pair=4, max_pair=12, mu_p=20, tau_p=0.4):
    def sigmoid(z, mu_p, tau_p):
        return 1 / (1 + torch.exp(-tau_p * torch.clamp(z - mu_p, min=-1e10, max=100)))
    n1 = S.shape[1] # for the batch, it has been consistent
    n2 = S.shape[2] 
    B = S.shape[0]
    S2 = S
    max_n = max(n1, n2)
    new_S = torch.nn.functional.pad(1 - S2, (0, max_n - n2, 0, max_n - n1, 0 , 0), value=2)
    # replace all the torch.nan element with 2
    new_S = torch.where(torch.isnan(new_S), torch.tensor(2.0).to(new_S.device), new_S)
    batch_set_pairs = batch_linear_assignment(new_S)
    org_pair = torch.arange(new_S.shape[1])[None,...].repeat(B,1).to(new_S.device)
    pairs = torch.stack((org_pair,batch_set_pairs),dim=-1)

    pairs = pairs[:,:n1, :] # B, n1, 2
    # select the [B, n1] scores according to the pairs indexing
    scores = torch.gather(S, 2, pairs[:,:,1].unsqueeze(-1).repeat(1,1,1)).squeeze(-1)
    scores = torch.where(torch.isnan(scores), torch.tensor(0.0).to(scores.device), scores)
    scores = torch.sort(scores, dim=-1, descending=True)[0]
    n1_batch = torch.sum(~torch.isnan(S[:,:,0]), dim=-1)
    n2_batch = torch.sum(~torch.isnan(S[:,0,:]), dim=-1)
    min_number = torch.min(n1_batch, n2_batch)
    n_pair = min_pair + torch.round(sigmoid(min_number, mu_p, tau_p) * (max_pair - min_pair)).int()
    k_indices = n_pair.unsqueeze(1)
    C = scores.shape[-1]
    mask = torch.arange(C).to(k_indices.device).expand(B, C) < k_indices
    score_select = scores * mask
    score = torch.sum(score_select, dim=-1) / n_pair

    return score

def lsar_score_torchB(S, mnt1, mnt2, min_pair=4, max_pair=12, mu_p=20, tau_p=0.4):
    # S in shape (B, N1, N2), mnt1 in shape (B, N1, 3), mnt2 in shape (B, N2, 3), and not all the score or mnts are valid,
    # it has the placeholder 0 for ensuring the same size for parallel computing
    def sigmoid(z, mu_p, tau_p):
        return 1 / (1 + torch.exp(-tau_p * torch.clamp(z - mu_p, min=-1e10, max=100)))
    
    def distance_theta(theta, theta2=None):
        theta2 = theta if theta2 is None else theta2
        d = (theta[:, :, None] - theta2[:, None] + 180) % 360 - 180
        return d
    
    def distance_R(mnts):
        d = torch.rad2deg(torch.atan2(mnts[:, :, 1, None] - mnts[:, None, :, 1], mnts[:,None, :, 0] - mnts[:, :, 0, None]))
        d = (mnts[:, :, 2, None] + d + 180) % 360 - 180
        return d
    
    def distance_mnts(mnts):
        d = torch.sqrt((mnts[:, :, 0, None] - mnts[:, None, :, 0])**2 + (mnts[:, :, 1, None] - mnts[:, None, :, 1])**2)
        return d
    
    def relax_labeling(mnts1, mnts2, scores, min_number, n_pair_sig): 
        # Parâmetros calibrados conforme Tabela 2 do paper [cite: 429]
        mu_1, tau_1 = 10, -8.0 / 5      # Distância espacial (pixels)
        mu_2, tau_2 = torch.pi/12, -30 # Diferença direcional
        mu_3, tau_3 = torch.pi/12, -30 # Ângulo radial
        w_R = 0.5                      # Peso da relaxação [cite: 352]
        n_rel = 5                      # Iterações [cite: 429]

        # 1. Cálculo das matrizes de compatibilidade geométrica [cite: 354, 356]
        D1 = torch.abs(distance_mnts(mnts1) - distance_mnts(mnts2))
        D2 = torch.deg2rad(torch.abs((distance_theta(mnts1[:, :, 2]) - distance_theta(mnts2[:,:, 2]) + 180) % 360 - 180))
        D3 = torch.deg2rad(torch.abs((distance_R(mnts1[:, :, :3]) - distance_R(mnts2[:, :, :3]) + 180) % 360 - 180))

        # Compatibilidade rho(t,k) [cite: 353]
        rp = (sigmoid(D1, mu_1, tau_1) * sigmoid(D2, mu_2, tau_2) * sigmoid(D3, mu_3, tau_3))
        
        B, N, _ = rp.shape
        indices = torch.arange(N)
        rp[:, indices, indices] = 0 # k != t 
        rp = torch.where(torch.isnan(rp), torch.tensor(0.0).to(rp.device), rp)
        
        lambda_t = torch.where(torch.isnan(scores), torch.tensor(0.0).to(scores.device), scores)
        scores_init = lambda_t.clone()

        # 2. Processo de Relaxação Iterativo [cite: 348, 362]
        for _ in range(n_rel): 
            # Suporte dos vizinhos ponderado pela confiança atual
            suporte = torch.sum(rp * lambda_t[:, None, :], axis=-1) / (min_number[:, None] - 1)
            lambda_t = w_R * lambda_t + (1 - w_R) * suporte

        # 3. Cálculo da Eficiência (A chave para eliminar ruído) [cite: 395]
        # Eficiência = Confiança Final / Confiança Inicial
        efficiency = lambda_t / torch.clamp(scores_init, min=1e-6)
        efficiency = torch.where(torch.isnan(efficiency), torch.tensor(0.0).to(efficiency.device), efficiency)

        # 4. Geração da lista de candidatos baseada em ranking de eficiência [cite: 398, 399]
        # Em vez de ordenar por score, ordenamos por eficiência para garantir geometria
        _, sorted_indices = torch.sort(efficiency, dim=1, descending=True)
        lambda_t_sorted = torch.gather(lambda_t, 1, sorted_indices)
        efficiency_sorted = torch.gather(efficiency, 1, sorted_indices)

        # 5. Seleção Dinâmica (Filtro de Qualidade)
        # Definimos que um par só contribui se a eficiência for alta (ex: > 0.7)
        # e limitamos ao n_pair da sigmóide para manter estabilidade estatística
        threshold_eff = (1.01)*(0.5 ** n_rel) 
        mask_qualidade = (efficiency_sorted > threshold_eff)
        # print(efficiency_sorted)
        
        # Criamos a máscara para o top N definido pela sigmóide [cite: 339]
        C = efficiency.shape[1]
        k_indices = n_pair_sig.unsqueeze(1) 
        mask_top_n = torch.arange(C).to(k_indices.device).expand(B, C) < k_indices
        
        # A máscara final combina o limite de quantidade com o filtro de qualidade
        final_mask = mask_top_n & mask_qualidade
        
        # Contagem real de pares que contribuíram (pode ser menor que n_pair_sig)
        n_pair_real = torch.clamp(final_mask.sum(dim=-1), min=1)
        
        # Score final: média apenas dos pares coerentes
        score = torch.sum(lambda_t_sorted * final_mask, dim=-1)
        
        return score, lambda_t, sorted_indices, n_pair_real.int()
    
    n1 = S.shape[1]
    n2 = S.shape[2]
    B = S.shape[0]

    # Handle empty templates (0 features): return zero scores
    if n1 == 0 or n2 == 0:
        device = S.device
        final_score = torch.zeros(B, device=device)
        pairs = torch.zeros((B, 0, 2), dtype=torch.long, device=device)
        scores = torch.zeros((B, 0), device=device)
        relaxed_scores = torch.zeros((B, 0), device=device)
        sorted_indices = torch.zeros((B, 0), dtype=torch.long, device=device)
        n_pair = torch.zeros(B, dtype=torch.long, device=device)
        return final_score, pairs, scores, relaxed_scores, sorted_indices, n_pair

    assert n1 == mnt1.shape[1] and n2 == mnt2.shape[1]
    S2 = S
    max_n = max(n1, n2)
    new_S = torch.nn.functional.pad(1 - S2, (0, max_n - n2, 0, max_n - n1, 0 , 0), value=2)
    new_S = torch.where(torch.isnan(new_S), torch.tensor(2.0).to(new_S.device), new_S)

    if n1 < n2:
        batch_set_pairs = batch_linear_assignment(new_S)
        org_pair = torch.arange(new_S.shape[1])[None,...].repeat(B,1).to(new_S.device)
        pairs = torch.stack((org_pair,batch_set_pairs),dim=-1)
        pairs = pairs[:,:n1, :]
        scores = torch.gather(S, 2, pairs[:,:,1].unsqueeze(-1).repeat(1,1,1)).squeeze(-1)
    else:
        batch_set_pairs = batch_linear_assignment(new_S.transpose(1,2)) 
        org_pair = torch.arange(new_S.shape[2])[None,...].repeat(B,1).to(new_S.device) 
        pairs = torch.stack((batch_set_pairs, org_pair), dim=-1) 
        pairs = pairs[:,:n2, :]
        scores = torch.gather(S, 1, pairs[:,:,0].unsqueeze(-2).repeat(1,1,1)).squeeze(-2)

    n1_batch = torch.sum(~torch.isnan(S[:,:,0]), dim=-1)
    n2_batch = torch.sum(~torch.isnan(S[:,0,:]), dim=-1)
    min_number = torch.min(n1_batch, n2_batch)
    n_pair = min_pair + torch.round(sigmoid(min_number, mu_p, tau_p) * (max_pair - min_pair)).int()
    mnt1_order = torch.gather(mnt1, 1, pairs[:,:,0].unsqueeze(-1).repeat(1,1,3))
    mnt2_order = torch.gather(mnt2, 1, pairs[:,:,1].unsqueeze(-1).repeat(1,1,3)) 
    final_score, relaxed_scores, sorted_indices, n_pair = relax_labeling(mnt1_order, mnt2_order, scores, min_number, n_pair) 

    return final_score, pairs, scores, relaxed_scores, sorted_indices, n_pair

def match(q_tpl, g_tpl, details=False):
    search_feat = q_tpl['feature']
    gallery_feat = g_tpl['feature']
    search_mask = q_tpl['mask']
    gallery_mask = g_tpl['mask']

    ndim_feat = 6
    relax = True
    binary = False
    normalize = True
    scores = calculate_score_torchB(search_feat, gallery_feat, search_mask, gallery_mask, ndim_feat=ndim_feat*2,  Normalize=normalize, N_mean=5, binary=binary, f2f_type=(2,1))
    if relax:
        final_score, pairs, scores, relaxed_scores, sorted_indices, n_pair = lsar_score_torchB(scores, q_tpl['mnt'], g_tpl['mnt'])
    else:
        score = scores

    if details:
        return {"score": final_score.cpu().numpy(),
                "pairs": pairs.cpu().numpy(),
                "scores": scores.cpu().numpy(),
                "relaxed_scores": relaxed_scores.cpu().numpy(),
                "sorted_indices": sorted_indices.cpu().numpy(),
                "n_pair": n_pair.cpu().numpy()}
    else:
        return final_score.cpu().numpy()

def match_with_details(q_tpl, g_tpl):
    search_feat = q_tpl['feature']
    gallery_feat = g_tpl['feature']
    search_mask = q_tpl['mask']
    gallery_mask = g_tpl['mask']

    ndim_feat = 6
    relax = True
    binary = False
    normalize = True
    scores = calculate_score_torchB(search_feat, gallery_feat, search_mask, gallery_mask, ndim_feat=ndim_feat*2,  Normalize=normalize, N_mean=5, binary=binary, f2f_type=(2,1))
    if relax:
        score, pairs, scores, relaxed_scores, sorted_indices, n_pair = lsar_score_torchB(scores, q_tpl['mnt'], g_tpl['mnt'])
    else:
        score = scores

    _to_cpu = lambda x: x.cpu().numpy() if torch.is_tensor(x) else x
    outputs = map(_to_cpu, (score, pairs, scores, relaxed_scores, sorted_indices, n_pair))
    return tuple(outputs)

# Classe de Dataset para criar pares de busca/galeria
class MatchDataset(Dataset):
    def __init__(self, queries, gallery):
        self.queries = queries
        self.gallery = gallery
        self.query_len = len(queries)
        self.gallery_len = len(gallery)

    def __len__(self):
        return self.query_len * self.gallery_len

    def __getitem__(self, index):
        query_idx = index // self.gallery_len
        gallery_idx = index % self.gallery_len
        
        query_template = self.queries[query_idx]
        gallery_template = self.gallery[gallery_idx]

        return {
            "search_desc": query_template['feature'].squeeze(0),
            "gallery_desc": gallery_template['feature'].squeeze(0),
            "search_mask": query_template['mask'].squeeze(0),
            "gallery_mask": gallery_template['mask'].squeeze(0),
            "search_mnt": query_template['mnt'].squeeze(0),
            "gallery_mnt": gallery_template['mnt'].squeeze(0),
            "index_pair": torch.tensor([query_idx, gallery_idx])
        }

# Função para padronizar o tamanho dos tensores em um lote (essencial para batching)
def pad_collate_fn(batch):
    def pad_to_max_N(tensor_list):
        # Verifica consistência de dimensões. Se houver mistura de 1D e 2D, promove 1D para (1, D)
        ndims = {t.ndim for t in tensor_list}
        if 1 in ndims and 2 in ndims:
             tensor_list = [t.unsqueeze(0) if t.ndim == 1 else t for t in tensor_list]
        
        max_N = max(t.shape[0] for t in tensor_list)
        padded_list = []
        for tensor in tensor_list:
            current_N = tensor.shape[0]
            if current_N < max_N:
                # Padding: (last dim, second to last dim, ...)
                padding_size = (0, 0) * (len(tensor.shape) - 1) + (0, max_N - current_N)
                padded_tensor = torch.nn.functional.pad(tensor, padding_size, value=float('nan'))
            else:
                padded_tensor = tensor
            padded_list.append(padded_tensor)
        return torch.stack(padded_list, dim=0)

    # Coleta todos os tensores do lote
    search_desc = [item["search_desc"] for item in batch]
    gallery_desc = [item["gallery_desc"] for item in batch]
    search_mask = [item["search_mask"] for item in batch]
    gallery_mask = [item["gallery_mask"] for item in batch]
    search_mnt = [item["search_mnt"] for item in batch]
    gallery_mnt = [item["gallery_mnt"] for item in batch]
    index_pairs = [item["index_pair"] for item in batch]

    # Aplica o padding
    batch_dict = {
        "search_desc": pad_to_max_N(search_desc),
        "gallery_desc": pad_to_max_N(gallery_desc),
        "search_mask": pad_to_max_N(search_mask),
        "gallery_mask": pad_to_max_N(gallery_mask),
        "search_mnt": pad_to_max_N(search_mnt),
        "gallery_mnt": pad_to_max_N(gallery_mnt),
        "index_pair": torch.stack(index_pairs)
    }
    return batch_dict


def identify(query_templates:list[dict], gallery_templates:list[dict], device:str='cpu', batch_size:int=64):
    """
    Realiza a identificação 1:N de forma otimizada, comparando uma lista de templates 
    de busca com uma lista de templates da galeria usando processamento em lote.
    """
    device = torch.device(device)
    num_queries = len(query_templates)
    num_gallery = len(gallery_templates)
    scores_matrix = np.zeros((num_queries, num_gallery))

    # Move to CPU to avoid GPU memory issues during batching
    query_templates = [{key: value.cpu() for key, value in tpl.items()} for tpl in query_templates]
    gallery_templates = [{key: value.cpu() for key, value in tpl.items()} for tpl in gallery_templates]
  
    # 1. Preparar o Dataset e DataLoader
    match_dataset = MatchDataset(query_templates, gallery_templates)
    match_loader = DataLoader(
        dataset=match_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # 2. Processar em lotes
    print("Iniciando a comparação em lote...")
    for batch in tqdm(match_loader, total=len(match_loader)):
        # Mover dados do lote para o dispositivo (GPU/CPU)
        search_feat = batch["search_desc"].to(device)
        gallery_feat = batch["gallery_desc"].to(device)
        search_mask = batch["search_mask"].to(device)
        gallery_mask = batch["gallery_mask"].to(device)
        search_mnt = batch["search_mnt"].to(device)
        gallery_mnt = batch["gallery_mnt"].to(device)
        index_pairs = batch["index_pair"]

        # 3. Calcular scores para o lote inteiro de uma vez
        with torch.no_grad():
            initial_scores = calculate_score_torchB(search_feat, gallery_feat, search_mask, gallery_mask, ndim_feat=12, Normalize=True)
            final_scores, pairs, scores, relaxed_scores, sorted_indices, n_pair = lsar_score_torchB(initial_scores, search_mnt, gallery_mnt)
        
        # 4. Atribuir os scores do lote à matriz final
        q_indices = index_pairs[:, 0]
        g_indices = index_pairs[:, 1]
        scores_matrix[q_indices, g_indices] = final_scores.cpu().numpy()

    return scores_matrix