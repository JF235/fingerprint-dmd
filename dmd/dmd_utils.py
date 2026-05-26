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

    # Non-empty path concatenates per-minutia patches of shape (1, H, W) into
    # (N, H, W); keep the empty case consistent so batched code can concat mixed
    # empty/non-empty results without shape errors.
    if mnt is None or len(mnt) == 0:
        return np.zeros((0, int(tar_shape[0]), int(tar_shape[1])), dtype=np.float32)

    patches = [_warp_affine_to_patch(img, pose, tar_shape, middle_shape, img_ppi) for pose in mnt]
    return np.concatenate(patches, axis=0)


def extract_patches_batch_gpu(images, mnts, patch_size=(128, 128), img_ppi=500, device='cuda'):
    """
    GPU-accelerated batch patch extraction. Equivalent to looping extract_patches()
    over each (image, minutia) pair, up to bilinear-interpolation rounding between
    cv2.warpAffine and F.grid_sample.

    Args:
        images: List of numpy arrays or tensors, each [H, W]
        mnts: List of minutiae arrays, each [N_i, 3] (x, y, angle_deg)
        patch_size: Tuple (H_dst, W_dst) for output patches
        img_ppi: Source image PPI; patches are normalized to 500 PPI
        device: 'cuda' or 'cpu'

    Returns:
        patches: Tensor [Total_N, 1, H_dst, W_dst]
        patch_counts: List of patch counts per image
    """
    H_dst, W_dst = int(patch_size[0]), int(patch_size[1])
    scale = img_ppi * 1.0 / 500  # middle_shape == tar_shape in extract_patches; ratio is 1
    inv_s = 1.0 / scale
    cx, cy = W_dst / 2.0, H_dst / 2.0
    device = torch.device(device)

    # Base output grid: (v, u) pixel coords of each output cell, shared across minutiae.
    uu = torch.arange(W_dst, device=device, dtype=torch.float32)
    vv = torch.arange(H_dst, device=device, dtype=torch.float32)
    gv, gu = torch.meshgrid(vv, uu, indexing='ij')  # [H_dst, W_dst]
    du_base = gu - cx
    dv_base = gv - cy

    all_patches = []
    patch_counts = []

    for img, mnt in zip(images, mnts):
        if mnt is None or len(mnt) == 0:
            patch_counts.append(0)
            continue

        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        img_t = img.to(device=device, dtype=torch.float32)
        img_t = (img_t - 127.5) / 127.5
        H_src, W_src = img_t.shape[-2], img_t.shape[-1]

        if not isinstance(mnt, torch.Tensor):
            mnt = torch.from_numpy(np.asarray(mnt))
        mnt_t = mnt.to(device=device, dtype=torch.float32)

        N = mnt_t.shape[0]
        patch_counts.append(N)

        px = mnt_t[:, 0].view(N, 1, 1)
        py = mnt_t[:, 1].view(N, 1, 1)
        ang = torch.deg2rad(mnt_t[:, 2])
        cos_a = torch.cos(ang).view(N, 1, 1)
        sin_a = torch.sin(ang).view(N, 1, 1)

        # Inverse of the cv2.getRotationMatrix2D + recenter affine: for output
        # pixel (u, v), sample source at:
        #   x_src = (cos*(u-cx) - sin*(v-cy)) / scale + px
        #   y_src = (sin*(u-cx) + cos*(v-cy)) / scale + py
        x_src = inv_s * (cos_a * du_base - sin_a * dv_base) + px  # [N, H_dst, W_dst]
        y_src = inv_s * (sin_a * du_base + cos_a * dv_base) + py

        # Normalize to grid_sample's [-1, 1] convention (align_corners=False):
        # x_norm = (2*x + 1) / W_src - 1. MUST use source dims, not patch dims.
        x_norm = (2.0 * x_src + 1.0) / W_src - 1.0
        y_norm = (2.0 * y_src + 1.0) / H_src - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1)  # [N, H_dst, W_dst, 2]

        img_batch = img_t.unsqueeze(0).unsqueeze(0).expand(N, 1, H_src, W_src)
        # padding_mode='zeros' matches cv2's borderValue=127.5 after img normalization.
        patches = F.grid_sample(img_batch, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        all_patches.append(patches)

    if not all_patches:
        return torch.zeros((0, 1, H_dst, W_dst), device=device), patch_counts

    return torch.cat(all_patches, dim=0), patch_counts


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
        # Empty batch: run one dummy patch to discover feature/mask dims,
        # then drop the row so downstream slicing lines up with (0, feat_dim).
        dummy = torch.zeros((1, 1, patches_tensor.shape[-2], patches_tensor.shape[-1]), device=device)
        with torch.no_grad():
            out = model.get_embedding(dummy)
        return {
            'feature': out['feature'][:0].cpu(),
            'mask':    out['mask'][:0].cpu(),
        }
    
    all_features = []
    all_masks = []
    
    with torch.no_grad():
        for i in range(0, N, max_batch_size):
            batch = patches_tensor[i:i+max_batch_size].to(device)
            embeddings = model.get_embedding(batch)
            
            all_features.append(embeddings['feature'].cpu())
            all_masks.append(embeddings['mask'].cpu())
    
    # Concatenate results: model.get_embedding returns (B, feat_dim) after flatten(1),
    # so the batch dim is 0, not 1.
    result = {
        'feature': torch.cat(all_features, dim=0),  # (N_total, feat_dim)
        'mask': torch.cat(all_masks, dim=0)          # (N_total, mask_dim)
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
        
        if all_patches_list:
            all_patches = np.concatenate(all_patches_list, axis=0)
            all_patches = torch.from_numpy(all_patches).to(device)
        else:
            all_patches = torch.zeros((0, 128, 128), device=device)

    embeddings = get_embeddings_batch(model, all_patches, device=device, max_batch_size=max_batch_size)
    
    # Step 3: Split embeddings back to individual templates.
    # embeddings['feature'] is (N_total, feat_dim); slice rows, not columns.
    templates = []
    start_idx = 0

    for mnt, count in zip(mnts, patch_counts):
        end_idx = start_idx + count
        mnt_arr = np.asarray(mnt, dtype=np.float32)
        if mnt_arr.ndim == 1:
            mnt_arr = mnt_arr.reshape(0, 3) if mnt_arr.size == 0 else mnt_arr[None]
        template = {
            'feature': embeddings['feature'][start_idx:end_idx].to(device),
            'mask':    embeddings['mask'][start_idx:end_idx].to(device),
            'mnt':     torch.from_numpy(mnt_arr).unsqueeze(0).to(device),
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
            suporte = torch.sum(rp * lambda_t[:, None, :], axis=-1) / torch.clamp(min_number[:, None] - 1, min=1)
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
    scores = calculate_score_torchB(search_feat, gallery_feat, search_mask, gallery_mask, ndim_feat=ndim_feat*2,  Normalize=normalize, N_mean=1327, binary=binary, f2f_type=(2,1))
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
    scores = calculate_score_torchB(search_feat, gallery_feat, search_mask, gallery_mask, ndim_feat=ndim_feat*2,  Normalize=normalize, N_mean=1327, binary=binary, f2f_type=(2,1))
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
            "search_desc": query_template['feature'],
            "gallery_desc": gallery_template['feature'],
            "search_mask": query_template['mask'],
            "gallery_mask": gallery_template['mask'],
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
    _sd = pad_to_max_N(search_desc)
    _gd = pad_to_max_N(gallery_desc)
    _sm = pad_to_max_N(search_mask)
    _gm = pad_to_max_N(gallery_mask)
    _smnt = pad_to_max_N(search_mnt)
    _gmnt = pad_to_max_N(gallery_mnt)
    
    batch_dict = {
        "search_desc": _sd,
        "gallery_desc": _gd,
        "search_mask": _sm,
        "gallery_mask": _gm,
        "search_mnt": _smnt,
        "gallery_mnt": _gmnt,
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


def match_pairs(query_templates, gallery_templates, device='cuda', batch_size=64,
                return_details=False, progress=True):
    """Score N parallel (query_i, gallery_i) template pairs in batched fashion.

    Unlike ``identify``, which computes a full Q×G matrix, this function evaluates
    exactly ``len(query_templates)`` pairs: pair *i* is
    ``(query_templates[i], gallery_templates[i])``.

    Args:
        query_templates:   list of N query template dicts (feature, mask, mnt)
        gallery_templates: list of N gallery template dicts (same length as queries)
        device:            torch device for batched scoring
        batch_size:        number of pairs scored per forward pass
        return_details:    if True, also returns per-pair LSAR minutia correspondences
        progress:          show a tqdm progress bar over batches

    Returns:
        scores: np.ndarray of shape (N,) float32 — DMD score per pair (NaN if any
                template in the pair had zero minutiae).
        If return_details=True, also returns:
            details: list of length N. details[i] is a dict with keys:
                'pairs'        : (k, 2) int64 — selected minutia index pairs
                                 (query_minutia_idx, gallery_minutia_idx) from LSAR
                'lambda'       : (k,) float32 — post-relaxation scores for each pair
                'n_pair'       : int — number of selected pairs (== len(pairs))
                Indices that fall on padded slots are filtered out so every entry
                points to a real minutia in the original (un-padded) template.
    """
    assert len(query_templates) == len(gallery_templates), \
        "query_templates and gallery_templates must have equal length"
    N = len(query_templates)
    if N == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return (empty, []) if return_details else empty

    device = torch.device(device)
    # Mirror identify(): move templates to CPU so we control device transfers.
    q_tpls = [{k: v.cpu() for k, v in t.items()} for t in query_templates]
    g_tpls = [{k: v.cpu() for k, v in t.items()} for t in gallery_templates]

    feat_dim = q_tpls[0]['feature'].shape[-1]
    mask_dim = q_tpls[0]['mask'].shape[-1]

    all_scores = np.full(N, np.nan, dtype=np.float32)
    all_details = [] if return_details else None

    iterator = range(0, N, batch_size)
    if progress:
        iterator = tqdm(iterator, total=(N + batch_size - 1) // batch_size, desc="match_pairs")

    for b0 in iterator:
        b1 = min(b0 + batch_size, N)
        qs = q_tpls[b0:b1]
        gs = g_tpls[b0:b1]
        B = len(qs)

        nq_list = [t['feature'].shape[0] for t in qs]
        ng_list = [t['feature'].shape[0] for t in gs]
        max_Nq = max(nq_list)
        max_Ng = max(ng_list)

        # Any pair with an empty template gets NaN score; LSAR shortcut would
        # need to be called with at least 1 mnt per side, so we skip those rows.
        valid_rows = [i for i in range(B) if nq_list[i] > 0 and ng_list[i] > 0]
        if not valid_rows:
            continue

        Bv = len(valid_rows)
        # NaN-padded tensors: lsar_score_torchB detects per-row mnt counts via
        # ``torch.sum(~torch.isnan(S[:, :, 0]), dim=-1)`` so NaN is meaningful.
        q_feat = torch.full((Bv, max_Nq, feat_dim), float('nan'), dtype=torch.float32)
        q_mask = torch.full((Bv, max_Nq, mask_dim), float('nan'), dtype=torch.float32)
        q_mnt  = torch.full((Bv, max_Nq, 3),        float('nan'), dtype=torch.float32)
        g_feat = torch.full((Bv, max_Ng, feat_dim), float('nan'), dtype=torch.float32)
        g_mask = torch.full((Bv, max_Ng, mask_dim), float('nan'), dtype=torch.float32)
        g_mnt  = torch.full((Bv, max_Ng, 3),        float('nan'), dtype=torch.float32)

        for vi, src_i in enumerate(valid_rows):
            qt = qs[src_i]
            gt = gs[src_i]
            nq, ng = nq_list[src_i], ng_list[src_i]
            q_feat[vi, :nq] = qt['feature']
            q_mask[vi, :nq] = qt['mask']
            # mnt is stored as (1, N, 3); squeeze the leading batch dim.
            q_mnt[vi, :nq] = qt['mnt'].squeeze(0)[:nq]
            g_feat[vi, :ng] = gt['feature']
            g_mask[vi, :ng] = gt['mask']
            g_mnt[vi, :ng] = gt['mnt'].squeeze(0)[:ng]

        q_feat = q_feat.to(device, non_blocking=True)
        q_mask = q_mask.to(device, non_blocking=True)
        q_mnt  = q_mnt.to(device,  non_blocking=True)
        g_feat = g_feat.to(device, non_blocking=True)
        g_mask = g_mask.to(device, non_blocking=True)
        g_mnt  = g_mnt.to(device,  non_blocking=True)

        with torch.no_grad():
            S = calculate_score_torchB(
                q_feat, g_feat, q_mask, g_mask,
                ndim_feat=12, Normalize=True, N_mean=1327,
            )
            final_score, pairs, _scores, relaxed, sorted_idx, n_pair = \
                lsar_score_torchB(S, q_mnt, g_mnt)

        final_np   = final_score.cpu().numpy()
        for vi, src_i in enumerate(valid_rows):
            all_scores[b0 + src_i] = final_np[vi]

        if return_details:
            pairs_np   = pairs.cpu().numpy()      # (Bv, max_n_pair_candidates, 2)
            relaxed_np = relaxed.cpu().numpy()    # (Bv, max_n_pair_candidates)
            sorted_np  = sorted_idx.cpu().numpy() # (Bv, max_n_pair_candidates)
            n_pair_np  = n_pair.cpu().numpy()     # (Bv,)
            # Sparse, per-batch fill so all_details[i] aligns with input index.
            slot_by_idx = {b0 + src_i: vi for vi, src_i in enumerate(valid_rows)}
            for i in range(b0, b1):
                vi = slot_by_idx.get(i)
                if vi is None:
                    all_details.append({'pairs': np.empty((0, 2), dtype=np.int64),
                                        'lambda': np.empty((0,), dtype=np.float32),
                                        'n_pair': 0})
                    continue
                k = int(n_pair_np[vi])
                top = sorted_np[vi, :k]
                p = pairs_np[vi, top]               # (k, 2): (q_idx, g_idx)
                lam = relaxed_np[vi, top]
                # Filter correspondences pointing into the NaN-padded region.
                src_nq, src_ng = nq_list[i - b0], ng_list[i - b0]
                valid = (p[:, 0] < src_nq) & (p[:, 1] < src_ng)
                all_details.append({
                    'pairs': p[valid].astype(np.int64),
                    'lambda': lam[valid].astype(np.float32),
                    'n_pair': int(valid.sum()),
                })

    if return_details:
        return all_scores, all_details
    return all_scores