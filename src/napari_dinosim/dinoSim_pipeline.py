import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .data_utils_biapy import crop_data_with_overlap, merge_data_with_overlap
from .utils import resizeLongestSide, mirror_border, remove_padding

class DinoSim_pipeline():
    def __init__(self, model, model_patch_size, device, img_preprocessing, 
                 feat_dim, dino_image_size=518, ):
        self.model = model
        self.dino_image_size = dino_image_size
        self.patch_h = self.patch_w = self.embedding_size = dino_image_size//model_patch_size
        self.img_preprocessing = img_preprocessing
        self.device = device
        self.feat_dim = feat_dim

        self.reference_color = torch.zeros(feat_dim, device=device)
        self.reference_emb = torch.zeros((self.embedding_size*self.embedding_size, feat_dim), device=device)
        self.exist_reference = False

        self.embeddings = np.array([])
        self.emb_precomputed = False
        self.original_size = []
        self.overlap = (0.5,0.5)
        self.padding = (0,0)
        self.crop_shape = (518,518,1)
        self.resized_ds_size, self.resize_pad_ds_size = [], []

    def pre_compute_embeddings(self, dataset, overlap = (0.5,0.5), padding=(0,0), 
                                    crop_shape=(512,512,1), verbose = True, batch_size=1,
                        ):
        print('Precomputing embeddings')
        self.original_size = dataset.shape
        self.overlap = overlap
        self.padding = padding
        self.crop_shape = crop_shape
        b,h,w,c = dataset.shape
        self.resized_ds_size, self.resize_pad_ds_size = [], []
        if h<crop_shape[0] and w<crop_shape[0]:
            dataset = np.array([resizeLongestSide(np_image, crop_shape[0]) for np_image in dataset])
            if len(dataset.shape) == 3:
                dataset = dataset[...,np.newaxis]
            self.resized_ds_size = dataset.shape
        if dataset.shape[1]%crop_shape[0] != 0 or dataset.shape[2]%crop_shape[1] != 0:
            desired_h, desired_w = np.ceil(dataset.shape[1]/crop_shape[0])*crop_shape[0], np.ceil(dataset.shape[2]/crop_shape[1])*crop_shape[1]
            dataset = np.array([mirror_border(np_image, sizeH=int(desired_h), sizeW=int(desired_w)) for np_image in dataset])
            self.resize_pad_ds_size = dataset.shape

        # needed format: b,h,w,c
        windows = crop_data_with_overlap(dataset, crop_shape=crop_shape, overlap=overlap, padding=padding, verbose=None)
        windows = torch.tensor(windows, device=self.device)
        prep_windows = self.img_preprocessing(windows)

        self.delete_precomputed_embeddings()
        self.embeddings = torch.zeros((len(windows), self.patch_h, self.patch_w, self.feat_dim), device=self.device)

        following_f = tqdm if verbose else lambda aux: aux
        for i in following_f(range(0,len(prep_windows), batch_size)):
            batch = prep_windows[i:i+batch_size]
            b,h,w,c = batch.shape # b,h,w,c
            crop_h, crop_w, _ = crop_shape
            overlap = (overlap[0] if w > crop_w else 0, 
                    overlap[1] if h > crop_h else 0)

            with torch.no_grad():
                encoded_window = self.model.forward_features(batch)['x_norm_patchtokens']
            self.embeddings[i:i+batch_size] = encoded_window.reshape(encoded_window.shape[0], self.patch_h, self.patch_w, self.feat_dim) # use all dims
                
        self.emb_precomputed = True
    
    def delete_precomputed_embeddings(self, ):
        del self.embeddings
        self.embeddings = np.array([])
        self.emb_precomputed = False
        torch.cuda.empty_cache()

    def delete_references(self,):
        del  self.reference_color, self.reference_emb, self.exist_reference
        self.reference_color = []
        self.reference_emb = []
        self.exist_reference = False
        torch.cuda.empty_cache()

    def set_reference_vector(self, list_coords, filter=None):
        self.delete_references()
        if len(self.resize_pad_ds_size) > 0:
            b, h, w, c = self.resize_pad_ds_size
            if len(self.resized_ds_size) > 0:
                original_resized_h, original_resized_w = self.resized_ds_size[1:3]
            else:
                original_resized_h, original_resized_w = self.original_size[1:3]
        elif len(self.resized_ds_size) > 0:
            b, h, w, c = self.resized_ds_size
            original_resized_h, original_resized_w = h, w
        else:
            b, h, w, c = self.original_size
            original_resized_h, original_resized_w = h, w

        n_windows_h = int(np.ceil(h / self.crop_shape[0]))
        n_windows_w = int(np.ceil(w / self.crop_shape[1]))

        # Calculate actual scaling factors
        scale_x = original_resized_w / self.original_size[2]
        scale_y = original_resized_h / self.original_size[1]

        # Calculate padding
        pad_left = (w - original_resized_w) / 2
        pad_top = (h - original_resized_h) / 2

        list_ref_colors, list_ref_embeddings = [], []
        for n, x, y in list_coords:
            # Apply scaling and padding to coordinates
            x_transformed = x * scale_x + pad_left
            y_transformed = y * scale_y + pad_top

            # Calculate crop index and relative position within crop
            n_crop = int(np.floor(x_transformed / self.crop_shape[1]) + 
                        np.floor(y_transformed / self.crop_shape[0]) * n_windows_w)
            x_coord = (x_transformed % self.crop_shape[1]) / self.crop_shape[1]
            y_coord = (y_transformed % self.crop_shape[0]) / self.crop_shape[0]

            emb_id = int(n_crop + n * n_windows_h * n_windows_w)
            
            # Validate embedding index
            if emb_id >= len(self.embeddings):
                raise ValueError(f"Invalid embedding index {emb_id} for coordinates ({n}, {x}, {y})")

            x_coord = min(round(x_coord * self.embedding_size), self.embedding_size-1)
            y_coord = min(round(y_coord * self.embedding_size), self.embedding_size-1)

            list_ref_colors.append(self.embeddings[emb_id][y_coord, x_coord])
            list_ref_embeddings.append(self.embeddings[emb_id])

        list_ref_colors, list_ref_embeddings = torch.stack(list_ref_colors), torch.stack(list_ref_embeddings)
        assert len(list_ref_colors) > 0, "No binary objects found in given masks"

        self.reference_color = torch.mean(list_ref_colors, dim=0)
        self.reference_emb = list_ref_embeddings
        self.generate_pseudolabels(filter)
        self.exist_reference = True

    def generate_pseudolabels(self, filter=None):
        reference_embeddings = self.reference_emb.view(-1, self.reference_emb.shape[-1])
        distances = torch.cdist(reference_embeddings, self.reference_color[None], p=2)

        if filter!=None:
            distances = distances.view((self.reference_emb.shape[0], 1, int(self.embedding_size), int(self.embedding_size)))
            distances = filter(distances)

        # normalize per image
        distances = self.quantile_normalization(distances)

        self.reference_pred_labels = distances.view(-1,1)

    def quantile_normalization(self, tensor, lower_quantile = 0.01, upper_quantile = 0.99):
        sorted_tensor, _ = tensor.flatten().sort()
        lower_bound = sorted_tensor[int(lower_quantile * (len(sorted_tensor)-1) )]
        upper_bound = sorted_tensor[int(upper_quantile * (len(sorted_tensor)-1) )]

        clipped_tensor = torch.clamp(tensor, lower_bound, upper_bound)
        return (clipped_tensor - lower_bound) / (upper_bound - lower_bound + 1e-8)

    def get_ds_distances_sameRef(self, verbose=True, k=5):
        distances = []
        following_f = tqdm if verbose else lambda x: x
        for i in following_f(range(len(self.embeddings))):
            encoded_windows = self.embeddings[i]
            total_features = encoded_windows.reshape(1, self.patch_h, self.patch_w, self.feat_dim).to(device=self.device) # use all dims

            mask = self._get_torch_knn_mask(total_features[0], k=k) # get distance map
            distances.append(mask.cpu().numpy())
        return np.array(distances)
    
    def _get_torch_knn_mask(self, image_representation, k=5):
        old_shape = image_representation.shape
        embs = image_representation.view(-1,old_shape[-1])
        ref = self.reference_emb.view(-1, self.reference_emb.shape[-1])
        distances_knn = torch.cdist(embs, ref, p=2)
        knn_values, knn_indices = torch.topk(distances_knn, k=k, largest=False, dim=1)
        knn_labels = self.reference_pred_labels[knn_indices] 

        predictions = torch.mean(knn_labels.to(torch.float32), dim=1)
        predictions = predictions.view(old_shape[:-1])
        return predictions
    
    def distance_post_processing(self, distances, low_res_filter, upsampling_mode):
        if len(self.resize_pad_ds_size) > 0:
            ds_shape = self.resize_pad_ds_size
        elif len(self.resized_ds_size) > 0:
            ds_shape = self.resized_ds_size
        else:
            ds_shape = self.original_size
        ds_shape = list(ds_shape)
        ds_shape[-1] = 1 # distances only has 1 channel
        b,h,w,c = ds_shape
        distances = np.array(distances)[..., np.newaxis]
        emb_h, emb_w = ((np.array((h,w))/self.crop_shape[:2])*self.embedding_size).astype(np.uint16)
        recons_parts = merge_data_with_overlap(distances, (b,emb_h,emb_w,c), overlap=self.overlap, padding=self.padding, verbose=False, out_dir=None, prefix="")
        recons_parts = torch.tensor(recons_parts, device=self.device).permute(0,3,1,2) # b,c,h,w

        if low_res_filter != None:
            recons_parts = low_res_filter(recons_parts)
            # Ensure 4D tensor (batch, channel, height, width)
            if len(recons_parts.shape) == 3:
                recons_parts = recons_parts.unsqueeze(1 if c==1 else 0)  # Add channel or batch dimension if missing
            elif len(recons_parts.shape) == 2:
                recons_parts = recons_parts.unsqueeze(0).unsqueeze(1)  # Add batch and channel dimensions

        if upsampling_mode != None:
            #resize to padded image size or resized image (small images)
            if len(self.resize_pad_ds_size) > 0 or len(self.resized_ds_size) > 0:
                recons_parts = F.interpolate(recons_parts, size=(h, w), mode=upsampling_mode, align_corners=False)

            #remove padding
            if len(self.resize_pad_ds_size) > 0:
                b,h,w,c = self.resized_ds_size if len(self.resized_ds_size)>0 else self.original_size
                recons_parts = remove_padding(recons_parts, (h,w))

            # resize to original size
            b,h,w,c = self.original_size
            recons_parts = F.interpolate(recons_parts, size=(h, w), mode=upsampling_mode, align_corners=False)

        return recons_parts.permute(0,2,3,1).cpu().numpy() # b,h,w,c
    
    def __del__(self):
        self.delete_references()
        self.delete_precomputed_embeddings()
        self.model = None
        torch.cuda.empty_cache()

    def save_reference(self, filepath):
        """Save the current reference color and embeddings to a file.
        
        Args:
            filepath (str): Path where to save the reference data
        """
        if not self.exist_reference:
            raise ValueError("No reference exists to save")
            
        torch.save({
            'reference_color': self.reference_color,
            'reference_emb': self.reference_emb,
            'reference_pred_labels': self.reference_pred_labels,
            'embedding_size': self.embedding_size,
            'feat_dim': self.feat_dim
        }, filepath)

    def load_reference(self, filepath, filter=None):
        """Load reference color and embeddings from a file.
        
        Args:
            filepath (str): Path to the saved reference data
            filter: Optional filter to apply when generating pseudolabels
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        
        # Verify compatibility
        if checkpoint['embedding_size'] != self.embedding_size:
            raise ValueError(f"Incompatible embedding_size: saved {checkpoint['embedding_size']} vs current {self.embedding_size}")
        if checkpoint['feat_dim'] != self.feat_dim:
            raise ValueError(f"Incompatible feat_dim: saved {checkpoint['feat_dim']} vs current {self.feat_dim}")
            
        self.reference_color = checkpoint['reference_color']
        self.reference_emb = checkpoint['reference_emb']
        self.reference_pred_labels = checkpoint['reference_pred_labels']
        self.exist_reference = True
        
        if filter is not None:
            self.generate_pseudolabels(filter)
