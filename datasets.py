class BraTSDataset(Dataset):
    def __init__(self, data_root, min_tumor_ratio=0.01, target_size=(224, 224)):
        self.data_root = data_root
        self.min_tumor_ratio = min_tumor_ratio
        self.target_size = target_size

        self.patient_ids = sorted(os.listdir(data_root))
        self.index = []

        self._build_index()

    def _build_index(self):
        for patient_id in self.patient_ids:
            try:
                seg_path = self._get_modality_path(patient_id, "seg")
                seg_volume = nib.load(seg_path).get_fdata()
            except (FileNotFoundError, Exception) as e:
                print(f"Skipping patient {patient_id}: {e}")
                continue

            total_pixels = seg_volume.shape[0] * seg_volume.shape[1]
            num_slices = seg_volume.shape[2]

            for slice_idx in range(num_slices):
                tumor_pixels = np.count_nonzero(seg_volume[:, :, slice_idx])
                ratio = tumor_pixels / total_pixels
                if ratio >= self.min_tumor_ratio:
                    self.index.append((patient_id, slice_idx))

    def _get_modality_path(self, patient_id, modality):
        filename = f"{patient_id}_{modality}.nii"
        return os.path.join(self.data_root, patient_id, filename)

    def _load_patient_volumes(self, patient_id):
        volumes = {}
        for modality in MODALITIES:
            path = self._get_modality_path(patient_id, modality)
            volumes[modality] = nib.load(path).get_fdata()

        seg_path = self._get_modality_path(patient_id, "seg")
        volumes["seg"] = nib.load(seg_path).get_fdata()

        return volumes

    def _normalize_slice(self, slice_2d):
        mean = slice_2d.mean()
        std = slice_2d.std()
        if std > 0:
            slice_2d = (slice_2d - mean) / std
        return slice_2d

    def _resize_slice(self, slice_2d):
        import cv2
        return cv2.resize(slice_2d, self.target_size, interpolation=cv2.INTER_LINEAR)

    def _resize_mask(self, mask_2d):
        import cv2
        return cv2.resize(mask_2d, self.target_size, interpolation=cv2.INTER_NEAREST)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        patient_id, slice_idx = self.index[idx]
        volumes = self._load_patient_volumes(patient_id)

        channels = []
        for modality in MODALITIES:
            slice_2d = volumes[modality][:, :, slice_idx]
            slice_2d = self._resize_slice(slice_2d)
            slice_2d = self._normalize_slice(slice_2d)
            channels.append(slice_2d)

        image = np.stack(channels, axis=0).astype(np.float32)

        seg_slice = volumes["seg"][:, :, slice_idx]
        seg_slice = self._resize_mask(seg_slice)
        mask = (seg_slice > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        return image_tensor, mask_tensor
