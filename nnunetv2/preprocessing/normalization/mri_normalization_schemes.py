import ants
import intensity_normalization.normalize as inorm

from .default_normalization_schemes import ImageNormalization, ZScoreNormalization


class MRINormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def __init__(self):
        super(MRINormalization, self).__init__()
        self.nyul = inorm.NyulNormalizer()
        self.zscore = ZScoreNormalization(use_mask_for_norm=self.use_mask_for_norm)
        self.modality = None

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)

        # z-score normalization masking will inherit
        normalized_image = self.zscore.run(image, seg)

        # n4 bias field correction
        ants_image = ants.from_numpy(masked_image)
        ants_image = ants.n4_bias_field_correction(ants_image)

        numpy_image = ants_image.numpy()

        # nyul normalization
        if self.modality is not None:
            normalized_image = self.nyul.normalize_image(normalized_image) #fix
        else:
            normalized_image = numpy_image

        return normalized_image



if __name__ == '__main__':
    import os

    train_nyul_path = ""

    file_paths = os.listdir(train_nyul_path)

    nyul = inorm.NyulNormalize()

    nyul._fit(file_paths, modalities="T1")

