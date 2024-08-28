import ants
import numpy as np
import os

from .default_normalization_schemes import ImageNormalization, ZScoreNormalization

current_dir = os.path.dirname(os.path.abspath(__file__))
# precomputed nyul percentiles from IXI training data.
t1_nyul_path = os.path.join(current_dir, "nyul_t1.npy")
t2_nyul_path = os.path.join(current_dir, "nyul_t2.npy")


class MRINormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def __init__(self, *args, **kwargs):
        super(MRINormalization, self).__init__(*args, **kwargs)
        self.nyul = NyulNormalize()
        self.modality = None

    def _zscore(self, image, seg):
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            mean = image.mean()
            std = image.std()
            image -= mean
            image /= (max(std, 1e-8))
        return image

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)

        # z-score normalization masking will inherit
        normalized_image = self._zscore(image, seg)

        if self.use_mask_for_norm:
            mask = seg >= 0
            normalized_image[~mask] = 0

        # nyul normalization
        normalized_image = self.nyul.normalize_image(normalized_image)

        return normalized_image


class MRINormalizationT1(MRINormalization):
    def __init__(self, *args, **kwargs):
        super(MRINormalizationT1, self).__init__(*args, **kwargs)
        self.nyul.load(t1_nyul_path)


class MRINormalizationT2(MRINormalization):
    def __init__(self, *args, **kwargs):
        super(MRINormalizationT2, self).__init__(*args, **kwargs)
        self.nyul.load(t2_nyul_path)


class NyulNormalize:
    '''
    implementation of Nyul and Udupa's intensity normalization method
    paper: 10.1109/42.836373

    '''
    def __init__(self, standard_scale = None):
        self.standard_scale = standard_scale
        self.percentiles = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

    def _compute_percentiles(self, image, percentiles):
        return np.percentile(image, percentiles)

    def fit(self, file_paths, zscore=True, use_ray=False):
        if use_ray:
            if zscore:
                @ray.remote
                def get_array(path):
                    image = ants.image_read(path).numpy()
                    image = (image - np.mean(image)) / np.std(image)
                    percentiles = self._compute_percentiles(image,
                                                            self.percentiles)
                    return percentiles
            else:
                @ray.remote
                def get_array(path):
                    image = ants.image_read(path).numpy()
                    percentiles = self._compute_percentiles(image,
                                                            self.percentiles)
                    return percentiles

            ray.init()
            futures = [get_array.remote(path) for path in file_paths]
            all_percentiles = ray.get(futures)
            ray.shutdown()
        else:
            images = [ants.image_read(path).numpy() for path in file_paths]
            if zscore:
                images = [(image - np.mean(image)) / np.std(image) for image in images]

            all_percentiles = []
            for image in images:
                percentiles.append(self._compute_percentiles(image, self.percentiles))

        self.standard_scale = np.mean(all_percentiles, axis=0)

    def normalize_image(self, image):
        return np.interp(image, self.percentiles, self.standard_scale)

    def save(self, path):
        np.save(path, self.standard_scale)

    def load(self, path):
        self.standard_scale = np.load(path)
