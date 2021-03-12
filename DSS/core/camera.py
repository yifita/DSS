import torch
from pytorch3d.renderer.cameras import (PerspectiveCameras,
                                        look_at_view_transform)


class CameraSampler(object):
    """
    create camera transformations looking at the origin of the coordinate
    from varying distance

    Attributes:
        R, T: (num_cams_total, 3, 3) and (num_cams_total, 3)
        camera_type (Class): class to create a new camera
        camera_params (dict): camera parameters to call camera_type
            (besides R, T)
    """

    def __init__(self, num_cams_total, num_cams_batch,
                 distance_range=(5, 10), sort_distance=True,
                 return_cams=True,
                 camera_type=PerspectiveCameras, camera_params=None):
        """
        Args:
            num_cams_total (int): the total number of cameras to sample
            num_cams_batch (int): the number of cameras per iteration
            distance_range (tensor or list): (num_cams_total, 2) or (1, 2)
                the range of camera distance for uniform sampling
            sort_distance: sort the created camera transformations by the
                distance in ascending order
            return_cams (bool): whether to return camera instances or just the R,T
            camera_type (class): camera type from pytorch3d.renderer.cameras
            camera_params (dict): camera parameters besides R, T
        """
        self.num_cams_batch = num_cams_batch
        self.num_cams_total = num_cams_total

        self.sort_distance = sort_distance
        self.camera_type = camera_type
        self.camera_params = {} if camera_params is None else camera_params

        # create camera locations
        distance_scale = distance_range[:, -1] - distance_range[:, 0]
        distances = torch.rand(num_cams_total) * distance_scale + \
            distance_range[:, 0]
        if sort_distance:
            distances, _ = distances.sort(descending=True)
        azim = torch.rand(num_cams_total) * 360 - 180
        elev = torch.rand(num_cams_total) * 180 - 90
        at = torch.rand((num_cams_total, 3)) * 0.1 - 0.05
        self.R, self.T = look_at_view_transform(
            distances, elev, azim, at=at, degrees=True)

        self._idx = 0

    def __len__(self):
        return (self.R.shape[0] + self.num_cams_batch - 1) // \
            self.num_cams_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration
        start_idx = self._idx * self.num_cams_batch
        end_idx = min(start_idx + self.num_cams_batch, self.R.shape[0])
        cameras = self.camera_type(R=self.R[start_idx:end_idx],
                                   T=self.T[start_idx:end_idx],
                                   **self.camera_params)
        self._idx += 1
        return cameras
