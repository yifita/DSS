import torch
from .camera import PinholeCamera
from .cloud import PointCloud

class Scene:
    def __init__(self, device=None):
        self.device = device
        # 3d color vector for ambient lighting
        self.ambientLight = torch.full([3,], 0.3, device=self.device)
        # nx6 matrix of simple point lights in world coordinates: (x,y,z,r,g,b)
        self.pointlightPositions = torch.zeros(0, 0, device=self.device)
        self.pointlightColors = torch.zeros(0, 0, device=self.device)
        self.sunDirections = torch.zeros(0, 0, device=self.device)
        self.sunColors = torch.zeros(0, 0, device=self.device)
        self.cameras = [PinholeCamera(device=self.device), ]
        self.cloud = PointCloud(device=self.device)
        self.background_color = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.merge_strategy = 'overwrite'

    def loadPoints(self, points):
        self.cloud.localPoints = points[:, 0:3]
        self.cloud.localNormals = points[:, 3:6]
        self.cloud.color = points[:, 6:9]