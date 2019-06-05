import torch


class PointCloud:
    def __init__(self, device=None):
        self.device = device
        self.localPoints = torch.tensor([], device=self.device)
        self.localNormals = torch.tensor([], device=self.device)
        self.color = torch.tensor([], device=self.device)
        # options: albedo, normal, diffuse
        self.shading = "normal"
        self.cutoffThreshold = 1.0
        self.boundingBoxCutoffThresholdMultiplier = 1
        self.boundingBoxExtraMargin = 1
        self.backwardBoundingBoxMargin = 50
        self.backfaceCulling = True
        self.position = torch.zeros([3], device=self.device)
        self.rotation = torch.eye(3, device=self.device)
        self.scale = torch.tensor(1.0, device=self.device)
        # options: nearestNeighbor, constant
        self.VrkMode = "constant"
        # we can assume that the sampling pattern in the local planar area around u_k is a jittered grid with sidelength h in object space.
        self.Vrk_h = 0.02
        # used to compute Vrk_h
        self.Vrk_nn_k = 6
        # Context variables, don't touch, splatter overwrites this
        # cameraPoints: nx4 matrix of homogeneous coordinates in the camera frame
        self.cameraPoints = None
        # cameraNormals: normals transformed in camera coordinate frame
        self.cameraNormals = None
        # points projected to camera plane
        self.projPoints = None
        # x0/x1 in camera plane: x0 is the vector from p_k to p_k moved by one pixel along the x axis -> movement of one pixels distance
        self.projX0 = None
        self.projX1 = None
        # absolute dot product of point k and camera normal k
        self.normalAngle = None
        # vector tilde x0/x1 in normal plane from projection of x0/x1, relative but not normalized
        # = movement vector corresponding to the distance between two pixels
        self.x0plane = None
        self.x1plane = None

    def model2WorldMatrix(self):
        """
        4x4 Model matrix that transforms points from local/model space to world space
        """
        Rs = self.scale * self.rotation
        mw = torch.eye(4, device=self.device)
        mw[0:3, 0:3] = Rs
        mw[0:3, 3] = self.position
        return mw
