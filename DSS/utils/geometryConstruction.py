import torch, math
from .mathHelper import normalize

def circle2d(samples):
    """ returns a torch tensor (x,y) (nx2) of 2d coordinates for a circle of radius r sampled at "samples" regular position """
    angls = 2.0*math.pi*torch.tensor([range(samples)]).float()/samples;
    pos2d = torch.cat((angls.cos().transpose(0,1), angls.sin().transpose(0,1)), 1);
    return pos2d

def circle3dmesh(samples):
    # get 2d coordinates
    circ2d = circle2d(samples)
    # append zero z-coordinate
    circ3d = torch.cat((circ2d, torch.zeros((circ2d.size()[0], 1), dtype=torch.float)), 1)
    # Add central point
    circ3d = torch.cat((torch.tensor([[0.0,0.0,0.0]], dtype=torch.float), circ3d), 0)
    # first point is now central point
    connectivity = []
    for i in range(samples):
        connectivity.append([0, 1+i, 1+(i+1)%samples])
    connectivity = torch.tensor(connectivity);
    return (circ3d, connectivity)

def triangle3dmesh():
    """
    Creates a single equilateral triangle with outer-circle radius 1, it is planar in xy, centered at (0,0,0) and one corner lies at (0,1,0)
    """
    points = torch.tensor([[math.cos(0), math.sin(0), 0], [math.cos(2*math.pi/3), math.sin(2*math.pi/3), 0], [math.cos(4/3*math.pi), math.sin(4/3*math.pi), 0]], dtype=torch.float)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int)
    return (points, faces)

def normal2RotMatrix(normal, device=None):
    z = torch.tensor([0.0,0.0,1.0], device=device)
    # TODO: check that normal is not aligned with z
    if abs(normal.dot(z)) > 0.999:
        ## close to z!
        # TODO: remove hack and make it proper
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device)
    u1 = normal.cross(z)
    u1 = u1 / u1.norm()
    u2 = normal.cross(u1)
    R = torch.stack((u1, u2, normal), 0)
    return R

def chessboard3dPoints(width, height, gammax, gammay):
    board = torch.zeros([width*height, 3]);
    for i in range(width):
        for j in range(height):
            board[i*height + j, 0] = i/width
            board[i*height + j, 1] = j/height
    board[:, 0] = board[:, 0].pow(gammax)
    board[:, 1] = board[:, 1].pow(gammay)
    normals = torch.zeros([width*height, 3])
    normals[:, 2] = 1.0
    colors = torch.ones([width*height, 3])
    for i in range(width):
        for j in range(height):
            if (i+j)%2 == 0:
                colors[i * height + j, :] = 0.0
    return (board, normals, colors)


def sphere3dPoints(samples, layers):
    # get 2d coordinates
    circ2d = circle2d(samples)
    # append zero z-coordinate
    circ3d = torch.cat((circ2d, torch.zeros((circ2d.size()[0], 1), dtype=torch.float)), 1)
    sphere = torch.tensor([[0.0,0.,1.], [0.,0.,-1.]])
    for l in range(1,layers+1):
        ring = circ3d
        layer = math.sin(math.pi*l/layers) * ring
        layer[:,2] = math.cos(math.pi*l/layers)
        sphere = torch.cat((sphere, layer), 0)
    colors = 255*torch.ones([samples*layers+2, 3])
    normals = normalize(sphere, 1)
    return (sphere, normals, colors)

def saddle3dPoints(width, height, ax, ay):
    board = torch.zeros([width*height, 3]);
    normals = torch.zeros([width*height, 3])
    for i in range(width):
        for j in range(height):
            x = (i/width)*2.0 - 1.0
            y = (j/height)*2.0 - 1.0
            ii = i*height + j
            board[ii, 0] = x
            board[ii, 1] = y
            board[ii, 2] = ax*x*x + ay*y*y
            normals[ii, 0] = 2*ax*x
            normals[ii, 1] = 2*ay*y
            normals[ii, 2] = -1
            normals[ii, :] = normals[ii, :]/torch.norm(normals[ii, :])
    colors = torch.ones([width*height, 3])
    return (board, normals, colors)

def pcl2Mesh(points, normals, colors, replacementMeshVerts, replacementMeshFaces, device=None):
    pointsCount = points.size()[0]
    vindex = 0
    # Position, Normal, RGB
    verts = torch.zeros([pointsCount*replacementMeshVerts.size()[0], 3+3+3], device=device)
    # connectivity, RGB
    faces = torch.zeros([pointsCount*replacementMeshFaces.size()[0], 3+3], dtype=torch.int, device=device)
    RV = replacementMeshVerts.size()[0]
    RF = replacementMeshFaces.size()[0]
    for i in range(pointsCount):
        p = points[i]
        # rotate according to normal
        R = normal2RotMatrix(normals[i, :], device=device)
        circ = p[None, 0:3] + replacementMeshVerts.matmul(R);
        s = i*RV
        e = (i+1)*RV
        verts[s:e, 0:3] = circ
        verts[s:e, 3:6] = normals[i, :]
        verts[s:e, 6:9] = 255*colors[i, :]
        pConn = replacementMeshFaces + i*RV;
        s = i*RF
        e = (i+1)*RF
        faces[s:e, 0:3] = pConn
        faces[s:e, 3:6] = 255*colors[i, :]
    # verts: n*(triangles+1) x (3+3+3)
    # faces: n*triangles x 3
    return (verts, faces)



