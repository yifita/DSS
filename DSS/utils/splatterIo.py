import torch
import math
import imageio
import csv
import os
from plyfile import PlyData, PlyElement
import json
import numpy as np
from glob import glob
from pytorch_points.utils.pc_utils import save_ply
from ..core.renderer import DSS
from ..core.cloud import PointCloud
from ..core.scene import Scene
from ..core.camera import PinholeCamera
from .mathHelper import dot, div, mul, det22, normalize
from .matrixConstruction import rotationMatrixX, rotationMatrixY, rotationMatrixZ, rotationMatrix, lookAt, batchLookAt

def saveAsPng(torchImage, fileName, cmin=None, cmax=None):
    """
    No unexpected index flips are performed:
    A matrix [[0, 120], [200, 255]] will create an image s.t.:
    the upper left corner is black
    the upper right corner is the darker grey
    the lower left corner is the brigter grey
    the lower right corner is white
    """
    torchImage = torchImage
    alpha = torch.sum(torchImage.abs(), dim=-1, keepdim=True)
    alpha = torch.where(alpha == 0, torch.zeros_like(alpha), torch.full_like(alpha, 255.0))
    h, w = torchImage.shape[:2]
    if torchImage.shape[-1] == 1:
        torchImage = torchImage.expand(-1, -1, 3)
    dirP = os.path.dirname(fileName)
    if dirP != '':
        os.makedirs(dirP, exist_ok=True)
    if cmin is None:
        cmin = torch.min(torchImage).cpu().item()
    if cmax is None:
        cmax = torch.max(torchImage).cpu().item()
    pixels = torchImage - cmin
    pixels = pixels/(cmax-cmin)
    pixels = pixels * 255.0
    pixels = torch.cat([pixels, alpha], dim=-1)
    imageio.imwrite(fileName, pixels.numpy().astype(np.uint8))

def readImage(fileName, device=None):
    img = imageio.imread(fileName)
    torchImage = torch.tensor(img, dtype=torch.float, device=device)/255.0
    return torchImage

def readMatrix(fileName, device=None):
    with open(fileName, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        rows = []
        for row in reader:
            r = list(map(lambda s : float(s), row))
            rows.append(r)
        return torch.tensor(rows, device=device)

def writeMatrix(fileName, matrix):
    with open(fileName, 'w') as csvfile:
        matrix = matrix.detach().cpu().numpy()
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerows(matrix)

def checkScenePaths(scenePathsInput):
    scenePaths = []
    for scenePath in scenePathsInput:
        if os.path.isfile(scenePath):
            scenePaths.append(scenePath)
        elif os.path.isdir(scenePath):
            files = [os.path.join(scenePath, f) for f in os.listdir(scenePath) if (f.split('.')[-1] == 'json' and os.path.isfile(os.path.join(scenePath,f)))]
            scenePaths.extend(files)
        else:
            print("Cloudn't find " + scenePath + ", skipping.")
    scenePaths = list(set(scenePaths))
    scenePaths.sort()
    return scenePaths

def writeCameras(scene, savePath):
    position = torch.cat([c.position for c in scene.cameras], dim=0)
    normal = torch.cat([c.rotation[:,:,2] for c in scene.cameras], dim=0)
    save_ply(position.cpu().numpy(), savePath, normals=normal.cpu().numpy())

def getBasename(path):
    basename = ".".join(os.path.basename(path).split('.')[:-1])
    return basename

def writeScene(scene, scenepath, cloudpath = None, text=False, indent=8):
    """ Write the scene at `scenepath` and the point cloud as a ply file at `cloudpath` """
    relCloudPath= cloudpath
    if relCloudPath is not None:
        relCloudPath = os.path.relpath(relCloudPath, os.path.dirname(scenepath))
    jsonDict = scene2Json(scene, relCloudPath)
    dirP = os.path.dirname(scenepath)
    os.makedirs(dirP, exist_ok=True)
    with open(scenepath, "w") as wfile:
        json.dump(jsonDict, wfile, indent=indent)
    cloud = scene.cloud
    if cloudpath is not None:
        writePlyCloud(cloudpath, cloud.localPoints, cloud.localNormals, cloud.color, text=text)

def readScene(filepath, device=None):
    data = None
    with open(filepath, "r") as rfile:
        data = json.load(rfile)
    if data == None:
        return None
    baseDir = os.path.dirname(os.path.realpath(filepath))
    return json2Scene(data, baseDir, device=device)

def scene2Json(scene, cloudpath=None):
    out = {}
    out['background-color'] = vector2Json(scene.background_color)
    out['ambient-light-color'] = vector2Json(scene.ambientLight)
    out['lights'] = suns2Json(scene.sunDirections, scene.sunColors) + pointlights2Json(scene.pointlightPositions, scene.pointlightColors)
    out['cameras'] = cameras2Json(scene.cameras)
    out['cloud'] = cloud2Json(scene.cloud, cloudpath=cloudpath)
    return out

def json2Scene(data, baseDir, device=None):
    # check data
    scene = Scene(device=device)
    if 'background-color' in data:
        scene.background_color = torch.tensor(data['background-color'], device=device, dtype=torch.float)
    if 'ambient-light-color' in data:
        scene.ambientLight = torch.tensor(data['ambient-light-color'], device=device, dtype=torch.float)
    if 'lights' in data:
        (scene.pointlightPositions, scene.pointlightColors, scene.sunDirections, scene.sunColors) = json2Lights(data, device)
    if 'cameras' in data:
        cam = data['cameras']
        scene.cameras = json2Cameras(cam, device)
    if 'cloud' in data:
        scene.cloud = json2Cloud(data['cloud'], baseDir, device)
    return scene

def json2Lights(data, device=None):
    pointlights = []
    suns = []
    for li, l in enumerate(data['lights']):
        if 'type' not in l:
            print("Warning: no light type given for light " + str(li) + ".")
            continue
        if 'color' not in l:
            print("Warning: No color given for light (" + str(li) + ", " + l['type'])
        if l['type'] == 'sun':
            if 'direction' not in l:
                print("Warning: No direction for light (" + str(li) + ", sun) given.")
                l['direction'] = [0.0,0.0,1.0]
            suns.append(l['direction'] + l['color'])
        elif l['type'] == 'pointlight':
            if 'position' not in l:
                print("Warning: No position given for ligth (" + str(li) + ", sun).")
                l['position'] = [0.0, 0.0, 0.0]
            pointlights.append(l['position'] + l['color'])
        else:
            print("Warning: Skipping unknown light source " + str(l['type']) + ".")
        l['type']
    pointlights = torch.tensor(pointlights, dtype=torch.float, device=device)
    suns = torch.tensor(suns, dtype=torch.float, device=device)
    if suns.size(0) > 0:
        sunDirections, sunColors = torch.chunk(suns, 2, dim=1)
    else:
        sunDirections = sunColors = suns

    if pointlights.size(0) > 0:
        pointlightPositions, pointlightColors = torch.chunk(pointlights, 2, dim=1)
    else:
        pointlightPositions = pointlightColors = pointlights

    return (pointlightPositions, pointlightColors, sunDirections, sunColors)

def suns2Json(sunDirections, sunColors):
    suns = []
    for i in range(sunDirections.shape[0]):
        sun = {}
        sun['type'] = 'sun'
        sun['direction'] = vector2Json(sunDirections[i])
        sun['color'] = vector2Json(sunColors[i])
        suns.append(sun)
    return suns

def pointlights2Json(pointlightPositions, pointlightColors):
    lights = []
    for i in range(pointlightPositions.shape[0]):
        light = {}
        light['type'] = 'pointlight'
        light['position'] = vector2Json(pointlightPositions[i])
        light['color'] = vector2Json(pointlightColors[i])
        lights.append(light)
    return lights

def json2Scalar(s, device=None):
    return torch.tensor([s], dtype=torch.float, device=device)

def scalar2Json(s):
    if type(s) == int or type(s) == float:
        return s
    # tensor
    return s.item()

def json2Vector(v, device=None):
    return torch.tensor(v, dtype=torch.float, device=device)

def vector2Json(v):
    return v.tolist()

def json2Matrix(m, device=None):
    return torch.tensor(m, dtype=torch.float, device=device)

def matrix2Json(m):
    return m.tolist()

def json2Position(pos, device=None):
    if len(pos) != 3:
        print("Position does not have correct number of coordiantes: " + str(pos))
    return json2Vector(pos, device=device)

def json2ScaleMatrix(data, device=None):
    if type(data) is float or type(data) is int:
        return data
    else:
        print("Only accept a float for isotropic scaling.")
        return 1.0


def json2RotationMatrix(data, device=None):
    if type(data) is list and len(data) == 3:
        data = {"type": "rotationMatrix", "matrix": data}
    if "type" in data:
        rt = data["type"]
        if rt == "rotationMatrix":
            for i in range(3):
                if len(data["matrix"][i]) != 3:
                    print("Expected three scalar entries in row of rotation matrix.")
                    return torch.eye(3, device=device)
            return json2Matrix(data["matrix"], device=device)

        if "X" in data:
            X = json2Matrix([data["X"]], device=device)
        if "Y" in data:
            Y = json2Matrix([data["Y"]], device=device)
        if "Z" in data:
            Z = json2Matrix([data["Z"]], device=device)
        S = math.pi/180
        if rt.find("Degree") > -1:
            X = X * S
            Y = Y * S
            Z = Z * S
        if rt.find("EulerXYZ") > -1:
            return rotationMatrixX(X).mm(rotationMatrixY(Y).mm(rotationMatrixZ(Z)))
        if rt.find("EulerXZY") > -1:
            return rotationMatrixX(X).mm(rotationMatrixZ(Z).mm(rotationMatrixY(Y)))
        if rt.find("EulerYXZ") > -1:
            return rotationMatrixY(Y).mm(rotationMatrixX(X).mm(rotationMatrixZ(Z)))
        if rt.find("EulerYZX") > -1:
            return rotationMatrixY(Y).mm(rotationMatrixZ(Z).mm(rotationMatrixX(X)))
        if rt.find("EulerZXY") > -1:
            return rotationMatrixZ(Z).mm(rotationMatrixX(X).mm(rotationMatrixY(Y)))
        if rt.find("EulerZYX") > -1:
            return rotationMatrixZ(Z).mm(rotationMatrixY(Y).mm(rotationMatrixX(X)))
    print("Could not read rotation.")
    return torch.eye(3)

def json2Cameras(cams, device=None):
    ok = True
    cameras = []
    # default is perspective camera
    for li, cam in enumerate(cams):
        if not ('type' in cam):
            cam['type'] = 'pinhole'
        if cam['type'] == 'pinhole':
            camera = PinholeCamera(device=device)
            if 'focalLength' in cam:
                camera.focalLength = torch.tensor(cam['focalLength'], dtype=torch.float, device=device)
            if 'principlePoint' in cam:
                pt = cam['principlePoint']
                if len(pt) != 2:
                    print("Warning: Principle point does not have correct number of coordinates: " + str(pt))
                else:
                    camera.principlePoint = torch.tensor(pt, dtype=torch.float, device=device)
        else:
            print("Warning: Unknown camera type: " + str(cam['type']))
            ok = False
        if ok and 'width' in cam:
            camera.width = cam['width']
            camera.height = cam['width']
            camera.sv = cam['width']
        if ok and 'sv' in cam:
            camera.sv = cam['sv']
        if ok and 'lookAt' in cam:
            try:
                fromP = torch.tensor(cam['lookAt']['from'], dtype=torch.float, device=device).unsqueeze(0)
            except KeyError as identifier:
                fromP = torch.FloatTensor([0, 0, 1], dtype=torch.float).to(device=device).unsqueeze(0)
            try:
                to = torch.tensor(cam['lookAt']['to'], dtype=torch.float, device=device).unsqueeze(0)
            except KeyError as identifier:
                to = torch.zeros(3, dtype=torch.float, device=device)
            try:
                up = torch.tensor(cam['lookAt']['up'], dtype=torch.float, device=device).unsqueeze(0)
            except KeyError as identifier:
                up = torch.FloatTensor([0, 1, 0], dtype=torch.float).to(device=device).unsqueeze(0)
            (camera.rotation, camera.position) = batchLookAt(fromP, to, up)
        elif ok:
            if 'position' in cam:
                if len(cam['position']) == 1:
                    cam['position'] = cam['position'].pop()
                camera.position = json2Position(cam['position'], device=device).unsqueeze(0)
            if 'rotation' in cam:
                if len(cam['rotation']) == 1:
                    cam['rotation'] = cam['rotation'].pop()
                camera.rotation = json2RotationMatrix(cam['rotation'], device=device).unsqueeze(0)
        if 'near' in cam:
            camera.near = cam['near']
        if 'far' in cam:
            camera.far = cam['far']
        cameras.append(camera)
    return cameras

def cameras2Json(cameras):
    _cameras = []
    for i, camera in enumerate(cameras):
        cam = {'type': camera.type}
        if camera.type == 'pinhole':
            cam['focalLength'] = scalar2Json(camera.focalLength)
            #cam['principlePoint'] = vector2Json(camera.principlePoint)
        elif camera.type == 'orthographic':
            cam['scaleX'] = scalar2Json(camera.scaleX)
            cam['scaleY'] = scalar2Json(camera.scaleY)
            cam['shear'] = scalar2Json(camera.shear)
        else:
            print("Warning: Unknown camera type: " + str(camera.type))
        cam['width'] = scalar2Json(camera.width)
        cam['height'] = scalar2Json(camera.height)
        cam['position'] = vector2Json(camera.position)
        cam['rotation'] = matrix2Json(camera.rotation)
        cam['near'] = scalar2Json(camera.near)
        cam['far'] = scalar2Json(camera.far)
        _cameras.append(cam)

    return _cameras

def json2Cloud(data, baseDir, device=None):
    cloud = PointCloud(device)
    if 'shading' in data:
        cloud.shading = data['shading']
    if 'backfaceCulling' in data:
        cloud.backfaceCulling = data['backfaceCulling']
    if 'position' in data:
        cloud.position = json2Position(data['position'], device=device)
    if 'rotation' in data:
        cloud.rotation = json2RotationMatrix(data['rotation'], device=device)
    if 'scale' in data:
        cloud.scale = json2Vector(data['scale'], device=device)

    if 'points' not in data:
        print("Warning: No point data given for cloud")
    elif 'points' in data:
        ## assume it is a path to a file with the data
        cloudPath = os.path.join(baseDir, data['points'])
        #print("Notice: Point cloud path: " + cloudPath)
        points = readCloud(cloudPath, device=device)
        cloud.localPoints = points[:,0:3]
        cloud.localNormals = points[:, 3:6]
        cloud.color = points[:,6:9]
    if 'color' in data:
        cloud.color = json2Scalar(data['color'], device=device)
        if hasattr(cloud, "localPoints"):
            if cloud.color.dim() == 1:
                cloud.color = cloud.color.unsqueeze(0).expand(cloud.localPoints.shape[0])
            elif cloud.color.dim() == 2 and cloud.color.shape[0] == 1:
                cloud.color = cloud.color.expand(cloud.localPoints.shape[0], -1)
            else:
                assert(cloud.color.shape == cloud.localPoints.shape)
    return cloud

def cloud2Json(cloud, cloudpath=None):
    out = {}
    out['shading'] = cloud.shading
    out['backfaceCulling'] = cloud.backfaceCulling
    out['position'] = vector2Json(cloud.position)
    out['rotation'] = matrix2Json(cloud.rotation)
    out['scale'] = scalar2Json(cloud.scale)
    out['color'] = vector2Json(cloud.color)
    if cloudpath is not None:
        out['points'] = cloudpath
    else:
        print("Warning: in-scene pointclouds not supported yet, please provide cloudpath")
    return out


# Returns a torch tensor (x,y,z,nx,ny,nz,r,g,b) (n x d)
# normal is unchanged
# rgb is scaled to [0,1]
def readPlyCloud(fileName, device=None):
    plydata = PlyData.read(fileName)
    verts = plydata["vertex"]
    if verts.count == 0:
        print("Warning: empty file!")
    zeros = [0] * verts.count
    ones = [1] * verts.count
    minusones = [-1] * verts.count
    x = verts["x"]
    y = verts["y"]
    z = verts["z"]
    try:
        nx = verts["nx"]
        ny = verts["ny"]
        nz = verts["nz"]
    except ValueError:
        nx = zeros
        ny = zeros
        nz = ones
    try:
        r = verts["red"] / 255.0
        g = verts["green"] / 255.0
        b = verts["blue"] / 255.0
    except ValueError:
        try:
            r = verts["r"] / 255.0
            g = verts["g"] / 255.0
            b = verts["b"] / 255.0
        except ValueError:
            r = ones
            g = ones
            b = ones
    points = torch.tensor([x,y,z,nx,ny,nz,r,g,b], dtype=torch.float, device=device)
    points = points.transpose(0,1)
    points[:,3:6] = normalize(points[:, 3:6], 1)
    return points

def writePlyCloud(cloudpath, points, normals, colors, text=False):
    verts = np.concatenate((points.detach().cpu().numpy(), normals.detach().cpu().numpy(), (255*colors.clamp(0.0, 1.0)).detach().cpu().numpy()), axis=1)
    verts = [tuple(row) for row in verts]
    elV = PlyElement.describe(np.array(verts, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
    dirP = os.path.dirname(cloudpath)
    os.makedirs(dirP, exist_ok=True)
    PlyData([elV], text=text).write(cloudpath)

def cleanWhiteSpaces(line):
    old = None
    while old != line:
        old = line
        line = line.replace("  ", " ")
    return line

def readOFFCloud(fileName, device=None):
    if not os.path.exists(fileName):
        print("File " + fileName + " not found!")
    with open(fileName, 'r') as f:
        preamble = f.readline()
        if preamble != "NOFF\n":
            print("File " + fileName + " not an OFF file! Preamble: " + preamble)
            return None
        index = f.readline().split(' ')
        V = int(index[0])
        verts = []
        for line in f:
            ln = cleanWhiteSpaces(line).split(' ')
            ln = list(map(lambda x : float(x), ln))
            tpl = [ln[0],ln[1],ln[2],0,0,0,1.0,1.0,1.0]
            if len(ln) == 6:
                tpl[3] = ln[3]
                tpl[4] = ln[4]
                tpl[5] = ln[5]
            verts.append(tpl)
            if len(verts) == V :
                break
        assert(V == len(verts))
        points = torch.tensor(verts, dtype=torch.float, device=device)
        points[:,3:6] = normalize(points[:,3:6], 1)
        return points

def readCloud(fileName, device=None):
    suffix = os.path.splitext(fileName)[1]
    if suffix == '.ply':
        return readPlyCloud(fileName, device=device)
    elif suffix == '.off':
        return readOFFCloud(fileName, device=device)
    print("Extension " + suffix + " of file " + fileName + " unknown.")
    return None

