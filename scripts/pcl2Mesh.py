from plyfile import PlyData, PlyElement
import numpy as np
import math
import torch
import os
import argparse

from neural_point_splatter.geometryConstruction import triangle3dmesh, circle3dmesh, normal2RotMatrix, pcl2Mesh
from neural_point_splatter.splatterIo import readCloud

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a ply holding a point cloud to a ply holding a mesh with circles or radius r.')
    parser.add_argument('input', metavar="input", 
                    help='Input file')
    parser.add_argument('-o', '--output', dest='output', 
            help='Output file')
    parser.add_argument('-r', '--radius', dest='radius', type=float, default="0.005",
            help='radius of circle to replace points with')
    parser.add_argument('-t', '--triangles', dest='triangles', type=int, default=5,
            help='number of triangles to use for approximation')
    parser.add_argument('--text', action='store_true', default=False, 
            help='By default, PLY files are written in binary format. Add this flag to store as text.')
    args = parser.parse_args()
    if(args.output is None):
        filename, file_extension = os.path.splitext(args.input)
        # for the moment only ply files
        args.output = filename + "_mesh" + ".ply"
    
    if not os.path.exists(args.input):
        print("ERROR: Please provide a path to an existing input file")
        parser.parse_args(['-h'])
        exit(-1)
    print("Will write to: " + args.output)
    if args.triangles > 3:
        (circ3, circ3Conn) = circle3dmesh(args.triangles)
    else:
        (circ3, circ3Conn) = triangle3dmesh(args.triangles)
    circ3 = args.radius * circ3
    points = readCloud(args.input).float()
    pointsCount = points.size()[0]
    print("Found " + str(pointsCount) + " points")
   
    (verts, faces) = pcl2Mesh(points[:,0:3], points[:,3:6], points[:,6:9], circ3, circ3Conn)
    verts = verts.numpy()
    faces = faces.numpy()
    verts = list(map(tuple, verts))
    faces = list(map(lambda row : (tuple(row[0:3]), row[3], row[4], row[5]), faces))
    npVerts = np.array(verts, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    npFaces = np.array(faces, dtype=[
        ('vertex_indices', 'i4', (3,)), 
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u2')
        ])
    elV = PlyElement.describe(npVerts, 'vertex')
    elF = PlyElement.describe(npFaces, 'face')
    print("Created vertices: " + str(len(verts)) + ", expected: " + str(pointsCount*(args.triangles+1)))
    print("Created triangles: " + str(len(faces)) + ", expected: " + str(pointsCount*args.triangles))
    PlyData([elV, elF], text=args.text).write(args.output)

