import os
from PIL import Image
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from numpy.linalg import norm
import scipy.sparse as sp
import scipy.optimize
import cv2
import openmesh as om

import chumpy as ch
from chumpy import optimization
from chumpy.utils import row, col
from opendr.renderer import ColoredRenderer, TexturedRenderer
from opendr.lighting import LambertianPointLight, SphericalHarmonics
from opendr.serialization import load_mesh
from opendr.geometry import VertNormals, Rodrigues
from opendr.camera import ProjectPoints
from opendr.filters import gaussian_pyramid

save_dir = os.path.join("learn_examples", "opendr_teapot")
os.makedirs(save_dir, exist_ok=True)
w, h = 256, 256


def get_earthmesh(trans, rotation):
    fname = "example_data/nasa_earth.obj"
    mesh = load_mesh(fname)

    if not hasattr(get_earthmesh, "mesh"):
        mesh.v = np.asarray(mesh.v, order='C')
        mesh.vc = mesh.v*0 + 1
        mesh.v -= row(np.mean(mesh.v, axis=0))
        mesh.v /= np.max(mesh.v)
        mesh.v *= 2.0
        get_earthmesh.mesh = mesh

    mesh = deepcopy(get_earthmesh.mesh)
    mesh.v = mesh.v.dot(cv2.Rodrigues(
        np.asarray(np.array(rotation), np.float64))[0])
    mesh.v = mesh.v + row(np.asarray(trans))
    return mesh


def read_and_process_mesh(fname, trans, rotation):
    mesh = load_mesh(fname)
    mesh.v = np.asarray(mesh.v, order='C')
    mesh.vc = np.ones_like(mesh.v)
    mesh.v -= row(np.mean(mesh.v, axis=0))
    mesh.v /= np.max(mesh.v)
    mesh.v *= 2.0
    mesh.v = mesh.v.dot(cv2.Rodrigues(
        np.asarray(np.array(rotation), np.float64))[0])
    if hasattr(mesh, "vn"):
        mesh.vn = mesh.vn.dot(cv2.Rodrigues(
            np.asarray(np.array(rotation), np.float64))[0])
    mesh.v = mesh.v + row(np.asarray(trans))
    return mesh


def create_callback(f, step=0):
    def cb(_):
        nonlocal step
        step = step+1
        Image.fromarray((f.r * 255).astype(np.uint8)
                        ).save(os.path.join(save_dir, "step_{:05d}.png".format(step)))
    return cb


def test_earth():
    m = get_earthmesh(trans=ch.array([0, 0, 0]), rotation=ch.zeros(3))
    # Create V, A, U, f: geometry, brightness, camera, renderer
    V = ch.array(m.v)
    A = SphericalHarmonics(vn=VertNormals(v=V, f=m.f),
                           components=[3., 2., 0., 0., 0., 0., 0., 0., 0.],
                           light_color=ch.ones(3))
    # camera
    U = ProjectPoints(v=V, f=[w, w], c=[w/2., h/2.], k=ch.zeros(5),
                      t=ch.zeros(3), rt=ch.zeros(3))
    f = TexturedRenderer(vc=A, camera=U, f=m.f, bgcolor=[0., 0., 0.],
                         texture_image=m.texture_image, vt=m.vt, ft=m.ft,
                         frustum={'width': w, 'height': h, 'near': 1, 'far': 20})

    # Parameterize the vertices
    translation, rotation = ch.array([0, 0, 8]), ch.zeros(3)
    f.v = translation + V.dot(Rodrigues(rotation))

    observed = f.r
    np.random.seed(1)
    # this is reactive
    # in the sense that changes to values will affect function which depend on them.
    translation[:] = translation.r + np.random.rand(3)
    rotation[:] = rotation.r + np.random.rand(3) * .2
    # Create the energy
    E_raw = f - observed
    E_pyr = gaussian_pyramid(E_raw, n_levels=6, normalization='size')

    Image.fromarray((observed * 255).astype(np.uint8)
                    ).save(os.path.join(save_dir, "reference.png"))
    step = 0
    Image.fromarray((f.r * 255).astype(np.uint8)
                    ).save(os.path.join(save_dir, "step_{:05d}.png".format(step)))

    print('OPTIMIZING TRANSLATION, ROTATION, AND LIGHT PARMS')
    free_variables = [translation, rotation]
    ch.minimize({'pyr': E_pyr}, x0=free_variables, callback=create_callback(f))
    ch.minimize({'raw': E_raw}, x0=free_variables, callback=create_callback(f))


# test_earth()
def test_teapot():
    # load teapot and sphere
    reference = read_and_process_mesh(
        "example_data/pointclouds/teapot_mesh.obj",
        trans=ch.array([0, 0, 0]), rotation=ch.array([np.pi, 0, 0]))
    target = read_and_process_mesh(
        "example_data/pointclouds/sphere_normal_2K_mesh.obj",
        trans=ch.array([0, 0, 0]), rotation=ch.array([0, 0, 0]))

    # reference
    V_ref = ch.array(reference.v)
    vc_ref = ch.array(reference.vc)
    A_ref = LambertianPointLight(v=V_ref, f=reference.f, num_verts=len(
        V_ref), light_pos=ch.array([-1000, -1000, -1000]), vc=vc_ref,
        light_color=ch.array([0.9, 0, 0])) +\
        LambertianPointLight(v=V_ref, f=reference.f, num_verts=len(
            V_ref), light_pos=ch.array([1000, -1000, -1000]), vc=vc_ref,
        light_color=ch.array([0.0, 0.9, 0])) +\
        LambertianPointLight(v=V_ref, f=reference.f, num_verts=len(
            V_ref), light_pos=ch.array([-1000, 1000, -1000]), vc=vc_ref,
        light_color=ch.array([0.0, 0.0, 0.9]))
    U_ref = ProjectPoints(v=V_ref, f=[w, w], c=[w/2., h/2.], k=ch.zeros(5),
                          t=ch.zeros(3), rt=ch.zeros(3))
    f_ref = ColoredRenderer(vc=A_ref, camera=U_ref, f=reference.f, bgcolor=[1.0, 1.0, 1.0],
                            frustum={'width': w, 'height': h, 'near': 1, 'far': 20})

    # target
    V_tgt = ch.array(target.v)
    vc_tgt = ch.array(target.vc)
    A_tgt = LambertianPointLight(v=V_tgt, f=target.f, num_verts=len(
        V_tgt), light_pos=ch.array([-1000, -1000, -1000]), vc=vc_tgt,
        light_color=ch.array([0.9, 0, 0])) +\
        LambertianPointLight(v=V_tgt, f=target.f, num_verts=len(
            V_tgt), light_pos=ch.array([1000, -1000, -1000]), vc=vc_tgt,
        light_color=ch.array([0.0, 0.9, 0])) +\
        LambertianPointLight(v=V_tgt, f=target.f, num_verts=len(
            V_tgt), light_pos=ch.array([-1000, 1000, -1000]), vc=vc_tgt,
        light_color=ch.array([0.0, 0.0, 0.9]))
    U_tgt = ProjectPoints(v=V_tgt, f=[w, w], c=[w/2., h/2.], k=ch.zeros(5),
                          t=ch.zeros(3), rt=ch.zeros(3))
    f_tgt = ColoredRenderer(vc=A_tgt, camera=U_tgt, f=target.f, bgcolor=[1.0, 1.0, 1.0],
                            frustum={'width': w, 'height': h, 'near': 1, 'far': 20})
    # offset = ch.zeros(V_tgt.shape)
    translation, rotation = ch.array([0, 0, 6]), ch.zeros(3)
    f_tgt.v = translation + V_tgt.dot(Rodrigues(rotation))
    f_ref.v = translation + V_ref.dot(Rodrigues(rotation))

    op_mesh_target = om.read_trimesh(
        "example_data/pointclouds/sphere_normal_2K_mesh.obj")

    n_rotations = 144

    # camera positions
    for index in range(n_rotations):
        rotation[:] = np.random.rand(3)*np.pi*2
        np.save(os.path.join(save_dir, "rot_v{:03d}".format(index)), rotation)
        img_ref = f_ref.r
        Image.fromarray((img_ref * 255).astype(np.uint8)
                        ).save(os.path.join(save_dir, "reference_v{:03d}.png".format(index)))
        img_tgt = f_tgt.r
        Image.fromarray((img_tgt * 255).astype(np.uint8)
                        ).save(os.path.join(save_dir, "target_v{:03d}.png".format(index)))

        E_raw = f_tgt - img_ref
        # E_pyr = gaussian_pyramid(E_raw, n_levels=6, normalization='size')
        free_variables = [V_tgt]
        # dogleg
        # Newton-CG
        # SLSQP
        # BFGS
        # trust-ncg
        method = "trust-ncg"
        maxiter = 30
        ch.minimize({'pyr': E_raw}, x0=free_variables, method=method, options=dict(maxiter=30),
                    callback=create_callback(f_tgt, step=index*maxiter))
        ch.minimize({'pyr': E_raw}, x0=free_variables, method=method, options=dict(maxiter=30),
                    callback=create_callback(f_tgt, step=index*maxiter))
        # is not the same?
        target.v = f_tgt.v.r.copy()
        # save mesh
        # mesh = pymesh.form_mesh(f_tgt.v.r, f_tgt.f)
        # pymesh.save_mesh(os.path.join(
        #     save_dir, "target_v{:03d}.obj".format(index)), mesh)
        point_array = op_mesh_target.points()
        point_array[:] = target.v
        np.copyto(op_mesh_target.points(), f_tgt.v.r)
        om.write_mesh(os.path.join(
            save_dir, "target_v{:03d}.obj".format(index)), op_mesh_target)


# rot_vector = np.load("learn_examples/opendr_teapot/rot_v143.npy")
rot = ch.array(np.deg2rad(np.array([-17, -11, -3])))
translation = np.array([0, 0, 5])
# mesh = read_and_process_mesh(
#     "example_data/pointclouds/sphere_normal_2K_mesh.obj",
#     trans=np.zeros(3), rotation=-rot_vector)
# mesh = read_and_process_mesh(
#     "learn_examples/opendr_teapot/target_v143.obj", trans=np.zeros(3), rotation=-rot_vector)
mesh = read_and_process_mesh(
    "example_data/pointclouds/teapot_mesh.obj",
    trans=ch.array([0, 0, 0]), rotation=ch.array([np.pi, 0, 0]))
V_ref = ch.array(mesh.v)
# reference
A_ref = LambertianPointLight(v=V_ref, f=mesh.f, num_verts=len(V_ref), light_pos=ch.array([-1000, -1000, -1000]), vc=mesh.vc,
                             light_color=ch.array([0.9, 0, 0])) +\
    LambertianPointLight(v=V_ref, f=mesh.f, num_verts=len(V_ref), light_pos=ch.array([1000, -1000, -1000]), vc=mesh.vc,
                         light_color=ch.array([0.0, 0.9, 0])) +\
    LambertianPointLight(v=V_ref, f=mesh.f, num_verts=len(V_ref), light_pos=ch.array(
        [-1000, 1000, -1000]), vc=mesh.vc, light_color=ch.array([0.0, 0.0, 0.9]))
U_ref = ProjectPoints(v=V_ref, f=[w, w], c=[w/2., h/2.], k=ch.zeros(5),
                      t=ch.zeros(3), rt=ch.zeros(3))
f_ref = ColoredRenderer(vc=A_ref, camera=U_ref, f=mesh.f, bgcolor=[1.0, 1.0, 1.0],
                        frustum={'width': w, 'height': h, 'near': 1, 'far': 20})
f_ref.v = translation + V_ref.dot(Rodrigues(rot))
Image.fromarray((f_ref.r * 255).astype(np.uint8)
                ).save(os.path.join(save_dir, "opendr_ref.png"))
