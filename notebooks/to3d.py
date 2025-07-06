import os, sys
import cv2
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from scipy.spatial import Delaunay

import numpy as np
from scipy import ndimage
import pickle

# Sample values from an image using interpolation
def sample_image(image, positions):
    # Get the image dimensions
    height, width = image.shape[:2]

    values = []
    for i in range(image.shape[-1]):
        # Perform interpolation sampling with linear interpolation (order=1)
        sampled_values = ndimage.map_coordinates(image[:,:,i], positions, order=1, mode='nearest')
        values.append(sampled_values)

    values = np.stack(values, axis=-1)
    return values

def crop_img(image):
    H, W = image.shape[:2]

    H = min(H, W)
    image = image[:H, :H]
    return image


def map_to_pixel(pattern_v, image, scale=1):
    W = image.shape[0]
    pos = pattern_v[:,:2]*scale
    pos = (pos+1)/2*(W-1)
    return pos


def divide(mesh, pattern):
    mesh_reduce_f = trimesh.Trimesh(mesh.vertices, mesh.faces[:len(pattern.faces)], validate=False, process=False)
    v, f = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    v, f = trimesh.remesh.subdivide(v, f)
    #v, f = trimesh.remesh.subdivide_loop(mesh.vertices, mesh.faces, iterations=2)
    mesh_new = trimesh.Trimesh(v, f, validate=False, process=False)
    
    v_barycentric, closest_face_idx = barycentric_faces(mesh_new, mesh_reduce_f)

    triangles = pattern.vertices[pattern.faces[closest_face_idx]]
    pattern_v_new = (triangles * v_barycentric[:, :, None]).sum(axis=-2)

    return mesh_new, pattern_v_new

def barycentric_faces(mesh_query, mesh_base):
    v_query = mesh_query.vertices
    base = trimesh.proximity.ProximityQuery(mesh_base)
    closest_pt, _, closest_face_idx = base.on_surface(v_query)
    triangles = mesh_base.triangles[closest_face_idx]
    v_barycentric = trimesh.triangles.points_to_barycentric(triangles, closest_pt)
    return v_barycentric, closest_face_idx


def select_boundary(mesh):
    unique_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    idx_boundary_v = np.unique(unique_edges.flatten())
    return idx_boundary_v, unique_edges

def get_contour(image, pattern, scale):
    v, f = trimesh.remesh.subdivide(pattern.vertices, pattern.faces)
    v, f = trimesh.remesh.subdivide(v, f)
    mesh_new = trimesh.Trimesh(v, f, validate=False, process=False)

    idx_boundary_v, _ = select_boundary(mesh_new)
    boundary_v = v[idx_boundary_v,:2]*scale

    W = image.shape[0]
    image_new = cv2.resize(image, (W//2, W//2))

    W = image_new.shape[0]
    boundary_v = (boundary_v+1)/2*(W-1)
    boundary_v = boundary_v.astype(int)
    boundary_v = np.clip(boundary_v, 2, W-3)

    image_new[boundary_v[:,0], boundary_v[:,1]] = 0
    image_new[boundary_v[:,0]-1, boundary_v[:,1]-1] = 0
    image_new[boundary_v[:,0]+1, boundary_v[:,1]+1] = 0
    image_new[boundary_v[:,0]+1, boundary_v[:,1]-1] = 0
    image_new[boundary_v[:,0]-1, boundary_v[:,1]+1] = 0
    image_new[boundary_v[:,0]+1, boundary_v[:,1]] = 0
    image_new[boundary_v[:,0]-1, boundary_v[:,1]] = 0
    image_new[boundary_v[:,0], boundary_v[:,1]-1] = 0
    image_new[boundary_v[:,0], boundary_v[:,1]+1] = 0
    image_new = image_new[:,:,::-1]
    return image_new



