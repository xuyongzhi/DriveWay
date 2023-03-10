import cv2
import laspy
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path

def points_to_density_img(vertices, width, height):
    xy_density, scale_3d_2d, mins_3d = get_density(vertices, width, height)
    density_img = draw_density_image(xy_density)
    density_img_eh = cv2.equalizeHist(density_img)
    return density_img, density_img_eh, scale_3d_2d, mins_3d

def get_density(points, width=512, height=512):
    imageSizes = np.array([width, height]).reshape((-1, 2))
    if 1:
        points = points[:,:2]
        mins = points.min(0, keepdims=True)
        maxs = points.max(0, keepdims=True)
        maxRange = (maxs - mins)[:, :2].max()
        padding = maxRange * 0.05
        mins = (maxs + mins) / 2 - maxRange / 2
        mins -= padding
        maxRange += padding * 2
        scale_3d_2d = 1 / maxRange * imageSizes
        coordinates = np.round((points - mins) * scale_3d_2d).astype(np.int32)
    else:
        coordinates = np.round(points[:, :2] * imageSizes).astype(np.int32)
        coordinates = np.minimum(np.maximum(coordinates, 0), imageSizes - 1)
    density = np.zeros((height, width))
    for i,uv in enumerate(coordinates):
        density[uv[1], width-uv[0]] += 1
    return density, scale_3d_2d, mins

def draw_density_image(density):
    ref = density.mean() * 0.5
    #ref = density.max() * 0.7
    densityImage = np.minimum(np.round(density / ref * 255).astype(np.uint8), 255)
    #if nChannels == 3:
    #    densityImage = np.stack([densityImage, densityImage, densityImage], axis=2)
    return densityImage

def read_laz(laz_path):
    points_all = []
    with laspy.open(str(laz_path)) as input_las:
        num_points = input_las.header.point_count
        print(f"Number of points:   {num_points}")
        print(input_las.header.point_format.dimension_names)
        for points in tqdm(input_las.chunk_iterator(2000000)):
            points = [ (p[0], p[1], p[2], p[3]) for p in points.array ]
            points = np.array(points) / 1000.0
            points_all.append(points)
    points_all = np.concatenate(points_all, 0)
    return points_all[:,:3]

def vis_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5, origin=[5,5,0])
    o3d.visualization.draw_geometries([pcd, mesh_frame])

def erosion(src, kernel_size, num_ite):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8) 
    eroded = cv2.erode(src, kernel, iterations=num_ite)
    return eroded

def dilate(src, kernel_size, num_ite):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8) 
    dilated = cv2.dilate(src, kernel, iterations=num_ite)
    return dilated