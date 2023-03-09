import open3d as o3d
import cv2
import laspy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import points_to_density_img, read_laz

class DriveDetect:
    width = 1024
    height = 1024

    def __init__(self, laz_path) -> None:
        assert laz_path.exists(), f'{laz_path} not exist'
        self.laz_path = laz_path
        self.data_dir = laz_path.parent
        self.ply_path = self.data_dir / 'pcd.ply'
        self.ply_10_downsampled_path = self.data_dir / 'pcd_10_downsampled.ply'
        self.ply_100_downsampled_path = self.data_dir / 'pcd_100_downsampled.ply'

        self.load_laz()

    def load_laz(self,):
        if self.load_pcd():
            return
       
        print('Reading', str(self.laz_path)) 
        points = read_laz(self.laz_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])

        pcd_10ds = pcd.voxel_down_sample(10)
        pcd_100ds = pcd.voxel_down_sample(100)
        o3d.io.write_point_cloud(str(self.ply_path), pcd)
        o3d.io.write_point_cloud(str(self.ply_10_downsampled_path), pcd_10ds)
        o3d.io.write_point_cloud(str(self.ply_100_downsampled_path), pcd_100ds)
        
        if 0:
            o3d.visualization.draw_geometries([pcd])
    
    def load_pcd(self):
        pcd_path = self.ply_100_downsampled_path
        if not pcd_path.exists():
            return False
        
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=6, origin=[0,0,5])
        if 1:
            o3d.visualization.draw_geometries([pcd, mesh_frame])

        self.pcd = pcd
        self.points = np.asarray(pcd.points)
        return True

    def detect(self):
        points = self.points
        #img = points_to_density_img(points, self.width, self.height)

        print()

    def points_to_img(self, points):

        pass

def vis_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

def main_test():
    data_dir = Path(__file__).parent / 'Data'
    laz_path = data_dir / '07nxs_xct_Output_laz1_4.laz'
    laz_path = data_dir / '07nxs_xct_Output_laz1_4.copc.laz'
    drive_det = DriveDetect(laz_path)
    drive_det.detect()
    print('Finished')

if __name__ == '__main__':
    main_test()