import open3d as o3d
import cv2
import laspy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import points_to_density_img, read_laz

class DriveDetect:
    width = 1024*2
    height = 1024*2
    sample_ratio = 1.0
    out_dir = Path('Out')
    if not out_dir.exists():
        out_dir.mkdir()

    def __init__(self, laz_path) -> None:
        assert laz_path.exists(), f'{laz_path} not exist'
        self.laz_path = laz_path
        self.data_dir = laz_path.parent
        self.ply_path = self.data_dir / 'pcd.ply'

        self.load_pcd()

    def load_pcd(self,):
        if self.ply_path.exists():
            pcd =  o3d.io.read_point_cloud(str(self.ply_path))
        else:
            print('Reading', str(self.laz_path)) 
            points = read_laz(self.laz_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(str(self.ply_path), pcd)

        pcd = pcd.random_down_sample(self.sample_ratio)
        
        if 0:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=6, origin=[0,0,5])
            o3d.visualization.draw_geometries([pcd, mesh_frame])
        pcd =  self.transform(pcd)
        self.points = np.asarray(pcd.points)
    
    def transform(self, pcd):
        R = pcd.get_rotation_matrix_from_xyz((0,0,np.pi/6 + np.pi))
        pcd.rotate(R, center=(0,0,0))
        return pcd
    
    def detect(self):
        density_img, density_img_eh, scale_3d_2d, mins_3d = points_to_density_img(self.points, self.width, self.height)
        cv2.imwrite(str(self.out_dir/'density.png'), density_img)
        cv2.imwrite(str(self.out_dir/'density_eh.png'), density_img_eh)

        img = density_img_eh

        img = cv2.GaussianBlur(img, (11, 11), 0)
        cv2.imwrite(str(self.out_dir/'GaussianBlur.png'), img)

        ret, thresh = cv2.threshold(img, 1, 255, 0)
        cv2.imwrite(str(self.out_dir/'thresh.png'), thresh)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [ c for c in contours if cv2.contourArea(c) > 10000]
        img_contour = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) *0
        cv2.drawContours(img_contour, contours, -1, (0,255,0), 3)
        cv2.imwrite(str(self.out_dir/'contour.png'), img_contour)

        if 0:
            edges = cv2.Canny(density_img, 100, 200)
            cv2.imwrite(str(self.out_dir/'canny.png'), edges)

        print('Detection finished')

    def points_to_img(self, points):

        pass

def vis_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

def main_test():
    data_dir = Path(__file__).parent / 'Data'
    laz_path = data_dir / '07nxs_xct_Output_laz1_4.laz'
    #laz_path = data_dir / '07nxs_xct_Output_laz1_4.copc.laz'
    drive_det = DriveDetect(laz_path)
    drive_det.detect()
    print('Finished')

if __name__ == '__main__':
    main_test()