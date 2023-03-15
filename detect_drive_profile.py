import open3d as o3d
import cv2
import os
import argparse
import logging
import numpy as np
from pathlib import Path
from timer import Timer
from utils import points_to_density_img, read_laz, erosion, dilate, vis_points, line_sampling, cal_width
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class ProfileDetect:
    '''
    Task: 
        Detect 2D profile from 3D point cloud.
    Assumption: 
        The point cloud has already been aligned with gravity. 
    Method:
        First, project the 3D point cloud into BEV density image. 
        Second, detect the contour from the BEV density image.
        Third, if necessary, the 2D contour can be lifted into 3D.
    Args:
        laz_path: path of input data
        out_dir: output directory
        sampling_ratio: Small sampling ratio enable fast detection but may result in inaccurate result if it is too small. 
        out_height:
        out_width: Larger output image size results in better result, but costs more time.
    Performance:
        The detection taks about 9 seconds on a Macbook Pro with sampling ratio of 0.1 and image size of (2048,2048). The current version can be significantly improved in efficiency.
        The criterion for accuracy is not well defined yet.
    '''
    out_height = 1024*2
    out_width = 1024*2
    sampling_ratio = 0.5
    cen_height = 0

    def __init__(self, 
                 laz_path: str, 
                 out_dir = None, 
                 sampling_ratio = 0.1,
                 out_height = 2048,
                 out_width = 2048,
                 debug = False
                 ):
        assert laz_path.exists(), f'{laz_path} not exist'

        self.laz_path = laz_path
        self.ply_path = laz_path.parent / 'pcd.ply'
        self.sampling_ratio = sampling_ratio
        self.out_height = out_height
        self.out_width = out_width
        self.debug = debug

        if out_dir is None:
            out_dir = Path('Out')
        if not out_dir.exists():
            out_dir.mkdir()
        self.out_dir = out_dir
        self.out_profile_3d_path = out_dir / 'out_profile_3d.ply'

        self.init_logger()
    
    def init_logger(self):
        log_path = self.out_dir/'profile_detection.log'
        if log_path.exists():
            os.remove(log_path)
        logging.basicConfig(filename=str(log_path), level=logging.INFO)
        self.logger = logging.getLogger()
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info(self.config)

    @property 
    def config(self):
        s = '\n' + ' config '.center(60,'-') + '\n'
        s += f'Input point cloud: {str(self.laz_path)} \n'
        s += f'Output directory: {str(self.out_dir)} \n'
        s += f'Sampling ratio: {self.sampling_ratio} \n'
        s += f'Output image size: {self.out_height, self.out_width} \n'
        s += f'Debug: {self.debug} \n'
        s += ''.center(60, '-')
        return s
    
    def timing(fn):
        def ware(self, *args, **kwargs):
            timer  = Timer()
            res = fn(self, *args, **kwargs)
            t = timer.since_start()
            self.logger.info(fn.__name__ + f': {t:.3f} s')
            return res
        return ware
    
    @timing 
    def load_pcd(self,):
        '''
        In order to accelerate the data loading in debuging stage, the point cloud will be saved as a ply while laz file loaded.
        '''
        if self.ply_path.exists():
            pcd =  o3d.io.read_point_cloud(str(self.ply_path))
        else:
            self.logger.info('Reading' + str(self.laz_path)) 
            points = read_laz(self.laz_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(str(self.ply_path), pcd)

        if self.sampling_ratio < 1:
            pcd = pcd.random_down_sample(self.sampling_ratio)
        
        if 0:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=10, origin=[5,5,0])
            o3d.visualization.draw_geometries([pcd, mesh_frame])
        pcd =  ProfileDetect.align_scene(pcd)
        self.raw_pcd = pcd
        self.raw_points = np.asarray(pcd.points)
    
    @staticmethod
    def align_scene(pcd, reverse=False):
        angle = np.pi/6 + np.pi
        if reverse:
            angle *= -1
        R = pcd.get_rotation_matrix_from_xyz((0,0,angle))
        pcd.rotate(R, center=(0,0,0))
        return pcd
    
    @timing 
    def detect(self):
        self.load_pcd()
        profile_2d, scale_3d_2d, mins_3d, img_profile_2d, profile_lines = self.detect_2D_contour()
        self.curve_fitting_2d(profile_lines)
        profile_3d = self.transform_2d_to_3d(profile_2d, scale_3d_2d, mins_3d, 4) 
        o3d.io.write_point_cloud(str(self.out_profile_3d_path), profile_3d)

        if 0:
            vis_pcd = [self.raw_pcd, profile_3d]
            o3d.visualization.draw_geometries(vis_pcd)

    @timing 
    def load_and_curve_fitting(self):
        profile_lines = np.loadtxt(self.out_dir/'profile_lines.txt')
        self.curve_fitting_2d(profile_lines)

    @timing 
    def detect_2D_contour(self):
        self.logger.info('Start detecting BEV contour')
        density_img, density_img_eh, scale_3d_2d, mins_3d = points_to_density_img(self.raw_points, self.out_width, self.out_height)
        if self.debug:
            cv2.imwrite(str(self.out_dir/'density.png'), density_img)
            cv2.imwrite(str(self.out_dir/'density_eh.png'), density_img_eh)

        img = density_img_eh

        img = cv2.GaussianBlur(img, (7, 7), 0)
        if self.debug:
            cv2.imwrite(str(self.out_dir/'GaussianBlur.png'), img)

        ret, thresh = cv2.threshold(img, 2, 255, 0)
        if self.debug:
            cv2.imwrite(str(self.out_dir/'thresh.png'), thresh)

        thresh = erosion(thresh, 5, 2)
        if self.debug:
            cv2.imwrite(str(self.out_dir/'eroded.png'), thresh)

        thresh = dilate(thresh, 5, 2)
        if self.debug:
            cv2.imwrite(str(self.out_dir/'dilated.png'), thresh)
        profile_lines = self.lsd_det(thresh)

        contours_2d, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = [cv2.contourArea(c) for c in contours_2d]
        profile_2d = contours_2d[np.argmax(contour_areas)].squeeze(1)

        img_profile_2d = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) *0
        cv2.drawContours(img_profile_2d, [profile_2d], -1, (0,255,0), 2)
        cv2.imwrite(str(self.out_dir/'out_drive_profile_2d.png'), img_profile_2d)
        np.savetxt(self.out_dir/'profile_2d.txt', profile_2d, fmt='%.5f')

        self.logger.info('Finish detecting BEV contour')
        return profile_2d, scale_3d_2d, mins_3d, img_profile_2d, profile_lines
    
    def curve_fitting_2d(self, profile_lines):
        from curve_fitting import MultiPolyFitting

        profile_2d = line_sampling(profile_lines, 1)
        mul_ply_fit = MultiPolyFitting()
        coeffs_ls = mul_ply_fit.fit(profile_2d)
        
        X = profile_2d[:,0].copy()
        X.sort()
        Y_0 = np.poly1d(coeffs_ls[0])(X.ravel())
        Y_1 = np.poly1d(coeffs_ls[1])(X.ravel())
        Y = (Y_0 + Y_1)/2 # center_curve

        #boundary_0 = np.stack([X,Y_0], axis=1)
        #boundary_1 = np.stack([X,Y_0], axis=1)
        #center_curve = np.stack([X,Y], axis=1)
        #fitted_curves = dict(center=center_curve, boundary_0=boundary_0, boundary_1=boundary_1)

        #width_2d = cal_width(coeffs_ls[0], coeffs_ls[1], X[len(X)//2])

        if 1:
            plt.figure()
            plt.plot(profile_2d[:,0], profile_2d[:,1], 'b.', label='input samples')
            plt.plot(X, Y_0, 'g-', label='boundary 0')
            plt.plot(X, Y_1, 'g-', label='boundary 1')
            plt.plot(X, Y, 'r-', label='center line')
            plt.savefig(str(self.out_dir/'fitted_curves.png'))

    def lsd_det(self, thresh_img):
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
        #gray = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2GRAY)
        lines, _, _, _ = lsd.detect(thresh_img)
        lengths = np.linalg.norm( lines[:,0,2:] - lines[:,0,:2], axis=1 )
        mask = lengths > 20
        lines = lines[mask]
        drawn_img = lsd.drawSegments(thresh_img*0,lines)
        cv2.imwrite(str(self.out_dir/'profile_lines.png'), drawn_img)
        lines = lines.squeeze(1)
        np.savetxt(str(self.out_dir/'profile_lines.txt'), lines)
        return lines

    def transform_2d_to_3d(self, contour_2d, scale_3d_2d, mins_3d, fixed_height):
        contour_2d = contour_2d.copy()
        contour_2d[:,0] = self.out_width - contour_2d[:,0]
        contour_2d = ProfileDetect.get_dense_contour(contour_2d)
        contour_3d = contour_2d / scale_3d_2d + mins_3d
        n = len(contour_3d)
        contour_3d = np.concatenate([contour_3d, np.zeros((n,1))+fixed_height], axis=1)
        pcd = o3d.geometry.PointCloud() 
        pcd.points = o3d.utility.Vector3dVector(contour_3d)
        colors = np.zeros_like(contour_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    
    @staticmethod
    def get_dense_contour(contour_2d, step=2):
        n = len(contour_2d)
        contour_dense = []
        for i in range(1,n):
            offset_i = contour_2d[i] - contour_2d[i-1]
            dis_i = np.linalg.norm(offset_i)
            dir_i = offset_i / dis_i
            k = int(dis_i / step)
            for j in range(k):
                contour_dense.append(contour_2d[i-1] + dir_i * j)
        contour_dense = np.stack(contour_dense)
        contour_dense = np.concatenate([contour_dense, contour_2d], axis=0)
        return contour_dense

def main_test(args):
    drive_det = ProfileDetect(args.input_laz_path,
                            args.output_dir,
                            args.sampling_ratio,
                            args.out_height,
                            args.out_weight,
                            args.debug)
    #drive_det.detect()
    drive_det.load_and_curve_fitting()

if __name__ == '__main__':
    data_dir = Path(__file__).parent / 'Data'
    default_laz_path = data_dir / '07nxs_xct_Output_laz1_4.laz'

    parser = argparse.ArgumentParser('detect 2D horizontal profile from 3D point cloud')
    parser.add_argument('--input_laz_path', 
                        type=str, 
                        default=default_laz_path, 
                        help='The input path of lidar point cloud in format of laz.')
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='The output directory.')
    parser.add_argument('--sampling_ratio',
                        type=float,
                        default=0.2,
                        help='The sampling ratio for point cloud preprocessing.')
    parser.add_argument('--out_height',
                        type=int,
                        default=2048)
    parser.add_argument('--out_weight',
                        type=int,
                        default=2048)
    parser.add_argument('--debug',
                        type=bool,
                        default=True)
    
    args = parser.parse_args()
    main_test(args)