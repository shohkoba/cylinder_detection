import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import time
from mpl_toolkits.mplot3d import Axes3D

sns.set()


class CylinderDetectior():
    def __init__(self, pcd):
        downpcd = pcd.voxel_down_sample(voxel_size=0.05)
        downpcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        normals = np.array(downpcd.normals)
        indexs = np.where(abs(normals[:, 1]) < np.sin(np.deg2rad(5)))[0]

        pcd_part = downpcd.select_by_index(indexs)

        points = np.array(pcd_part.points)
        self.points = np.delete(points, obj=1, axis=1)
        normals = np.array(pcd_part.normals)
        self.normals = np.delete(normals, obj=1, axis=1)
        norms = np.tile(np.linalg.norm(self.normals, axis=1), (2, 1))
        norms = norms.transpose((1, 0))
        self.normals /= norms

    def intersection(self, points1, points2, normals1, normals2):
        A = np.stack([normals1, -normals2])
        A = A.transpose((1, 2, 0))
        A_inv = np.linalg.pinv(A)
        diff = points2 - points1
        diff = diff[:, :, np.newaxis]
        param = np.squeeze(np.matmul(A_inv, diff))
        param_reshape = np.stack([param[:, 0], param[:, 0]]).transpose((1, 0))
        ans = points1 + param_reshape * normals1
        return ans

    def test_intersection(self, num=3):
        idx1 = np.random.choice(len(self.points), size=num)
        idx2 = np.random.choice(len(self.points), size=num)
        centers = self.intersection(
            self.points[idx1],
            self.points[idx2],
            self.normals[idx1],
            self.normals[idx2])

        fig, ax = plt.subplots(figsize=(10, 10))
        cm = plt.get_cmap('tab20')
        for i in range(num):
            color = cm(int(i / num * cm.N))
            ax.scatter(centers[i][0], centers[i][1], s=50, color=color)
            ax.scatter([self.points[idx1[i]][0], self.points[idx2[i]][0]],
                       [self.points[idx1[i]][1], self.points[idx2[i]][1]], s=20, color=color)
            ax.plot([self.points[idx1[i]][0], self.points[idx1[i]][0] + self.normals[idx1[i]][0]],
                    [self.points[idx1[i]][1], self.points[idx1[i]][1] + self.normals[idx1[i]][1]], color=color)
            ax.plot([self.points[idx2[i]][0], self.points[idx2[i]][0] + self.normals[idx2[i]][0]],
                    [self.points[idx2[i]][1], self.points[idx2[i]][1] + self.normals[idx2[i]][1]], color=color)
        ax.set_aspect('equal')
        ax.set_title('intersecton test')
        fig.savefig('test_intersection.png')

    def distance(self, point1, point2, axis=1):
        return np.linalg.norm(point1 - point2, axis=axis)

    def count_circle_point_num(self, centers, radiuses, diff=0.01):
        points_tile = np.tile(self.points, (centers.shape[0], 1, 1))
        centers_tile = np.tile(centers, (self.points.shape[0], 1, 1))
        centers_tile = centers_tile.transpose((1, 0, 2))
        radiuses_tile = np.tile(radiuses, (self.points.shape[0], 1))
        radiuses_tile = radiuses_tile.transpose((1, 0))
        d = self.distance(points_tile, centers_tile, axis=2)
        count_edge = np.count_nonzero(abs(radiuses_tile - d) < diff, axis=1)
        count_body = np.count_nonzero(d < radiuses_tile - diff, axis=1)
        return count_edge, count_body

    def get_centers(self, r_min=0.1, r_max=0.3):
        rs1 = np.arange(r_min, r_max, 0.01)
        rs2 = np.arange(-r_max, -r_min, 0.01)
        rs = np.concatenate([rs1, rs2])
        # normals = np.dot(self.normals, r)
        # print(normals.shape)
        # centers1 = self.points + self.normals
        # centers2 = self.points - self.normals
        centers = []
        for r in rs:
            new_centers = self.points + r * self.normals
            centers.append(new_centers)
        centers = np.array(centers).reshape(-1, 2)
        return centers

    def calc_circle(self, num):
        # idx1 = np.random.choice(len(self.points), size=num)
        # idx2 = np.random.choice(len(self.points), size=num)
        # idx1 = list(range(len(self.points))) * len(self.points)
        # idx2 = np.repeat(range(len(self.points)), len(self.points))
        # print(idx1)
        # print(idx2)
        # centers = self.intersection(
        #     self.points[idx1],
        #     self.points[idx2],
        #     self.normals[idx1],
        #     self.normals[idx2])
        centers = self.get_centers()

        # r1 = self.distance(centers, self.points[idx1])
        # r2 = self.distance(centers, self.points[idx2])
        # radiuses = (r1 + r2) / 2

        # ans_idx = (radiuses < 0.5) & (np.abs(r1 - r2) < 0.5)
        # print(ans_idx)
        # centers, radiuses = centers[ans_idx], radiuses[ans_idx]

        # count_edge, count_body = self.count_circle_point_num(centers, radiuses)
        # ans_idx = (count_edge > 40) & (count_body < 20)
        # return centers[ans_idx], radiuses[ans_idx]
        return centers, np.zeros_like(centers)

    def plot_result(self, ax, centers, radiuses, circle=False):
        ax.scatter(self.points[:, 0], self.points[:, 1],
                   alpha=0.2, color='tab:blue')
        ax.scatter(centers[:, 0], centers[:, 1], alpha=0.005, color='tab:red')
        if circle:
            for center, radius in zip(centers, radiuses):
                c = patches.Circle(
                    xy=center,
                    radius=radius,
                    ec='tab:red',
                    fill=False,
                    alpha=0.2,
                    color='tab:orange')
                ax.add_patch(c)
        ax.set_aspect('equal')


def main():
    color_raw = o3d.io.read_image('images/color/color16087244608437330.png')
    depth_raw = o3d.io.read_image('images/depth/depth16087244608437330.png')
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    start_time = time.time()

    cylinderDetectior = CylinderDetectior(pcd)
    # cylinderDetectior.test_intersection()
    centers, radiuses = cylinderDetectior.calc_circle(num=10000)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('elapsed time: {} [s]'.format(elapsed_time))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].set_title('RGB image')
    ax[0].imshow(rgbd_image.color)
    ax[0].grid(False)
    ax[1].set_title('depth image')
    ax[1].imshow(rgbd_image.depth, cmap='jet')
    ax[1].grid(False)
    fig.savefig('rgbd.png')

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_title('result')
    cylinderDetectior.plot_result(ax, centers, radiuses)
    fig.savefig('result.png')

    hist, xedges, yedges = np.histogram2d(
        centers[:, 0], centers[:, 1], bins=100)
    hist[hist < np.max(hist) * 0.6] = 0
    
    np.save('hist', hist)

    plt.clf()
    plt.imshow(hist.T, origin='lower')
    plt.grid(False)
    fig.savefig('heatmap.png')
    # plt.show()


if __name__ == '__main__':
    main()
