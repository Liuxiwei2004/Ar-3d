import os
import glob
import argparse
import numpy as np
import cv2


def calibrate_camera(image_dir: str, rows: int, cols: int, square_size: float, output_path: str, show_corners: bool = False):
    """
    使用棋盘格图像标定相机内参矩阵 K 和畸变系数 D。

    参数:
        image_dir: 标定图像所在文件夹路径
        rows: 棋盘格内角点行数 (例如 6)
        cols: 棋盘格内角点列数 (例如 9)
        square_size: 单个棋盘格方格的实际物理边长 (单位: 米, 例如 0.025)
        output_path: 输出 .npz 文件路径
        show_corners: 是否可视化检测到的角点
    """
    # 亚像素角点优化终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 构建世界坐标系下的 3D 棋盘格角点坐标 (Z=0 平面)
    # 形状: (rows * cols, 3)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D 世界坐标点集合
    imgpoints = []  # 2D 图像像素坐标点集合

    # 支持常见图片后缀
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in directory '{image_dir}'! ")

    img_size = None
    valid_count = 0

    for fname in image_paths:
        img = cv2.imread(fname)
        if img is None:
            print(f"[Warning] Unable to read image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (width, height)

        # 查找棋盘格内角点
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            valid_count += 1
            objpoints.append(objp)

            # 亚像素级角点精确化
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_subpix)
            print(f"[Success] Corner detected: {os.path.basename(fname)}")

            if show_corners:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, (cols, rows), corners_subpix, ret)
                cv2.imshow("Chessboard Corners", vis)
                cv2.waitKey(200)
        else:
            print(f"[Skip] Failed to detect full corner: {os.path.basename(fname)}")

    if show_corners:
        cv2.destroyAllWindows()

    if valid_count < 3:
        raise RuntimeError(f"There are too few valid calibration images (Only {valid_count})， please provide at least 10~20 clear checkerboard photos. ")

    print(f"\nPerforming camera calibration based on {valid_count}/{len(image_paths)} valid images...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # 计算所有有效图像的平均重投影误差 (Mean Reprojection Error)
    total_error = 0.0
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        pts_gt = imgpoints[i].reshape(-1, 2).astype(np.float64)
        pts_proj = imgpoints_proj.reshape(-1, 2).astype(np.float64)
        error = cv2.norm(pts_gt, pts_proj, cv2.NORM_L2) / len(pts_gt)
        total_error += error
    mean_reproj_error = total_error / len(objpoints)

    # 确保输出目录存在并保存标准键名
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(
        output_path,
        camera_matrix=camera_matrix.astype(np.float64),
        dist_coeffs=dist_coeffs.astype(np.float64),
        mean_reproj_error=np.float64(mean_reproj_error)
    )

    print("\n================ Camera calibration results ================")
    print(f"Calibration parameters have been saved to: {output_path}")
    print(f"Camera intrinsic reference matrix camera_matrix (K):\n{camera_matrix}")
    print(f"Coefficient of distortion dist_coeffs (D):\n{dist_coeffs}")
    print(f"Mean Reprojection Error: {mean_reproj_error:.4f} px")
    print("==============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration using Planar Chessboard")
    parser.add_argument("--image_dir", type=str, default="assets/calibration_images", help="Catalog of calibration images")
    parser.add_argument("--rows", type=int, default=6, help="The number of rows of corner points in the checkerboard")
    parser.add_argument("--cols", type=int, default=9, help="The number of corner points in the checkerboard")
    parser.add_argument("--square_size", type=float, default=0.025, help="Physical side length of a single square (in meters)")
    parser.add_argument("--output", type=str, default="assets/camera_params.npz", help="Output calibration file save path")
    parser.add_argument("--show", action="store_true", help="Whether to display the corner detection window")
    args = parser.parse_args()

    calibrate_camera(args.image_dir, args.rows, args.cols, args.square_size, args.output, args.show)