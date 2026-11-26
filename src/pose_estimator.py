import os
import numpy as np
import cv2


class PoseEstimator:
    """
    基于 ArUco 标记的 6-DoF 相机位姿估计器。
    坐标系约定（右手系，原点位于 Marker 中心）：
        - +X 轴：沿 Marker 平面向右
        - +Y 轴：沿 Marker 平面向上
        - +Z 轴：垂直于 Marker 表面向外（朝向观察者）
    角点顺序约定（与 OpenCV ArUco 检测结果一致）：
        1. 左上 (-L/2, +L/2, 0)
        2. 右上 (+L/2, +L/2, 0)
        3. 右下 (+L/2, -L/2, 0)
        4. 左下 (-L/2, -L/2, 0)
    """

    def __init__(
        self,
        camera_params_path: str = "assets/camera_params.npz",
        marker_size: float = 0.05,
        aruco_dict_type: int = cv2.aruco.DICT_4X4_50,
        target_id: int = None
    ):
        if not os.path.exists(camera_params_path):
            raise FileNotFoundError(
                f"Camera parameter file '{camera_params_path}' was not found, please run calibration.py to finish calibration first! "
            )

        data = np.load(camera_params_path)
        self.camera_matrix = data["camera_matrix"].astype(np.float64)
        self.dist_coeffs = data["dist_coeffs"].astype(np.float64)
        self.marker_size = float(marker_size)
        self.target_id = target_id  # 若为 None，则默认追踪检测到的第一个 Marker

        # 定义 Marker 局部坐标系下的 4 个 3D 角点（单位：米）
        half = self.marker_size / 2.0
        self.obj_points = np.array([
            [-half,  half, 0.0],  # Top-Left
            [ half,  half, 0.0],  # Top-Right
            [ half, -half, 0.0],  # Bottom-Right
            [-half, -half, 0.0],  # Bottom-Left
        ], dtype=np.float64)

        # 初始化 ArUco 检测器（兼容 OpenCV 4.7+ 新版 API 与旧版 API）
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            self._use_new_api = True
        else:
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self._use_new_api = False

    def estimate(self, frame: np.ndarray):
        """
        对单帧图像检测 ArUco 标记并求解位姿 (rvec, tvec)。

        返回:
            success (bool): 是否成功检测到目标并求解位姿
            rvec (np.ndarray | None): 旋转向量, shape (3, 1), float64
            tvec (np.ndarray | None): 平移向量, shape (3, 1), float64
        """
        if frame is None or frame.size == 0:
            return False, None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.detector_params)

        # 未检测到任何 Marker 时的异常兜底
        if ids is None or len(ids) == 0:
            return False, None, None

        ids_flat = ids.flatten()

        # 选择目标 Marker 索引
        selected_idx = 0
        if self.target_id is not None:
            matches = np.where(ids_flat == self.target_id)[0]
            if len(matches) == 0:
                return False, None, None
            selected_idx = int(matches[0])

        # 提取对应 Marker 的 2D 角点，shape -> (4, 2)
        img_points = corners[selected_idx].reshape((4, 2)).astype(np.float64)

        # 使用 IPPE_SQUARE（专为平面正方形标记优化的 PnP 算法，若版本不支持则回退到 ITERATIVE）
        pnp_flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
        ok, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            img_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=pnp_flag
        )

        if not ok:
            return False, None, None

        # 严格保证输出 shape 为 (3, 1) 且为 float64
        rvec = np.asarray(rvec, dtype=np.float64).reshape((3, 1))
        tvec = np.asarray(tvec, dtype=np.float64).reshape((3, 1))
        return True, rvec, tvec


# =====================================================================
# 对外统一标准函数接口（供 Member 2 直接调用）
# =====================================================================
_DEFAULT_ESTIMATOR = None


def get_pose(
    frame: np.ndarray,
    camera_params_path: str = "assets/camera_params.npz",
    marker_size: float = 0.05
):
    """
    标准位姿估计接口。
    Returns:
        success: bool
        rvec: np.ndarray, shape (3, 1)
        tvec: np.ndarray, shape (3, 1)
    """
    global _DEFAULT_ESTIMATOR
    if _DEFAULT_ESTIMATOR is None or _DEFAULT_ESTIMATOR.marker_size != float(marker_size):
        _DEFAULT_ESTIMATOR = PoseEstimator(
            camera_params_path=camera_params_path,
            marker_size=marker_size
        )
    return _DEFAULT_ESTIMATOR.estimate(frame)


# =====================================================================
# Member 1 本地自测入口：验证检测效果并绘制 3D 坐标轴
# =====================================================================
if __name__ == "__main__":
    test_img_path = "assets/test_images/sample.jpg"
    params_path = "assets/camera_params.npz"
    marker_len = 0.05  # 请修改为你实际打印的 Marker 边长（单位：米）

    if not os.path.exists(test_img_path):
        print(f"[Tip] Please put a test photo with ArUco tag in: {test_img_path}")
    else:
        img = cv2.imread(test_img_path)
        estimator = PoseEstimator(camera_params_path=params_path, marker_size=marker_len)
        success, rvec, tvec = estimator.estimate(img)

        print(f"Test for success: {success}")
        if success:
            print(f"rvec (shape={rvec.shape}):\n{rvec}")
            print(f"tvec (shape={tvec.shape}):\n{tvec}")

            # 在图像上画出 X(红)、Y(绿)、Z(蓝) 坐标轴验证方向是否正确
            vis = img.copy()
            cv2.drawFrameAxes(vis, estimator.camera_matrix, estimator.dist_coeffs, rvec, tvec, marker_len * 0.8)
            os.makedirs("results", exist_ok=True)
            cv2.imwrite("results/member1_pose_check.jpg", vis)
            print("[Success] Axis verification chart has been saved to: results/member1_pose_check.jpg")
            print("Check points: the red axis (X) should face right, the green axis (Y) should face up, and the blue axis (Z) should face out of the vertical paper!")