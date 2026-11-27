import os
import cv2


def main():
    os.makedirs("assets/markers", exist_ok=True)
    # 与约定一致：使用 DICT_4X4_50 字典，生成 ID=0 的标记
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_id = 0
    marker_size_px = 600  # 高分辨率便于清晰打印

    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)

    # 添加白色边框（Quiet Zone），防止打印裁剪后难以识别外轮廓
    border_size = 60
    marker_with_border = cv2.copyMakeBorder(
        marker_img, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=255
    )

    out_path = f"assets/markers/aruco_4x4_id{marker_id}.png"
    cv2.imwrite(out_path, marker_with_border)
    print(f"[Success] ArUco marker has been saved to: {out_path}")
    print("Please print the picture and use a ruler to measure the actual physical side length (in meters) of the black square.")


if __name__ == "__main__":
    main()