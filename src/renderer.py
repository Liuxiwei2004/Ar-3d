import cv2
import numpy as np


class ARRenderer:
    """
    Augmented Reality Renderer.

    Coordinate convention (temporary):
        Origin: center of the marker
        X: right
        Y: up
        Z: normal to the marker plane
        Marker plane: XY plane
        Unit: meters
    """

    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        marker_size=0.05,
        axis_length=0.04,
        cube_size=0.025,
        cube_height=0.04,
        cube_direction=1,
        show_axis=True,
        show_cube=True,
        show_faces=True,
        face_alpha=0.35,
    ):
        self.camera_matrix = np.asarray(
            camera_matrix, dtype=np.float64
        )
        self.dist_coeffs = np.asarray(
            dist_coeffs, dtype=np.float64
        )

        self.marker_size = float(marker_size)
        self.axis_length = float(axis_length)
        self.cube_size = float(cube_size)
        self.cube_height = float(cube_height)
        self.cube_direction = 1 if cube_direction >= 0 else -1

        self.show_axis = show_axis
        self.show_cube = show_cube
        self.show_faces = show_faces
        self.face_alpha = float(face_alpha)

        self._validate_parameters()

    def _validate_parameters(self):
        if self.camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape (3, 3)")

        if self.marker_size <= 0:
            raise ValueError("marker_size must be positive")

        if self.axis_length <= 0:
            raise ValueError("axis_length must be positive")

        if self.cube_size <= 0 or self.cube_height <= 0:
            raise ValueError("cube dimensions must be positive")

        if not 0.0 <= self.face_alpha <= 1.0:
            raise ValueError("face_alpha must be between 0 and 1")

    def project_points(self, points_3d, rvec, tvec):
        """Project 3D object-coordinate points onto the image."""
        points_3d = np.asarray(
            points_3d, dtype=np.float64
        ).reshape(-1, 3)

        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

        projected, _ = cv2.projectPoints(
            points_3d,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )

        return np.round(projected.reshape(-1, 2)).astype(np.int32)

    def _draw_axes(self, image, rvec, tvec):
        """Draw X, Y and Z coordinate axes."""
        origin = np.array([0.0, 0.0, 0.0])

        axis_points = np.array([
            origin,
            [self.axis_length, 0.0, 0.0],
            [0.0, self.axis_length, 0.0],
            [0.0, 0.0, self.axis_length],
        ])

        points_2d = self.project_points(
            axis_points, rvec, tvec
        )

        origin_2d = tuple(points_2d[0])

        # OpenCV uses BGR color order.
        axes = [
            (1, (0, 0, 255)),    # X: red
            (2, (0, 255, 0)),    # Y: green
            (3, (255, 0, 0)),    # Z: blue
        ]

        for point_index, color in axes:
            endpoint = tuple(points_2d[point_index])

            cv2.line(
                image,
                origin_2d,
                endpoint,
                color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )

        cv2.circle(
            image,
            origin_2d,
            4,
            (255, 255, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

    def _get_cube_vertices(self):
        """Create the eight 3D vertices of a cube."""
        half = self.cube_size / 2.0
        z_bottom = 0.0
        z_top = self.cube_direction * self.cube_height

        return np.array([
            [-half, -half, z_bottom],
            [ half, -half, z_bottom],
            [ half,  half, z_bottom],
            [-half,  half, z_bottom],
            [-half, -half, z_top],
            [ half, -half, z_top],
            [ half,  half, z_top],
            [-half,  half, z_top],
        ], dtype=np.float64)

    def _draw_cube_edges(self, image, vertices_2d):
        """Draw the twelve edges of the cube."""
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        for start, end in edges:
            cv2.line(
                image,
                tuple(vertices_2d[start]),
                tuple(vertices_2d[end]),
                (0, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    def _draw_cube_faces(self, image, vertices_2d):
        """Draw translucent cube faces."""
        faces = [
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [4, 5, 6, 7],
        ]

        overlay = image.copy()

        face_colors = [
            (255, 100, 50),
            (50, 180, 255),
            (100, 255, 100),
            (255, 100, 200),
            (100, 200, 255),
        ]

        for face, color in zip(faces, face_colors):
            polygon = vertices_2d[face].reshape(-1, 1, 2)

            cv2.fillPoly(
                overlay,
                [polygon],
                color,
                lineType=cv2.LINE_AA,
            )

        cv2.addWeighted(
            overlay,
            self.face_alpha,
            image,
            1.0 - self.face_alpha,
            0,
            dst=image,
        )

    def render(self, frame, rvec, tvec):
        """
        Render virtual objects onto a camera frame.

        Args:
            frame: BGR image from the camera.
            rvec: Rotation vector from solvePnP.
            tvec: Translation vector from solvePnP.

        Returns:
            A new image with AR objects rendered.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty")

        if rvec is None or tvec is None:
            return frame.copy()

        output = frame.copy()

        if self.show_cube:
            vertices_3d = self._get_cube_vertices()
            vertices_2d = self.project_points(
                vertices_3d, rvec, tvec
            )

            if self.show_faces:
                self._draw_cube_faces(output, vertices_2d)

            self._draw_cube_edges(output, vertices_2d)

        if self.show_axis:
            self._draw_axes(output, rvec, tvec)

        return output
