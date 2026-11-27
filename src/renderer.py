import cv2
import numpy as np
import trimesh


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
        show_shadow=True,
        shadow_alpha=0.30,
        shadow_blur=21,
        shadow_scale=1.5,
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
        self.show_shadow = show_shadow
        self.shadow_alpha = float(shadow_alpha)
        self.shadow_blur = int(shadow_blur)
        self.shadow_scale = float(shadow_scale)

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
        
        if not 0.0 <= self.shadow_alpha <= 1.0:
            raise ValueError("shadow_alpha must be between 0 and 1")

        if self.shadow_blur < 1 or self.shadow_blur % 2 == 0:
            raise ValueError("shadow_blur must be a positive odd number")

        if self.shadow_scale <= 0:
            raise ValueError("shadow_scale must be positive")

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
    
    def _draw_shadow(self, image, rvec, tvec):
        """
        Draw a soft elliptical shadow on the marker plane.

        The shadow is defined in marker coordinates,
        then projected onto the camera image.
        """

        # Ellipse center on the marker plane.
        center_x = 0.0
        center_y = -self.cube_size * 0.35

        # Ellipse radii in meters.
        radius_x = self.cube_size * self.shadow_scale
        radius_y = self.cube_size * 0.55

        # Generate ellipse boundary points.
        angles = np.linspace(
            0,
            2 * np.pi,
            80,
            endpoint=False,
        )

        ellipse_points = np.array([
            [
                center_x + radius_x * np.cos(angle),
                center_y + radius_y * np.sin(angle),
                0.0005,
            ]
            for angle in angles
        ], dtype=np.float64)

        # Project ellipse to image coordinates.
        projected = self.project_points(
            ellipse_points,
            rvec,
            tvec,
        )

        # Create a mask for the shadow.
        mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        cv2.fillPoly(
            mask,
            [projected.reshape(-1, 1, 2)],
            255,
            lineType=cv2.LINE_AA,
        )

        # Soften the shadow boundary.
        blurred_mask = cv2.GaussianBlur(
            mask,
            (self.shadow_blur, self.shadow_blur),
            0,
        )

        # Blend a dark layer onto the image.
        shadow_layer = image.copy()

        shadow_layer[blurred_mask > 0] = (25, 25, 25)

        alpha_mask = (
            blurred_mask.astype(np.float32) / 255.0
        ) * self.shadow_alpha

        alpha_mask = alpha_mask[:, :, np.newaxis]

        result = (
            image.astype(np.float32) * (1.0 - alpha_mask)
            + shadow_layer.astype(np.float32) * alpha_mask
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    
    def load_obj_model(self, obj_path, target_size=0.04):
        """
        Load and normalize an OBJ model.

        Args:
            obj_path: Path to the OBJ file.
            target_size: Desired maximum model dimension in meters.

        Returns:
            Dictionary containing normalized vertices and triangular faces.
        """

        if target_size <= 0:
            raise ValueError("target_size must be positive")

        obj_path = str(obj_path)

        # Load OBJ as a mesh.
        mesh = trimesh.load(
            obj_path,
            force="mesh",
            process=True,
        )

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Loaded OBJ is not a valid triangular mesh")

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("OBJ model contains no vertices or faces")

        # Extract vertices and triangular faces.
        vertices = np.asarray(
            mesh.vertices,
            dtype=np.float64,
        ).copy()

        faces = np.asarray(
            mesh.faces,
            dtype=np.int32,
        ).copy()

        # Move model center to the origin.
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)

        center = (bounds_min + bounds_max) / 2.0
        vertices -= center

        # Normalize model size.
        dimensions = vertices.max(axis=0) - vertices.min(axis=0)
        max_dimension = float(dimensions.max())

        if max_dimension <= 1e-12:
            raise ValueError("OBJ model has zero size")

        scale = target_size / max_dimension
        vertices *= scale

        return {
            "vertices": vertices,
            "faces": faces,
            "center": center,
            "scale": scale,
            "vertex_count": len(vertices),
            "face_count": len(faces),
        }

    
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
    
    def _transform_model_vertices(self, model):
        """
        Place a centered OBJ model on the marker plane.

        The model is centered in X and Y.
        Its bottom (positive direction) or top (negative direction)
        is aligned with the marker plane.
        """
        if not isinstance(model, dict):
            raise TypeError("model must be a dictionary")

        if "vertices" not in model or "faces" not in model:
            raise ValueError("model must contain vertices and faces")

        vertices = np.asarray(
            model["vertices"], dtype=np.float64
        ).reshape(-1, 3).copy()

        if len(vertices) == 0:
            raise ValueError("model contains no vertices")

        if not np.all(np.isfinite(vertices)):
            raise ValueError("model vertices contain invalid values")

        # Move the model so that its bottom or top touches Z = 0.
        if self.cube_direction > 0:
            vertices[:, 2] -= vertices[:, 2].min()
        else:
            vertices[:, 2] -= vertices[:, 2].max()

        vertices[:, 2] *= self.cube_direction

        return vertices
    
    def _draw_obj_model(self, image, model, rvec, tvec):
        """
        Project and render an OBJ mesh with depth sorting
        and feature-edge rendering.
        """
        vertices_3d = self._transform_model_vertices(model)
        faces = np.asarray(model["faces"], dtype=np.int32)

        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("OBJ faces must be triangular")

        if len(faces) == 0:
            raise ValueError("OBJ model contains no faces")

        if np.any(faces < 0) or np.any(faces >= len(vertices_3d)):
            raise ValueError("OBJ face indices are out of range")

        # Calculate camera-space coordinates for depth sorting.
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        camera_vertices = (
            rotation_matrix @ vertices_3d.T + tvec
        ).T

        vertices_2d = self.project_points(
            vertices_3d, rvec, tvec
        )

        # Calculate face normals.
        face_normals = []

        for face in faces:
            v0, v1, v2 = vertices_3d[face]
            normal = np.cross(v1 - v0, v2 - v0)
            length = np.linalg.norm(normal)

            if length > 1e-12:
                normal = normal / length
            else:
                normal = np.zeros(3)

            face_normals.append(normal)

        face_normals = np.asarray(face_normals)

        # Sort faces from far to near.
        face_depths = np.array([
            camera_vertices[face, 2].mean()
            for face in faces
        ])

        face_order = np.argsort(face_depths)[::-1]

        # Draw triangles in depth order.
        overlay = image.copy()
        face_color = (80, 180, 230)

        for face_index in face_order:
            face = faces[face_index]
            polygon = vertices_2d[face].reshape(-1, 1, 2)

            cv2.fillPoly(
                overlay,
                [polygon],
                face_color,
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

        # Build an edge map to identify actual model edges.
        edge_map = {}

        for face_index, face in enumerate(faces):
            for i in range(3):
                a = int(face[i])
                b = int(face[(i + 1) % 3])
                edge = tuple(sorted((a, b)))

                edge_map.setdefault(edge, []).append(
                    face_normals[face_index]
                )

        # Hide edges between coplanar triangles.
        coplanar_threshold = np.cos(np.deg2rad(3.0))

        for edge, normals in edge_map.items():
            draw_edge = len(normals) == 1

            if len(normals) > 1:
                for i in range(len(normals)):
                    for j in range(i + 1, len(normals)):
                        dot = abs(float(np.dot(
                            normals[i], normals[j]
                        )))

                        if dot < coplanar_threshold:
                            draw_edge = True
                            break

                    if draw_edge:
                        break

            if draw_edge:
                start = tuple(vertices_2d[edge[0]])
                end = tuple(vertices_2d[edge[1]])

                cv2.line(
                    image,
                    start,
                    end,
                    (30, 30, 30),
                    1,
                    cv2.LINE_AA,
                )

        return image
    
    def render(self, frame, rvec, tvec, model=None):
        """
        Render virtual objects onto a camera frame.

        Args:
            frame: BGR image from the camera.
            rvec: Rotation vector from solvePnP.
            tvec: Translation vector from solvePnP.
            model: Optional OBJ model returned by load_obj_model().
                  If None, render the built-in cube.

        Returns:
            A new image with AR objects rendered.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty")

        if rvec is None or tvec is None:
            return frame.copy()

        output = frame.copy()

        # Render the built-in cube or the OBJ model.
        if model is None:
            if self.show_shadow and self.show_cube:
                output = self._draw_shadow(output, rvec, tvec)

            if self.show_cube:
                vertices_3d = self._get_cube_vertices()
                vertices_2d = self.project_points(
                    vertices_3d, rvec, tvec
                )

                if self.show_faces:
                    self._draw_cube_faces(output, vertices_2d)

                self._draw_cube_edges(output, vertices_2d)
        else:
            if self.show_shadow:
                output = self._draw_shadow(output, rvec, tvec)

            self._draw_obj_model(
                output, model, rvec, tvec
            )

        if self.show_axis:
            self._draw_axes(output, rvec, tvec)

        return output
