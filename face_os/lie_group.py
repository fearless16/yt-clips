"""Lie-Group Transforms Module.

Geometric transforms using Lie-group theory.

Groups:
    SE(2): Special Euclidean group (rotation + translation)
        - theta: rotation angle (radians)
        - tx, ty: translation
    
    SIM(2): Similarity group (rotation + translation + scale)
        - theta: rotation angle (radians)
        - tx, ty: translation
        - scale: uniform scale

Mathematical Properties:
    - Group closure: T1 * T2 ∈ G
    - Associativity: (T1 * T2) * T3 = T1 * (T2 * T3)
    - Identity: T * I = I * T = T
    - Inverse: T * T^-1 = I
    - No skew or flip: det(R) = 1

Exponential/Logarithmic Maps:
    exp: Lie algebra → Lie group
    log: Lie group → Lie algebra

Geodesic Interpolation:
    T(t) = exp((1-t) * log(T1) + t * log(T2))

References:
    - Lie groups for computer vision (Ma et al.)
    - SE(2) and SIM(2) for 2D transforms
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SE2Transform:
    """SE(2) transform: rotation + translation.

    Group: SE(2) = SO(2) ⋉ R²
    Lie algebra: se(2) = [theta, tx, ty]
    """

    theta: float  # Rotation angle (radians)
    tx: float     # Translation x
    ty: float     # Translation y

    @staticmethod
    def identity() -> 'SE2Transform':
        """Identity transform."""
        return SE2Transform(theta=0.0, tx=0.0, ty=0.0)

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 homogeneous matrix.

        Returns:
            3x3 transformation matrix
        """
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([
            [c, -s, self.tx],
            [s,  c, self.ty],
            [0,  0,  1]
        ], dtype=np.float64)

    @staticmethod
    def from_matrix(M: np.ndarray) -> 'SE2Transform':
        """Extract from 3x3 matrix.

        Args:
            M: 3x3 transformation matrix

        Returns:
            SE2Transform
        """
        return SE2Transform(
            theta=np.arctan2(M[1, 0], M[0, 0]),
            tx=M[0, 2],
            ty=M[1, 2]
        )

    def log(self) -> np.ndarray:
        """Lie algebra: [theta, tx, ty].

        Returns:
            3D vector in Lie algebra
        """
        return np.array([self.theta, self.tx, self.ty])

    @staticmethod
    def exp(v: np.ndarray) -> 'SE2Transform':
        """Exponential map from Lie algebra.

        Args:
            v: 3D vector [theta, tx, ty]

        Returns:
            SE2Transform
        """
        return SE2Transform(theta=v[0], tx=v[1], ty=v[2])

    def compose(self, other: 'SE2Transform') -> 'SE2Transform':
        """Compose with another SE(2) transform.

        Args:
            other: Other transform

        Returns:
            Composed transform
        """
        M1 = self.to_matrix()
        M2 = other.to_matrix()
        M3 = M1 @ M2
        return SE2Transform.from_matrix(M3)

    def inverse(self) -> 'SE2Transform':
        """Inverse transform.

        Returns:
            Inverse transform
        """
        M = self.to_matrix()
        M_inv = np.linalg.inv(M)
        return SE2Transform.from_matrix(M_inv)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a 2D point.

        Args:
            point: 2D point [x, y]

        Returns:
            Transformed point
        """
        c, s = np.cos(self.theta), np.sin(self.theta)
        x, y = point[0], point[1]
        x_new = c * x - s * y + self.tx
        y_new = s * x + c * y + self.ty
        return np.array([x_new, y_new])


@dataclass
class SIM2Transform:
    """SIM(2) transform: rotation + translation + scale.

    Group: SIM(2) = SO(2) ⋉ R² × R⁺
    Lie algebra: sim(2) = [theta, tx, ty, log(scale)]
    """

    theta: float  # Rotation angle (radians)
    tx: float     # Translation x
    ty: float     # Translation y
    scale: float  # Uniform scale

    @staticmethod
    def identity() -> 'SIM2Transform':
        """Identity transform."""
        return SIM2Transform(theta=0.0, tx=0.0, ty=0.0, scale=1.0)

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 homogeneous matrix.

        Returns:
            3x3 transformation matrix
        """
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([
            [self.scale * c, -self.scale * s, self.tx],
            [self.scale * s,  self.scale * c, self.ty],
            [0,               0,               1]
        ], dtype=np.float64)

    @staticmethod
    def from_matrix(M: np.ndarray) -> 'SIM2Transform':
        """Extract from 3x3 matrix.

        Args:
            M: 3x3 transformation matrix

        Returns:
            SIM2Transform
        """
        # Extract scale from rotation part
        R = M[:2, :2]
        scale = np.sqrt(np.linalg.det(R))
        
        # Extract rotation
        R_normalized = R / scale
        theta = np.arctan2(R_normalized[1, 0], R_normalized[0, 0])
        
        return SIM2Transform(
            theta=theta,
            tx=M[0, 2],
            ty=M[1, 2],
            scale=scale
        )

    def log(self) -> np.ndarray:
        """Lie algebra: [theta, tx, ty, log(scale)].

        Returns:
            4D vector in Lie algebra
        """
        return np.array([self.theta, self.tx, self.ty, np.log(self.scale)])

    @staticmethod
    def exp(v: np.ndarray) -> 'SIM2Transform':
        """Exponential map from Lie algebra.

        Args:
            v: 4D vector [theta, tx, ty, log(scale)]

        Returns:
            SIM2Transform
        """
        return SIM2Transform(
            theta=v[0],
            tx=v[1],
            ty=v[2],
            scale=np.exp(v[3])
        )

    def compose(self, other: 'SIM2Transform') -> 'SIM2Transform':
        """Compose with another SIM(2) transform.

        Args:
            other: Other transform

        Returns:
            Composed transform
        """
        M1 = self.to_matrix()
        M2 = other.to_matrix()
        M3 = M1 @ M2
        return SIM2Transform.from_matrix(M3)

    def inverse(self) -> 'SIM2Transform':
        """Inverse transform.

        Returns:
            Inverse transform
        """
        M = self.to_matrix()
        M_inv = np.linalg.inv(M)
        return SIM2Transform.from_matrix(M_inv)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a 2D point.

        Args:
            point: 2D point [x, y]

        Returns:
            Transformed point
        """
        c, s = np.cos(self.theta), np.sin(self.theta)
        x, y = point[0], point[1]
        x_new = self.scale * (c * x - s * y) + self.tx
        y_new = self.scale * (s * x + c * y) + self.ty
        return np.array([x_new, y_new])


def interpolate_se2(
    T1: SE2Transform,
    T2: SE2Transform,
    t: float,
) -> SE2Transform:
    """Geodesic interpolation on SE(2).

    T(t) = exp((1-t) * log(T1) + t * log(T2))

    Args:
        T1: Start transform
        T2: End transform
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated transform
    """
    v1 = T1.log()
    v2 = T2.log()
    v_interp = (1 - t) * v1 + t * v2
    return SE2Transform.exp(v_interp)


def interpolate_sim2(
    T1: SIM2Transform,
    T2: SIM2Transform,
    t: float,
) -> SIM2Transform:
    """Geodesic interpolation on SIM(2).

    T(t) = exp((1-t) * log(T1) + t * log(T2))

    Args:
        T1: Start transform
        T2: End transform
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated transform
    """
    v1 = T1.log()
    v2 = T2.log()
    v_interp = (1 - t) * v1 + t * v2
    return SIM2Transform.exp(v_interp)


def se2_log(T: SE2Transform) -> np.ndarray:
    """Logarithmic map for SE(2).

    Args:
        T: SE(2) transform

    Returns:
        Lie algebra vector
    """
    return T.log()


def se2_exp(v: np.ndarray) -> SE2Transform:
    """Exponential map for SE(2).

    Args:
        v: Lie algebra vector

    Returns:
        SE(2) transform
    """
    return SE2Transform.exp(v)


def sim2_log(T: SIM2Transform) -> np.ndarray:
    """Logarithmic map for SIM(2).

    Args:
        T: SIM(2) transform

    Returns:
        Lie algebra vector
    """
    return T.log()


def sim2_exp(v: np.ndarray) -> SIM2Transform:
    """Exponential map for SIM(2).

    Args:
        v: Lie algebra vector

    Returns:
        SIM(2) transform
    """
    return SIM2Transform.exp(v)


def geodesic_distance_se2(
    T1: SE2Transform,
    T2: SE2Transform,
) -> float:
    """Geodesic distance between two SE(2) transforms.

    d(T1, T2) = ||log(T1^-1 * T2)||

    Args:
        T1: First transform
        T2: Second transform

    Returns:
        Geodesic distance
    """
    T1_inv = T1.inverse()
    T_rel = T1_inv.compose(T2)
    v = T_rel.log()
    return float(np.linalg.norm(v))


def geodesic_distance_sim2(
    T1: SIM2Transform,
    T2: SIM2Transform,
) -> float:
    """Geodesic distance between two SIM(2) transforms.

    d(T1, T2) = ||log(T1^-1 * T2)||

    Args:
        T1: First transform
        T2: Second transform

    Returns:
        Geodesic distance
    """
    T1_inv = T1.inverse()
    T_rel = T1_inv.compose(T2)
    v = T_rel.log()
    return float(np.linalg.norm(v))
