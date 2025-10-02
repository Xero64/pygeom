from typing import TYPE_CHECKING

from numpy import arctan2

from .vector2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import NDArray


def angle_between_vectors(veca: Vector2D, vecb: Vector2D) -> 'NDArray':
    adb = veca.dot(vecb)
    axbm = veca.cross(vecb)
    return arctan2(axbm, adb)
