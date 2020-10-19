from bounds import BBox

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class Ray:
    def __init__(self, o, d):
        self.o = np.array(o)
        self.d = np.array(d)
        self.d_inv = 1 / self.d

    def propagate(self, t):
        return self.o + self.d * t


class Box3DIntersection(BBox):
    def __init__(self, minCoord=None, maxCoord=None):
        super().__init__(minCoord, maxCoord)

    def setFromBBox(self, bbox: BBox):
        self._minCoord = bbox.minCoord
        self._maxCoord = bbox.maxCoord

    def intersectsWithRay(self, ray: Ray):
        # originated from https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
        # Find parameter t of ray for minimum and maximum bounds
        txmin = self.rayIntersection(self.minCoord, ray, 0)
        txmax = self.rayIntersection(self.maxCoord, ray, 0)

        tymin = self.rayIntersection(self.minCoord, ray, 1)
        tymax = self.rayIntersection(self.maxCoord, ray, 1)

        tzmin = self.rayIntersection(self.minCoord, ray, 2)
        tzmax = self.rayIntersection(self.maxCoord, ray, 2)

        # Swap min max values corresponding to direction of ray
        if ray.d[0] < 0:
            _txmin, _txmax = txmin, txmax
            txmin = _txmax
            txmax = _txmin

        if ray.d[1] < 0:
            _tymin, _tymax = tymin, tymax
            tymin = _tymax
            tymax = _tymin

        if ray.d[2] < 0:
            _tzmin, _tzmax = tzmin, tzmax
            tzmin = _tzmax
            tzmax = _tzmin

        # ray does not hit the box in x or y direction
        if (txmin > tymax) | (tymin > tymax):
            return False

        tmin = max(txmin, tymin, tzmin)
        tmax = min(txmax, tymax, tzmax)

        # ray does not hit the box in z direction
        if (tmin > tzmax) | (tzmin > tmax):
            return False

        return (tmin > 0.0) & (tmax > 0.0)

    def setMinCoord(self, minCoord):
        self._minCoord = minCoord

    def setMaxCoord(self, maxCoord):
        self._maxCoord = maxCoord

    @staticmethod
    def rayIntersection(bbox, ray: Ray, idx):
        t = (bbox[idx] - ray.o[idx]) * ray.d_inv[idx]
        return t

    def isIn_array(self, points):
        isIn_bool_array0 = (self.minCoord[0] < points[:, 0]) & (points[:, 0] < self.maxCoord[0])
        isIn_bool_array1 = (self.minCoord[1] < points[:, 1]) & (points[:, 1] < self.maxCoord[1])
        isIn_bool_array2 = (self.minCoord[2] < points[:, 2]) & (points[:, 2] < self.maxCoord[2])
        isIn_bool_array_final = isIn_bool_array0 & isIn_bool_array1 & isIn_bool_array2
        return isIn_bool_array_final


if __name__ == "__main__":
    import random

    def plot_bbox(ax, b0x, b0y, b0z, b1x, b1y, b1z):
        color = 'blue'
        ax.plot([b0x, b0x], [b0y, b0y], [b0z, b1z], c=color)
        ax.plot([b0x, b0x], [b1y, b1y], [b0z, b1z], c=color)
        ax.plot([b1x, b1x], [b0y, b0y], [b0z, b1z], c=color)
        ax.plot([b1x, b1x], [b1y, b1y], [b0z, b1z], c=color)

        ax.plot([b0x, b0x], [b0y, b1y], [b0z, b0z], c=color)
        ax.plot([b0x, b0x], [b0y, b1y], [b1z, b1z], c=color)
        ax.plot([b1x, b1x], [b0y, b1y], [b0z, b0z], c=color)
        ax.plot([b1x, b1x], [b0y, b1y], [b1z, b1z], c=color)

        ax.plot([b0x, b1x], [b0y, b0y], [b0z, b0z], c=color)
        ax.plot([b0x, b1x], [b0y, b0y], [b1z, b1z], c=color)
        ax.plot([b0x, b1x], [b1y, b1y], [b0z, b0z], c=color)
        ax.plot([b0x, b1x], [b1y, b1y], [b1z, b1z], c=color)

    b0x = 2
    b0y = 2
    b0z = 2
    b1x = 4
    b1y = 4
    b1z = 4

    bbox = Box3DIntersection([b0x, b0y, b0z], [b1x, b1y, b1z])

    ray = Ray([1, 1, 1], [1.1, 1.1, 1.1])
    print(bbox.intersectsWithRay(ray))

    n = 25
    t = 3
    ps_start = np.zeros((n, 3))
    ps_end = np.zeros((n, 3))
    colors = []
    for i in range(n):
        ox = 1
        oy = 1
        oz = 1
        # ox = random.random() * 2
        # oy = random.random() * 2
        # oz = random.random() * 2
        dx = random.random() * 2
        dy = random.random() * 2
        dz = random.random() * 2

        ray = Ray([ox, oy, oz], [dx, dy, dz])

        ps_start[i] = [ox, oy, oz]
        ps_end[i] = ray.propagate(t)

        ray_intersects = bbox.intersectsWithRay(ray)
        if ray_intersects:
            colors.append('green')
        else:
            colors.append('red')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx, (p_start, p_end) in enumerate(zip(ps_start, ps_end)):
        xs = p_start[0], p_end[0]
        ys = p_start[1], p_end[1]
        zs = p_start[2], p_end[2]
        ax.plot(xs, ys, zs, colors[idx])

    plot_bbox(ax, b0x, b0y, b0z, b1x, b1y, b1z)

    plt.show()
