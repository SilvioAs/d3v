import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class Icosphere:
    def upper_vertex(self, i):
        return self.octahedron_vertex(self.r, self.alpha, i)

    def lower_vertex(self, i):
        return self.octahedron_vertex(self.r, -self.alpha, i)

    @staticmethod
    def octahedron_vertex(r, alpha, i):
        x = r * np.cos(alpha) * np.cos(np.radians(72) * i)
        y = r * np.cos(alpha) * np.sin(np.radians(72) * i)
        z = r * np.sin(alpha)
        return x, y, z

    def __init__(self, r, x=0, y=0, z=0):
        self.xc = x
        self.yc = y
        self.zc = z
        self.r = r
        self.alpha = np.arctan(0.5)

        self.triangle_coords = np.array([])
        self.triangle_normals = np.array([])

        self.create()

    def create(self):
        self.triangle_coords = []

        n = [0, 0, self.r]  # north pole
        s = [0, 0, -self.r]  # south pole

        # Start from top - 5 triangles
        for i in range(5):
            vertex2 = self.upper_vertex(i)
            vertex3 = self.upper_vertex(i + 1)
            self.triangle_coords.append([n, vertex2, vertex3])

        # Middle part - 10 triangles
        for i in range(5):
            # --> 2 triangles per iteration
            vertex1 = self.upper_vertex(i)
            vertex2 = self.lower_vertex(i)
            vertex3 = self.lower_vertex(i+1)
            self.triangle_coords.append([vertex1, vertex2, vertex3])

            vertex1 = self.upper_vertex(i)
            vertex2 = self.upper_vertex(i+1)
            vertex3 = self.lower_vertex(i+1)
            self.triangle_coords.append([vertex1, vertex3, vertex2])

        # Lower part - 5 triangles
        for i in range(5):
            vertex2 = self.lower_vertex(i)
            vertex3 = self.lower_vertex(i+1)
            self.triangle_coords.append([s, vertex3, vertex2])

        self.triangle_coords = np.array(self.triangle_coords)
        self.triangle_normals = np.array(self.triangle_coords)
        self.triangle_coords[:, :, 0] += self.xc
        self.triangle_coords[:, :, 1] += self.yc
        self.triangle_coords[:, :, 2] += self.zc

    @staticmethod
    def v3_length(v3):
        return np.sqrt(v3[0] ** 2 + v3[1] ** 2 + v3[2] ** 2)

    def subDivide(self):
        self.triangle_coords[:, :, 0] -= self.xc
        self.triangle_coords[:, :, 1] -= self.yc
        self.triangle_coords[:, :, 2] -= self.zc

        subdivided_triangles = []
        for triangle_idx, triangle in enumerate(self.triangle_coords):
            print(triangle_idx)
            new_vertices = np.zeros((3, 3))
            new_vertices[0] = (triangle[0] + triangle[1]) / 2
            new_vertices[0] *= self.r / self.v3_length(new_vertices[0])

            new_vertices[1] = (triangle[1] + triangle[2]) / 2
            new_vertices[1] *= self.r / self.v3_length(new_vertices[1])

            new_vertices[2] = (triangle[2] + triangle[0]) / 2
            new_vertices[2] *= self.r / self.v3_length(new_vertices[2])

            new_triangle1 = [new_vertices[0], new_vertices[1], new_vertices[2]]
            new_triangle2 = [triangle[0], new_vertices[0], new_vertices[2]]
            new_triangle3 = [triangle[1], new_vertices[1], new_vertices[0]]
            new_triangle4 = [triangle[2], new_vertices[2], new_vertices[1]]
            subdivided_triangles.append(new_triangle1)
            subdivided_triangles.append(new_triangle2)
            subdivided_triangles.append(new_triangle3)
            subdivided_triangles.append(new_triangle4)

        self.triangle_coords = np.array(subdivided_triangles)
        self.triangle_normals = np.array(subdivided_triangles)
        self.triangle_coords[:, :, 0] += self.xc
        self.triangle_coords[:, :, 1] += self.yc
        self.triangle_coords[:, :, 2] += self.zc

    def get_coords(self):
        return self.triangle_coords

    def get_normals(self):
        return self.triangle_normals


if __name__ == "__main__":
    icosphere = Icosphere(1.0, 5, 5, 5)
    # icosphere.subDivide()
    # icosphere.subDivide()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    coords = icosphere.get_coords()
    coords = coords[15:20]
    for triangle_idx, triangle in enumerate(coords):
        ax.plot(
            [triangle[0, 0], triangle[1, 0]],
            [triangle[0, 1], triangle[1, 1]],
            [triangle[0, 2], triangle[1, 2]], color='black'
        )

        ax.plot(
            [triangle[1, 0], triangle[2, 0]],
            [triangle[1, 1], triangle[2, 1]],
            [triangle[1, 2], triangle[2, 2]], color='black'
        )

        ax.plot(
            [triangle[2, 0], triangle[0, 0]],
            [triangle[2, 1], triangle[0, 1]],
            [triangle[2, 2], triangle[0, 2]], color='black'
        )

    ax.scatter(coords[:, :, 0], coords[:, :, 1], coords[:, :, 2])
    plt.show()
