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

        self.triangles = np.array([])

        self.create()

    def create(self):
        self.triangles = []

        n = [0, 0, self.r]  # north pole
        s = [0, 0, -self.r]  # south pole

        # Start from top - 5 triangles
        for i in range(5):
            vertex2 = self.upper_vertex(i)
            vertex3 = self.upper_vertex(i + 1)
            self.triangles.append([n, vertex2, vertex3])

        # Middle part - 10 triangles
        for i in range(5):
            # --> 2 triangles per iteration
            vertex1 = self.upper_vertex(i)
            vertex2 = self.lower_vertex(i)
            vertex3 = self.lower_vertex(i+1)
            self.triangles.append([vertex1, vertex2, vertex3])

            vertex1 = self.upper_vertex(i)
            vertex2 = self.upper_vertex(i+1)
            vertex3 = self.lower_vertex(i+1)
            self.triangles.append([vertex1, vertex2, vertex3])

        # Lower part - 5 triangles
        for i in range(5):
            vertex2 = self.lower_vertex(i)
            vertex3 = self.lower_vertex(i+1)
            self.triangles.append([s, vertex2, vertex3])

        self.triangles = np.array(self.triangles)
        self.triangles[:, :, 0] += self.xc
        self.triangles[:, :, 1] += self.yc
        self.triangles[:, :, 2] += self.zc

    @staticmethod
    def v3_length(v3):
        return np.sqrt(v3[0] ** 2 + v3[1] ** 2 + v3[2] ** 2)

    def subDivide(self):
        self.triangles[:, :, 0] -= self.xc
        self.triangles[:, :, 1] -= self.yc
        self.triangles[:, :, 2] -= self.zc

        subdivided_triangles = []
        for triangle_idx, triangle in enumerate(self.triangles):
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

        self.triangles = np.array(subdivided_triangles)
        self.triangles[:, :, 0] += self.xc
        self.triangles[:, :, 1] += self.yc
        self.triangles[:, :, 2] += self.zc


if __name__ == "__main__":
    icosphere = Icosphere(1.0, 5, 5, 5)
    icosphere.subDivide()
    icosphere.subDivide()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for triangle_idx, triangle in enumerate(icosphere.triangles):
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

    ax.scatter(icosphere.triangles[:, :, 0], icosphere.triangles[:, :, 1], icosphere.triangles[:, :, 2])
    plt.show()
