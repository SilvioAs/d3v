from rayTracing import Ray, Box3DIntersection
from signals import Signals
from selinfo import SelectionInfo
import openmesh as om
import numpy as np
from selection import Selector
import time


class DefaultSelector(Selector):
    def __init__(self):
        super().__init__()

    def select(self, los, geometry):
        tSS = time.perf_counter()
        self.selectList(los, geometry)
        dtS = time.perf_counter() - tSS
        print("Selection time, s:", dtS)

    def selectList(self, los, geometry):
        if not len(geometry):
            return
        sis = []
        for geo in geometry:
            # 1. test bounding box
            ray = Ray([los[0].x(), los[0].y(), los[0].z()], [los[1].x(), los[1].y(), los[1].z()])
            intrsectLeafs = []
            isInBox, intrsectLeafs = geo.subdivboxtree.getIntersectedLeafs(ray, intrsectLeafs)

            points = geo.mesh.points()
            fv_indices = geo.mesh.fv_indices()

            # 2. test mesh in intersected subdivision box tree leafs
            if isInBox:
                for leaf in intrsectLeafs:
                    meshres = self.getMeshInterscection(ray, leaf.facets, points, fv_indices)
                    # meshres = self.getMeshInterscectionSDBTNew(ray, leaf.facets, geo.mesh)
                    if len(meshres) > 0:
                        si = SelectionInfo()
                        si.update(meshres[0], meshres[1], geo)
                        sis.append(si)

        # Looks for geometry with shortest distance and gives it to
        if len(sis) > 0:
            si = sis[0]
            i = 1
            while i < len(sis):
                if sis[i].getDistance() < si.getDistance():
                    si = sis[i]
                i = i + 1
            # nakon sto je selekcija odradjena
            # fill in sve podatke u SelectionInfo object
            # selected je selekcionirana geometrija
            selected = si.getGeometry()
            selected.onSelected(si)

        else:
            selected = None
            si = SelectionInfo()
        # obavijesti sve zainteresirane da je selekcija promijenjena
        Signals.get().selectionChanged.emit(si)

    def getMeshInterscection(self, ray: Ray, fhlist, points, fv_indices):
        infinity = float("inf")

        chosen_fv_indices = fv_indices[fhlist]
        chosen_points = points[chosen_fv_indices]

        ds = self.rayIntersectsTriangleMollerTrumboreSDBT(ray, chosen_points[:, 0], chosen_points[:, 1], chosen_points[:, 2])
        mask = ds != infinity
        intersectedFacets = fhlist[mask]
        intersectedFacetsDistances = ds[mask]

        if len(intersectedFacets) == 0:
            return []

        idx_min = np.argmin(intersectedFacetsDistances)
        result = [intersectedFacetsDistances[idx_min], intersectedFacets[idx_min]]
        return result

    def rayIntersectsTriangleMollerTrumboreSDBT(self, ray: Ray, v0, v1, v2):
        # https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
        # base on  Java Implementation code
        e = 0.00000001
        infinity = float("inf")
        K = ray.d
        P0 = ray.o
        edge1 = np.subtract(v1, v0)
        edge2 = np.subtract(v2, v0)
        h = np.cross(K, edge2)
        a = np.sum(edge1 * h, axis=1)
        results = np.zeros(len(a))
        results[(-e < a) & (a < e)] = infinity

        f = 1.0 / a
        s = np.subtract(P0, v0)
        u = np.multiply(f, np.sum(s * h, axis=1))
        results[(u < 0.0) | (u > 1.0)] = infinity

        q = np.cross(s, edge1)
        v = f * np.sum(K * q, axis=1)
        results[(v < 0.0) | (u + v > 1.0)] = infinity

        t = np.multiply(f, np.sum(edge2 * q, axis=1))
        mask = results != infinity
        results[mask] = t[mask]
        return results
