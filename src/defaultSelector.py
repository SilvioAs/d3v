from PySide2.QtCore import QObject
from PySide2.QtGui import QVector3D
from geometry import Geometry
from signals import Signals
from selinfo import SelectionInfo
import openmesh as om
import numpy as np
import math
from selection import Selector
from bounds import BBox
import time
from array import array
from multiprocessing import Pool, Process, Queue, Manager
import os
import cProfile


class pyvector3:
    def __init__(self):
        self._vec3 = [0, 0, 0]

    def setFromQt(self, vecQt):
        self._vec3[0] = vecQt.x()
        self._vec3[1] = vecQt.y()
        self._vec3[2] = vecQt.z()

    def setFromNp(self, vecnp):
        self._vec3[0] = vecnp[0]
        self._vec3[1] = vecnp[1]
        self._vec3[2] = vecnp[2]

    def setFromScalars(self, x, y, z):
        self._vec3[0] = x
        self._vec3[1] = y
        self._vec3[2] = z

    def copyFrom(self, template):
        self._vec3[0] = template._vec3[0]
        self._vec3[1] = template._vec3[1]
        self._vec3[2] = template._vec3[2]

    @property
    def X(self):
        return self._vec3[0]

    @X.setter
    def X(self, newX):
        self._vec3[0] = newX

    @property
    def Y(self):
        return self._vec3[1]

    @property
    def XYZ(self):
        return self._vec3

    @Y.setter
    def Y(self, newY):
        self._vec3[1] = newY

    @property
    def Z(self):
        return self._vec3[2]

    @Z.setter
    def Z(self, newZ):
        self._vec3[2] = newZ


class dmnsn_ray:
    def __init__(self, los):
        self._x0 = pyvector3()
        self._x0.setFromQt(los[0])  # P0
        self._n = pyvector3()
        self._n.setFromQt(los[1])  # K

    @property
    def x0(self):
        return self._x0

    @property
    def n(self):
        return self._n


class dmnsn_optimized_ray:
    def __init__(self, ray: dmnsn_ray):
        self._x0 = ray.x0
        self._n_inv = pyvector3()
        self._n_inv.setFromScalars((1.0 / ray.n.X) if ray.n.X != 0 else math.inf,
                                   (1.0 / ray.n.Y) if ray.n.X != 0 else math.inf,
                                   (1.0 / ray.n.Z) if ray.n.Z != 0 else math.inf)

    @property
    def x0(self):
        return self._x0

    @property
    def n_inv(self):
        return self._n_inv


class dmnsn_aabb:
    def __init__(self):
        self._min = pyvector3()
        self._max = pyvector3()

    def setFromBBox(self, box: BBox):
        self._min.setFromNp(box.minCoord)
        self._max.setFromNp(box.maxCoord)

    def dmnsn_ray_box_intersection(self, optray: dmnsn_optimized_ray, t):
        # This is actually correct, even though it appears not to handle edge cases
        # (ray.n.{x,y,z} == 0).  It works because the infinities that result from
        # dividing by zero will still behave correctly in the comparisons.  Rays
        # which are parallel to an axis and outside the box will have tmin == inf
        # or tmax == -inf, while rays inside the box will have tmin and tmax
        # unchanged.

        tx1 = (self.min.X - optray.x0.X) * optray.n_inv.X
        tx2 = (self.max.X - optray.x0.X) * optray.n_inv.X

        tmin = min(tx1, tx2)
        tmax = max(tx1, tx2)

        ty1 = (self.min.Y - optray.x0.Y) * optray.n_inv.Y
        ty2 = (self.max.Y - optray.x0.Y) * optray.n_inv.Y

        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))

        tz1 = (self.min.Z - optray.x0.Z) * optray.n_inv.Z
        tz2 = (self.max.Z - optray.x0.Z) * optray.n_inv.Z

        tmin = max(tmin, min(tz1, tz2))
        tmax = min(tmax, max(tz1, tz2))

        return tmax >= max(0.0, tmin) and tmin < t

    def isIn_array(self, points):
        isIn_bool_array0 = (self.min._vec3[0] < points[:, 0]) & (points[:, 0] < self.max._vec3[0])
        isIn_bool_array1 = (self.min._vec3[1] < points[:, 1]) & (points[:, 1] < self.max._vec3[1])
        isIn_bool_array2 = (self.min._vec3[2] < points[:, 2]) & (points[:, 2] < self.max._vec3[2])
        isIn_bool_array_final = isIn_bool_array0 & isIn_bool_array1 & isIn_bool_array2
        return isIn_bool_array_final

    def isIn(self, point: pyvector3):
        for i in range(3):
            if (self.min._vec3[i] > point._vec3[i]):
                return False
            if (self.max._vec3[i] < point._vec3[i]):
                return False

        return True

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max


class SubDivBoxTree(dmnsn_aabb):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        self.facets = []
        self.nodes = []
        self.node_list = []
        self._maxfacets = 1000
        self.name = ""

    def getIntersectedLeafs(self, optray, t, intrsectLeafs):
        if self.dmnsn_ray_box_intersection(optray, t):
            if self.isleaf:
                intrsectLeafs.append(self)
            else:
                for node in self.nodes:
                    node.getIntersectedLeafs(optray, t, intrsectLeafs)

        return len(intrsectLeafs) > 0

    def new(self):
        cb = SubDivBoxTree(None)
        cb.min.copyFrom(self.min)
        cb.max.copyFrom(self.max)
        return cb

    def createTreeRoot(self, box: BBox):
        if not self.mesh.has_face_normals():
            self.mesh.request_face_normals()
            self.mesh.update_face_normals()

        ar_fv_indices = self.mesh.fv_indices().tolist()
        ar_points = self.mesh.points().tolist()
        self.createTreeRootList(box, ar_fv_indices, ar_points)

    def createTreeRootList(self, box: BBox, fv_indices: [], points: []):
        tsTR = time.perf_counter()
        self.setFromBBox(box)
        self.name = "root"
        nf = len(fv_indices)
        facets = np.array(range(nf))
        self.setFacets(facets)
        fv_indices = np.array(fv_indices)
        points = np.array(points)
        self.createTree(fv_indices, points)
        dtTR = time.perf_counter() - tsTR
        print("Tree creation time, s:", dtTR)
        # self.printTreeInfo()

    def createTree(self, fv_indices: [], points: []):
        if self.numFacets > self._maxfacets:
            self.subdivideOn2New(fv_indices, points)
            for node in self.nodes:
                node.createTree(fv_indices, points)

    def subdivideOn2New(self, fv_indices: [], points: []):
        # determine max deltas of bbox
        dx = self.max.X - self.min.X
        dy = self.max.Y - self.min.Y
        dz = self.max.Z - self.min.Z
        dmax = max(dx, dy, dz)

        # Copy full bounding box two times and split them half
        sbox1 = self.copy()
        sbox1.name = self.name + "_1"
        sbox2 = self.copy()
        sbox2.name = self.name + "_2"
        if dx == dmax:
            sbox1.max.X = (self.max.X + self.min.X) * 0.5
            sbox2.min.X = sbox1.max.X
        elif dy == dmax:
            sbox1.max.Y = (self.max.Y + self.min.Y) * 0.5
            sbox2.min.Y = sbox1.max.Y
        else:
            sbox1.max.Z = (self.max.Z + self.min.Z) * 0.5
            sbox2.min.Z = sbox1.max.Z

        faceCGs = self.calcAllFacetsCG(self.facets, fv_indices, points)

        isIn_sbox1 = sbox1.isIn_array(faceCGs)
        facets_sbox1 = self.facets[isIn_sbox1]
        facets_sbox2 = self.facets[~isIn_sbox1]
        sbox1.setFacets(facets_sbox1)
        sbox2.setFacets(facets_sbox2)

        # Clear the parent bounding box
        self.clearFacets()
        # If the splitted bounding box contains face, append it to nodes -> Starts over in SubDivideOn2
        if sbox1.numFacets > 0:
            self.nodes.append(sbox1)
        if sbox2.numFacets > 0:
            self.nodes.append(sbox2)

    @staticmethod
    def calcAllFacetsCG(face_indices, all_fv_indices, points):
        fv_indices = all_fv_indices[face_indices]
        face_vertices = points[fv_indices]
        faceCGs = face_vertices.sum(axis=1) / 3
        return faceCGs


    """
    Utilitiy functions
    """
    def printTreeInfo(self):
        print(self.name, end="", flush=True)
        if self.isleaf:
            print(", is leaf, ", end="", flush=True)
            print(self.numFacets, end="", flush=True)
            print(" faces.")
        else:
            print(", not leaf.")
        for node in self.nodes:
            node.printTreeInfo()

    def copy(self):
        cb = SubDivBoxTree(self.mesh)
        cb.min.copyFrom(self.min)
        cb.max.copyFrom(self.max)
        return cb

    def addFacet(self, fh):
        self.facets.append(fh)

    def setFacets(self, ifhs):
        self.facets = ifhs

    def clearFacets(self):
        self.facets = np.array([])

    @property
    def isnode(self):
        return len(self.nodes) > 0

    @property
    def isleaf(self):
        return len(self.facets) > 0

    @property
    def numFacets(self):
        return len(self.facets)


class DefaultSelector(Selector):
    def __init__(self):
        super().__init__()

    def select(self, los, geometry):
        tSS = time.perf_counter()
        # self.selectOld(los,geometry)
        cProfile.runctx("self.selectList(los, geometry)", locals(), globals())
        # self.selectList(los, geometry)
        # self.selectNP(los, geometry)

        dtS = time.perf_counter() - tSS
        print("Selection time, s:", dtS)

    def selectList(self, los, geometry):
        if not len(geometry):
            return
        sis = []
        intrsectLeafs = []
        for geo in geometry:
            isInBox = True
            # 1. test bounding box
            t = 99999999
            ray = dmnsn_ray(los)
            opt_ray = dmnsn_optimized_ray(ray)
            intrsectLeafs.clear()
            isInBox = geo.subdivboxtree.getIntersectedLeafs(opt_ray, t, intrsectLeafs)

            points = np.array(geo.mesh.points().tolist())
            # points = geo.mesh.points().tolist()
            fv_indices = np.array(geo.mesh.fv_indices().tolist())
            # fv_indices = geo.mesh.fv_indices().tolist()

            # 2. test mesh in intersected subdivision box tree leafs
            if isInBox:
                for leaf in intrsectLeafs:
                    meshres = self.getMeshInterscectionSDBTNew(ray, leaf.facets, points, fv_indices)
                    # meshres = self.getMeshInterscectionSDBTNew(ray, leaf.facets, geo.mesh)
                    if len(meshres) > 0:
                        si = SelectionInfo()
                        si.update(meshres[0], meshres[1], geo)
                        sis.append(si)

        # selected je selected geometry
        # si je SelectionInfo --> sadrzi podatke o selekciji

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

    def getMeshInterscectionSDBTNew(self, ray: dmnsn_ray, fhlist, points, fv_indices):
        result = []
        # intersectedFacets = []
        # intersectedFacetsDistances = []
        # Find all intersected facets
        infinity = float("inf")

        chosen_fv_indices = fv_indices[fhlist]
        chosen_points = points[chosen_fv_indices]

        ds = self.rayIntersectsTriangleMollerTrumboreSDBT(ray, chosen_points[:, 0], chosen_points[:, 1], chosen_points[:, 2])
        mask = ds != infinity
        intersectedFacets = fhlist[mask]
        intersectedFacetsDistances = ds[mask]

        # for ifh in fhlist:
        #     d = self.rayIntersectsTriangleMollerTrumboreSDBT(ray,
        #                                                      points[fv_indices[ifh][0]],
        #                                                      points[fv_indices[ifh][1]],
        #                                                      points[fv_indices[ifh][2]])
        #     if d != infinity:
        #         intersectedFacets.append(ifh)
        #         intersectedFacetsDistances.append(d)
        # Find the closest point
        ii = -1
        if len(intersectedFacets) > 0:
            ii = 0
        i = 1
        while i < len(intersectedFacets):
            if intersectedFacetsDistances[i] < intersectedFacetsDistances[ii]:
                ii = i
            i = i + 1
        if ii > -1:
            result.append(intersectedFacetsDistances[ii])
            result.append(intersectedFacets[ii])
        return result

    def rayIntersectsTriangleMollerTrumboreSDBT(self, ray: dmnsn_ray, v0, v1, v2):
        # each v is one vertex
        # https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
        # base on  Java Implementation code
        e = 0.00000001
        infinity = float("inf")
        K = ray.n.XYZ
        P0 = ray.x0.XYZ
        edge1 = np.subtract(v1, v0)
        edge2 = np.subtract(v2, v0)
        h = np.cross(K, edge2)
        a = np.sum(edge1 * h, axis=1)
        # a = np.dot(edge1, h.transpose())
        results = np.zeros(len(a))
        results[(-e < a) & (a < e)] = infinity
        # if -e < a < e:
        #     return infinity  # This ray is parallel to this triangle.

        f = 1.0 / a
        s = np.subtract(P0, v0)
        # u = np.multiply(f, np.dot(s, h))
        u = np.multiply(f, np.sum(s * h, axis=1))
        results[(u < 0.0) | (u > 1.0)] = infinity
        # if u < 0.0 or u > 1.0:
        #     return infinity

        q = np.cross(s, edge1)

        # v = f * np.dot(K, q)
        v = f * np.sum(K * q, axis=1)
        results[(v < 0.0) | (u + v > 1.0)] = infinity
        # if v < 0.0 or u + v > 1.0:
        #     return infinity
        # At this stage we can compute t to find out where the intersection point is on the line.

        mask = results != infinity
        t = np.multiply(f, np.sum(edge2 * q, axis=1))
        results[mask] = t[mask]
        # t = np.multiply(f, np.dot(edge2, q))
        return results
        # return t

    def rayIntersectsTriangleMollerTrumboreSDBT_notOptimized(self, ray: dmnsn_ray, v0, v1, v2):
        # https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
        # base on  Java Implementation code
        e = 0.00000001
        infinity = float("inf")
        K = ray.n.XYZ
        P0 = ray.x0.XYZ
        edge1 = np.subtract(v1, v0)
        edge2 = np.subtract(v2, v0)
        h = np.cross(K, edge2)
        a = np.dot(edge1, h)

        if -e < a < e:
            return infinity  # This ray is parallel to this triangle.

        f = 1.0 / a
        s = np.subtract(P0, v0)
        u = np.multiply(f, np.dot(s, h))
        if u < 0.0 or u > 1.0:
            return infinity

        q = np.cross(s, edge1)
        v = f * np.dot(K, q)
        if v < 0.0 or u + v > 1.0:
            return infinity
        # At this stage we can compute t to find out where the intersection point is on the line.
        t = np.multiply(f, np.dot(edge2, q))
        return t
        # if t > e  and t < 1 - e:
        #    return  t
        # else:
        #    return infinity

    def getMeshInterscectionSDBTNew_notOptimized(self, ray: dmnsn_ray, fhlist, mesh: om.TriMesh):
        result = []
        intersectedFacets = []
        intersectedFacetsDistances = []
        # Find all intersected facets
        infinity = float("inf")
        coords = []
        points = mesh.points().tolist()
        fv_indices = mesh.fv_indices().tolist()
        for ifh in fhlist:
            d = self.rayIntersectsTriangleMollerTrumboreSDBT(ray,
                                                             points[fv_indices[ifh][0]],
                                                             points[fv_indices[ifh][1]],
                                                             points[fv_indices[ifh][2]])
            if d != infinity:
                intersectedFacets.append(ifh)
                intersectedFacetsDistances.append(d)
        # Find the closest point
        ii = -1
        if len(intersectedFacets) > 0:
            ii = 0
        i = 1
        while i < len(intersectedFacets):
            if intersectedFacetsDistances[i] < intersectedFacetsDistances[ii]:
                ii = i
            i = i + 1
        if ii > -1:
            result.append(intersectedFacetsDistances[ii])
            result.append(intersectedFacets[ii])
        return result

    def getMeshInterscectionSDBT(self, ray: dmnsn_ray, fhlist, mesh: om.TriMesh):
        result = []
        intersectedFacets = []
        intersectedFacetsDistances = []
        # Find all intersected facets
        infinity = float("inf")
        coords = []
        for fh in fhlist:
            coords.clear()
            for vh in mesh.fv(fh):  # vertex handle
                p = mesh.point(vh)
                coords.append(p)
            v0 = coords[0]
            v1 = coords[1]
            v2 = coords[2]
            d = self.rayIntersectsTriangleMollerTrumboreSDBT(ray, v0, v1, v2)
            if d != infinity:
                intersectedFacets.append(fh)
                intersectedFacetsDistances.append(d)
        # Find the closest point
        ii = -1
        if len(intersectedFacets) > 0:
            ii = 0
        i = 1
        while i < len(intersectedFacets):
            if intersectedFacetsDistances[i] < intersectedFacetsDistances[ii]:
                ii = i
            i = i + 1
        if ii > -1:
            result.append(intersectedFacetsDistances[ii])
            result.append(intersectedFacets[ii])
        return result

    def getMeshInterscection(self, K, P0, mesh: om.TriMesh):
        result = []
        intersectedFacets = []
        intersectedFacetsDistances = []
        # Find all intersected facets
        infinity = float("inf")
        for fh in mesh.faces():
            d = self.rayIntersectsTriangleMollerTrumbore(K, P0, fh, mesh)
            if d != infinity:
                intersectedFacets.append(fh)
                intersectedFacetsDistances.append(d)
        # Find the closest point
        ii = -1
        if len(intersectedFacets) > 0:
            ii = 0
        i = 1
        while i < len(intersectedFacets):
            if intersectedFacetsDistances[i] < intersectedFacetsDistances[ii]:
                ii = i
            i = i + 1
        if ii > -1:
            result.append(intersectedFacetsDistances[ii])
            result.append(intersectedFacets[ii])
        return result

    def selectOld(self, los, geometry):
        if not len(geometry):
            return
        P0Q = los[0]
        KQ = los[1]
        # transform to np arrays
        K = np.array([KQ.x(), KQ.y(), KQ.z()])
        P0 = np.array([P0Q.x(), P0Q.y(), P0Q.z()])

        # geometry je lista geometrije iz koje treba izracunati selekciju
        sis = []
        for geo in geometry:
            isInBox = True
            # 1. test bounding box
            t = 99999999
            box = dmnsn_aabb()
            box.setFromBBox(geo.bbox)
            ray = dmnsn_ray(los)
            opt_ray = dmnsn_optimized_ray(ray)
            isInBox = self.dmnsn_ray_box_intersection(opt_ray, box, t)
            # 2. test mesh in geo
            if isInBox:
                meshres = self.getMeshInterscection(K, P0, geo.mesh)
                if len(meshres) > 0:
                    si = SelectionInfo()
                    si.update(meshres[0], meshres[1], geo)
                    sis.append(si)

        # selected je selected geometry
        # si je SelectionInfo --> sadrzi podatke o selekciji

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

    def rayIntersectsTriangleMollerTrumbore(self, K, P0, face: om.FaceHandle, mesh: om.TriMesh):
        # https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
        # base on  Java Implementation code
        e = 0.00000001
        infinity = float("inf")
        coords = []
        for vh in mesh.fv(face):  # vertex handle
            p = mesh.point(vh)
            coords.append(p)
        v0 = coords[0]
        v1 = coords[1]
        v2 = coords[2]

        edge1 = np.subtract(v1, v0)
        edge2 = np.subtract(v2, v0)
        h = np.cross(K, edge2)
        a = np.dot(edge1, h)

        if -e < a < e:
            return infinity  # This ray is parallel to this triangle.

        f = 1.0 / a
        s = np.subtract(P0, v0)
        u = np.multiply(f, np.dot(s, h))
        if u < 0.0 or u > 1.0:
            return infinity

        q = np.cross(s, edge1)
        v = f * np.dot(K, q)
        if v < 0.0 or u + v > 1.0:
            return infinity
        # At this stage we can compute t to find out where the intersection point is on the line.
        t = np.multiply(f, np.dot(edge2, q))
        return t
        # if t > e  and t < 1 - e:
        #    return  t
        # else:
        #    return infinity

    def dmnsn_ray_box_intersection(self, optray: dmnsn_optimized_ray, box: dmnsn_aabb, t):

        # This is actually correct, even though it appears not to handle edge cases
        # (ray.n.{x,y,z} == 0).  It works because the infinities that result from
        # dividing by zero will still behave correctly in the comparisons.  Rays
        # which are parallel to an axis and outside the box will have tmin == inf
        # or tmax == -inf, while rays inside the box will have tmin and tmax
        # unchanged.

        tx1 = (box.min.X - optray.x0.X) * optray.n_inv.X
        tx2 = (box.max.X - optray.x0.X) * optray.n_inv.X

        tmin = min(tx1, tx2)
        tmax = max(tx1, tx2)

        ty1 = (box.min.Y - optray.x0.Y) * optray.n_inv.Y
        ty2 = (box.max.Y - optray.x0.Y) * optray.n_inv.Y

        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))

        tz1 = (box.min.Z - optray.x0.Z) * optray.n_inv.Z
        tz2 = (box.max.Z - optray.x0.Z) * optray.n_inv.Z

        tmin = max(tmin, min(tz1, tz2))
        tmax = min(tmax, max(tz1, tz2))

        return tmax >= max(0.0, tmin) and tmin < t
