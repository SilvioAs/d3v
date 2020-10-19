import time

import numpy as np

from bounds import BBox
from rayTracing import Box3DIntersection


class SubDivBoxTree(Box3DIntersection):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        self.facets = []
        self.nodes = []
        self._maxfacets = 1000
        self.name = ""

    def getIntersectedLeafs(self, ray, intrsectLeafs):
        if self.intersectsWithRay(ray):
            if self.isleaf:
                intrsectLeafs.append(self)
            else:
                for node in self.nodes:
                    isInBox, intrsectLeafs = node.getIntersectedLeafs(ray, intrsectLeafs)

        return len(intrsectLeafs) > 0, intrsectLeafs

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
        dx = self.maxCoord[0] - self.minCoord[0]
        dy = self.maxCoord[1] - self.minCoord[1]
        dz = self.maxCoord[2] - self.minCoord[2]
        dmax = max(dx, dy, dz)

        # Copy full bounding box two times and split them half
        sbox1 = self.copy()
        sbox1.name = self.name + "_1"
        sbox2 = self.copy()
        sbox2.name = self.name + "_2"
        if dx == dmax:
            sbox1.maxCoord[0] = (self.maxCoord[0] + self.minCoord[0]) * 0.5
            sbox2.minCoord[0] = sbox1.maxCoord[0]
        elif dy == dmax:
            sbox1.maxCoord[1] = (self.maxCoord[1] + self.minCoord[1]) * 0.5
            sbox2.minCoord[1] = sbox1.maxCoord[1]
        else:
            sbox1.maxCoord[2] = (self.maxCoord[2] + self.minCoord[2]) * 0.5
            sbox2.minCoord[2] = sbox1.maxCoord[2]

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
        cb.setMinCoord(self.minCoord.copy())
        cb.setMaxCoord(self.maxCoord.copy())
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
