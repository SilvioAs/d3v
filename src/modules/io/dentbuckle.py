from iohandlers import IOHandler
from signals import Signals
from geometry import Geometry
import openmesh as om
import os
import numpy as np
import csv

class DentBuckleImporter(IOHandler):
    def __init__(self):
        super().__init__()

    def importGeometry(self, fileName):
        if len(fileName) < 1:
            return
        filename, file_extension = os.path.splitext(fileName)
        if file_extension != ".debu":
            return
        dbb=DentBuckle(fileName)
        g = Geometry()
        g.mesh = dbb.getmesh()
        Signals.get().geometryImported.emit(g)

    def getImportFormats(self):
        return (".debu")


def createIOHandler():
    return DentBuckleImporter()

class DentBuckle ():
    def __init__(self,fileName):
        self.filename = fileName
    def getmesh(self):
        #m=self.test()
        m=self.dentbucklemesh()
        return m
    def test(self):
        mesh= om.TriMesh()
        vhandle = [0]*5
        data = np.array([0, 1, 0])
        vhandle[0] = mesh.add_vertex(data)
        #vhandle.append(mesh.add_vertex(data))
        data = np.array([1, 0, 0])
        vhandle[1] = mesh.add_vertex(data)
        data = np.array([2, 1, 0])
        vhandle[2] = mesh.add_vertex(data)
        data = np.array([0, -1, 0])
        vhandle[3] = mesh.add_vertex(data)
        data = np.array([2, -1, 0])
        vhandle[4] = mesh.add_vertex(data)

        fh0 = mesh.add_face(vhandle[0], vhandle[1], vhandle[2])
        fh1 = mesh.add_face(vhandle[1], vhandle[3], vhandle[4])
        fh2 = mesh.add_face(vhandle[0], vhandle[3], vhandle[1])

        vh_list = [vhandle[2], vhandle[1], vhandle[4]]
        fh3 = mesh.add_face(vh_list)
        return mesh
        pass
    def dentbucklemesh(self):
        stlpath=""
        with open(self.filename, newline='') as csvfile:
            i=0
            hfr = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for row in hfr:
                if i== 0:
                    stlpath=row[0]
                #print(', '.join(row))
        if stlpath != "":
            abspath='\\'.join(self.filename.split('\\')[0:-1])
            abspath=abspath+'\\'+stlpath
            mesh= om.read_trimesh(abspath)
        #read self.filename
        return mesh
        pass


