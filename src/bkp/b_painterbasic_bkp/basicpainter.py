from PySide2.QtCore import Slot
from PySide2.QtGui import QOpenGLShaderProgram, QOpenGLShader
from PySide2.QtGui import QOpenGLVersionProfile, QOpenGLContext
from PySide2.QtGui import QSurfaceFormat
from PySide2.QtWidgets import QMessageBox
from painters import Painter
from signals import Signals, DragInfo
from glvertdatasforhaders import VertDataCollectorCoord3fNormal3fColor4f
from glhelp import GLEntityType
from OpenGL import GL
from PySide2.QtCore import QCoreApplication
from geometry import Geometry
import openmesh as om
import numpy as np
from selinfo import SelectionInfo
from PySide2.QtGui import QBrush, QPainter, QPen, QPolygon, QColor, QFont
from PySide2.QtCore import QRect, Qt
from PySide2.QtWidgets import QApplication
import os
import time
from multiprocessing.pool import ThreadPool as Pool
# from multiprocessing import Pool


class BasicPainter(Painter):
    def __init__(self):
        super().__init__()
        self._dentsvertsdata = {}  # dictionary that holds vertex data for all primitive and  submodel combinations
        self._geo2Add = []
        self._geo2Rebuild = []
        self._geo2Remove = []
        self._doSelection = False
        self._si = SelectionInfo()
        self.program = 0
        self.projMatrixLoc = 0
        self.mvMatrixLoc = 0
        self.normalMatrixLoc = 0
        self.lightPosLoc = 0
        # self.vertexShader = self.vertexShaderSourceCore()
        # self.fragmentShader = self.fragmentShaderSourceCore()
        self.vertexShader = self.vertexShaderSource()
        self.fragmentShader = self.fragmentShaderSource()
        # model / geometry
        self.addGeoCount = 0
        Signals.get().selectionChanged.connect(self.onSelected)
        self.paintDevice = 0
        self.selType = 0  # 0 - geometry
        # self.selType = 1  # 1 - facet
        self._showBack = False
        self._multFactor = 1
        self.showBack = True

    @property
    def showBack(self):
        return self._showBack

    @showBack.setter
    def showBack(self, newShowBack):
        self._showBack = newShowBack
        self._multFactor = 1
        if self._showBack:
            self._multFactor = 2

    def initializeGL(self):
        paintDevice = QApplication.instance().mainFrame.glWin
        self.width = paintDevice.vport.width()
        self.height = paintDevice.vport.height()
        super().initializeGL()
        self.program = QOpenGLShaderProgram()
        # profile = QOpenGLVersionProfile()
        # profile.setVersion(2, 0)
        # context = QOpenGLContext.currentContext()
        # print("paintr init "+str(context))
        # self.glf = context.versionFunctions(profile)
        # if not self.glf:
        #     QMessageBox.critical(None, "Failed to Initialize OpenGL",
        #                          "Could not initialize OpenGL. This program requires OpenGL x.x or higher. Please check your video card drivers.")
        self.glf.initializeOpenGLFunctions()
        self.glf.glClearColor(0.0, 0.0, 0.0, 1)
        self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, self.vertexShader)
        self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, self.fragmentShader)
        self.program.link()
        self.program.bind()
        self.projMatrixLoc = self.program.uniformLocation("projMatrix")
        self.mvMatrixLoc = self.program.uniformLocation("mvMatrix")
        self.normalMatrixLoc = self.program.uniformLocation("normalMatrix")
        self.lightPosLoc = self.program.uniformLocation("lightPos")
        self.program.release()

    def setprogramvalues(self, proj, mv, normalMatrix, lightpos):
        self.program.bind()
        self.program.setUniformValue(self.lightPosLoc, lightpos)
        self.program.setUniformValue(self.projMatrixLoc, proj)
        self.program.setUniformValue(self.mvMatrixLoc, mv)
        self.program.setUniformValue(self.normalMatrixLoc, normalMatrix)
        self.program.release()

    def paintGL(self):
        super().paintGL()
        self.glf.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.glf.glEnable(GL.GL_DEPTH_TEST)
        self.glf.glEnable(GL.GL_CULL_FACE)
        # self.glf.glDisable(GL.GL_CULL_FACE)
        self.program.bind()
        for key, value in self._dentsvertsdata.items():
            value.drawvao(self.glf)
        self.program.release()

    def resizeGL(self, w: int, h: int):
        super().resizeGL(w, h)

    def updateGL(self):
        super().updateGL()
        self.updateGeometry()

    def resetmodel(self):
        """!
        Reset the model

        Cleans the dictionary
        """
        for key, value in self._dentsvertsdata.items():
            value.free()
        self._dentsvertsdata.clear()

    def removeDictItem(self, key):
        """!
        Reset the item

        Cleans the dictionary
        """
        if key in self._dentsvertsdata:
            self._dentsvertsdata[key].free()
            self._dentsvertsdata.pop(key, None)

    def initnewdictitem(self, key, enttype):
        """!
        Initialize a new dictionary item that holds data for rendering
        @param key: (\b str) item key
        @param enttype: (GLEntityType) primitive drawing entity type
        @retval None
        """

        self._dentsvertsdata[key] = VertDataCollectorCoord3fNormal3fColor4f(enttype)

    def appendlistdata_f3xyzf3nf4rgba(self, key, x, y, z, nx, ny, nz, r, g, b, a):
        """!
        Append Vertex collector dictionary item with new vertex data
        @param key: (\b str) dictonary key
        @param x: (\b float) x coordinate
        @param y: (\b float) y coordinate
        @param z: (\b float) z coordinate
        @param nx: (\b float) x normal coordinate
        @param ny: (\b float) y normal coordinate
        @param nz: (\b float) z normal coordinate
        @retval: (\b int) index of the added vertex
        """
        return self._dentsvertsdata[key].appendlistdata_f3xyzf3nf4rgba(x, y, z, nx, ny, nz, r, g, b, a)

    def appenddictitemsize(self, key, numents):
        """!
        Append dictionary item size with the specified number of entities
        :@param key:(str) key
        :@param numents:(\b int) number of entities to be added
        """
        self._dentsvertsdata[key].appendsize(numents * self._multFactor)

    def allocatememory(self):
        """!
        Allocate memory for all dictionary items that holds data for rendering

        Allocation size is based on the information collected by client calls to appenddictitemsize()
        """

        for key, value in self._dentsvertsdata.items():
            value.allocatememory()

    def allocatememory(self, key):
        """!
        Allocate memory for all dictionary items that holds data for rendering

        Allocation size is based on the information collected by client calls to appenddictitemsize()
        """
        self._dentsvertsdata[key].allocatememory()

    def bindData(self, key):
        self._dentsvertsdata[key].setupVertexAttribs(self.glf)
        atrList = self._dentsvertsdata[key].GetAtrList()
        for ent in atrList:
            self.program.bindAttributeLocation(ent[0], ent[1])

    # Shader code ********************************************************
    def vertexShaderSourceCore(self):
        return """#version 150
                in vec4 vertex;
                in vec3 normal;
                out vec3 vert;
                out vec3 vertNormal;
                out vec4 colorV;
                uniform mat4 projMatrix;
                uniform mat4 mvMatrix;
                uniform mat3 normalMatrix;
                void main() {
                   vert = vertex.xyz;
                   vertNormal = normalMatrix * normal;
                   gl_Position = projMatrix * mvMatrix * vertex;
                   colorV = color;
                }"""

    def fragmentShaderSourceCore(self):
        return """#version 150
                in highp vec3 vert;
                in highp vec3 vertNormal;
                in highp vec4 colorV; 
                out highp vec4 fragColor;
                uniform highp vec3 lightPos;
                void main() {
                   highp vec3 L = normalize(lightPos - vert);
                   highp float NL = max(dot(normalize(vertNormal), L), 0.0);
                   highp vec3 col = clamp(colorV.rgb * 0.2 + colorV.rgb * 0.8 * NL, 0.0, 1.0);
                   fragColor = vec4(col, colorV.a);
                }"""

    def vertexShaderSource(self):
        return """attribute vec4 vertex;
                attribute vec3 normal;
                attribute vec4 color;
                varying vec3 vert;
                varying vec3 vertNormal;
                varying vec4 colorV;
                uniform mat4 projMatrix;
                uniform mat4 mvMatrix;
                uniform mat3 normalMatrix;
                void main() {
                   vert = vertex.xyz;
                   vertNormal = normalMatrix * normal;
                   gl_Position = projMatrix * mvMatrix * vertex;
                   colorV = color;
                }"""

    def fragmentShaderSource(self):
        return """varying highp vec3 vert;
                varying highp vec3 vertNormal;
                varying highp vec4 colorV; 
                uniform highp vec3 lightPos;
                void main() {
                   highp vec3 L = normalize(lightPos - vert);
                   highp float NL = max(dot(normalize(vertNormal), L), 0.0);
                   highp vec3 col = clamp(colorV.rgb * 0.2 + colorV.rgb * 0.8 * NL, 0.0, 1.0);
                   gl_FragColor = vec4(col, colorV.a);
                }"""

    # Painter methods implementation code ********************************************************

    def addGeometry(self, geometry: Geometry):
        self._geo2Add.append(geometry)
        self.requestGLUpdate()

    def removeGeometry(self, geometry: Geometry):
        self._geo2Remove.append(geometry)
        self.requestGLUpdate()
        pass

    def rebuildGeometry(self, geometry: Geometry):
        self._geo2Rebuild.append(geometry)
        self.requestGLUpdate()
        pass

    def delayedAddGeometry(self, geometry: Geometry):
        # tsAG = time.perf_counter()
        self.addGeoCount = self.addGeoCount + 1
        key = geometry.guid
        # self.resetmodel()
        self.initnewdictitem(key, GLEntityType.TRIA)
        nf = geometry.mesh.n_faces()
        self.appenddictitemsize(key, nf)
        self.allocatememory(key)
        # tsAG1 = time.perf_counter()
        self.addMeshdata4oglmdl(key, geometry)
        # dtAG1 = time.perf_counter() - tsAG1
        self.bindData(key)

        # dtAG = time.perf_counter() - tsAG
        # print("Add geometry time, s:", dtAG)
        # print("addMeshdata4oglmdl time, s:", dtAG)

    def delayedRebuildGeometry(self, geometry: Geometry):
        key = geometry.guid
        self.removeDictItem(key)
        self.initnewdictitem(key, GLEntityType.TRIA)
        nf = geometry.mesh.n_faces()
        self.appenddictitemsize(key, nf)
        self.allocatememory(key)
        self.addMeshdata4oglmdl(key, geometry)
        self.bindData(key)

    def delayedRemoveGeometry(self, geometry: Geometry):
        key = geometry.guid
        self.removeDictItem(key)

    def addSelection(self):
        if self.selType == 0:
            pass
        else:
            key = 0
            self.removeDictItem(key)
            if self._si.haveSelection():
                self.initnewdictitem(key, GLEntityType.TRIA)
                nf = self._si.nFaces() * 2
                self.appenddictitemsize(key, nf)
                self.allocatememory(key)
                self.addSelData4oglmdl(key, self._si, self._si.geometry)
                self.bindData(key)

    def updateGeometry(self):
        if len(self._geo2Remove) > 0:
            for geometry in self._geo2Remove:
                self.delayedRemoveGeometry(geometry)
            self._geo2Remove.clear()
        if len(self._geo2Add) > 0:
            for geometry in self._geo2Add:
                self.delayedAddGeometry(geometry)
            self._geo2Add.clear()
        if len(self._geo2Rebuild) > 0:
            for geometry in self._geo2Rebuild:
                self.delayedRebuildGeometry(geometry)
            self._geo2Rebuild.clear()
        if self._doSelection:
            self.addSelection()
            self._doSelection = False

    def addSelData4oglmdl(self, key, si, geometry):
        mesh = geometry.mesh
        for fh in si.allfaces:
            n = mesh.normal(fh)
            c = [1.0, 0.0, 1.0, 1.0]
            for vh in mesh.fv(fh):  # vertex handle
                p = mesh.point(vh)
                self.appendlistdata_f3xyzf3nf4rgba(key,
                                                   p[0] + n[0] / 100, p[1] + n[1] / 100, p[2] + n[2] / 100,
                                                   n[0], n[1], n[2],
                                                   c[0], c[1], c[2], c[3])
            for vh in mesh.fv(fh):  # vertex handle
                p = mesh.point(vh)
                self.appendlistdata_f3xyzf3nf4rgba(key,
                                                   p[0] - n[0] / 100, p[1] - n[1] / 100, p[2] - n[2] / 100,
                                                   n[0], n[1], n[2],
                                                   c[0], c[1], c[2], c[3])
        return

    def addOneFace(self, key, fv, ifh, cstype, c, ar_points, ar_face_normals, ar_face_colors, ar_vertex_colors):
        pp = []
        cc = []
        nn = []
        n = ar_face_normals[ifh]
        if cstype == 1:
            c = ar_face_colors[ifh]

        for iv in fv:
            p = ar_points[iv]
            if cstype == 2:
                c = ar_vertex_colors[iv]

            if self._showBack:
                pp.append(p)
                nn.append(n)
                cc.append(c)

            self.appendlistdata_f3xyzf3nf4rgba(key,
                                               p[0], p[1], p[2],
                                               n[0], n[1], n[2],
                                               c[0], c[1], c[2], c[3])

        if self._showBack:
            nv = len(pp)
            for iv in range(nv):
                ivi = nv - 1 - iv
                self.appendlistdata_f3xyzf3nf4rgba(key,
                                                   pp[ivi][0], pp[ivi][1], pp[ivi][2],
                                                   -nn[ivi][0], -nn[ivi][1], -nn[ivi][2],
                                                   cc[ivi][0], cc[ivi][1], cc[ivi][2], cc[ivi][3])

    def addMeshdata4oglmdl(self, key, geometry):
        tsAMD = time.perf_counter()
        mesh = geometry.mesh

        # color data
        cstype = 0  # color source type
        useMeshColor = True
        ar_face_colors, ar_vertex_colors = None, None
        if self.selType == 0:
            if self._si.geometry.guid == geometry.guid:
                c = [1.0, 0.0, 1.0, 1.0]
                useMeshColor = False
            else:
                c = [0.4, 1.0, 1.0, 1.0]  # default color
        elif useMeshColor and mesh.has_face_colors():
            ar_face_colors = mesh.face_colors()
            cstype = 1
        elif useMeshColor and mesh.has_vertex_colors():
            ar_vertex_colors = mesh.vertex_colors()
            cstype = 2
        else:
            c = [0.4, 1.0, 1.0, 1.0]  # default color

        # normals data
        if not mesh.has_face_normals():  # normals are necessary for correct lighting effect
            mesh.request_face_normals()
            mesh.update_face_normals();
        ar_face_normals = mesh.face_normals()

        ar_fv_indices = mesh.fv_indices().tolist()
        ar_points = mesh.points().tolist()

        for ifh, fv in enumerate(ar_fv_indices):
            self.addOneFace(key, fv, ifh, cstype, c, ar_points, ar_face_normals, ar_face_colors, ar_vertex_colors)

        dtAMD = time.perf_counter() - tsAMD
        print("Add mesh data total:", dtAMD)
        return


    def addMeshdata4oglmdl_bkp_silvio(self, key, geometry):
        tsAMD = time.perf_counter()
        mesh = geometry.mesh
        ar_fv_indices = mesh.fv_indices().tolist()
        ar_points = mesh.points().tolist()

        # color data
        cstype = 0  # color source type
        useMeshColor = True
        if self.selType == 0:
            if self._si.geometry.guid == geometry.guid:
                c = [1.0, 0.0, 1.0, 1.0]
                useMeshColor = False
            else:
                c = [0.4, 1.0, 1.0, 1.0]  # default color
        elif useMeshColor and mesh.has_face_colors():
            ar_face_colors = mesh.face_colors()
            cstype = 1
        elif useMeshColor and mesh.has_vertex_colors():
            ar_vertex_colors = mesh.vertex_colors()
            cstype = 2
        else:
            c = [0.4, 1.0, 1.0, 1.0]  # default color

        # normals data
        if not mesh.has_face_normals():  # normals are necessary for correct lighting effect
            mesh.request_face_normals()
            mesh.update_face_normals();
        ar_face_normals = mesh.face_normals()

        nf = mesh.n_faces()

        ifh = 0
        for ifh in range(nf):
            fv = ar_fv_indices[ifh]
            pp = []
            cc = []
            nn = []
            n = ar_face_normals[ifh]
            if cstype == 1:
                c = ar_face_colors[ifh]

            for iv in fv:
                p = ar_points[iv]
                if cstype == 2:
                    c = ar_vertex_colors[iv]

                if self._showBack:
                    pp.append(p)
                    nn.append(n)
                    cc.append(c)

                self.appendlistdata_f3xyzf3nf4rgba(key,
                                                   p[0], p[1], p[2],
                                                   n[0], n[1], n[2],
                                                   c[0], c[1], c[2], c[3])

            if self._showBack:
                nv = len(pp)
                for iv in range(nv):
                    ivi = nv - 1 - iv
                    self.appendlistdata_f3xyzf3nf4rgba(key,
                                                       pp[ivi][0], pp[ivi][1], pp[ivi][2],
                                                       -nn[ivi][0], -nn[ivi][1], -nn[ivi][2],
                                                       cc[ivi][0], cc[ivi][1], cc[ivi][2], cc[ivi][3])
        dtAMD = time.perf_counter() - tsAMD
        print("Add mesh data total:", dtAMD)
        return

        for fh in mesh.faces():
            pp = []
            cc = []
            nn = []

            n = mesh.normal(fh)

            if useMeshColor and mesh.has_face_colors():
                c = mesh.color(fh)
            for vh in mesh.fv(fh):  # vertex handle
                vit = mesh.vv(vh)  # iterator
                p = mesh.point(vh)
                if useMeshColor and mesh.has_vertex_colors():
                    c = mesh.color(vh)
                iv = 0
                if self._showBack:
                    pp.append(p)
                    nn.append(n)
                    cc.append(c)
                self.appendlistdata_f3xyzf3nf4rgba(key,
                                                   p[0], p[1], p[2],
                                                   n[0], n[1], n[2],
                                                   c[0], c[1], c[2], c[3])
            if self._showBack:
                nv = len(pp)
                for iv in range(nv):
                    ivi = nv - 1 - iv
                    self.appendlistdata_f3xyzf3nf4rgba(key,
                                                       pp[ivi][0], pp[ivi][1], pp[ivi][2],
                                                       -nn[ivi][0], -nn[ivi][1], -nn[ivi][2],
                                                       cc[ivi][0], cc[ivi][1], cc[ivi][2], cc[ivi][3])

        return

    def addMeshdata4oglmdl_bkp(self, key, geometry):
        isGeometrySelected = not self._si.isEmpty()
        if isGeometrySelected:
            isGeometrySelected = self._si.geometry is geometry
        mesh = geometry.mesh
        if not mesh.has_face_normals():  # normals are necessary for correct lighting effect
            mesh.request_face_normals()
            mesh.update_face_normals();
        nf = mesh.n_faces()
        verts = mesh.vertices()
        if not mesh.has_face_colors() and not mesh.has_vertex_colors():
            c = [0.4, 1.0, 1.0, 1.0]  # default color
        for fh in mesh.faces():
            n = mesh.normal(fh)
            if isGeometrySelected and self._si.getFace() is fh:
                c1 = [1.0, 0.0, 1.0, 1.0]
                for vh in mesh.fv(fh):  # vertex handle
                    vit = mesh.vv(vh)  # iterator
                    p = mesh.point(vh)
                    if mesh.has_vertex_colors():
                        c = mesh.color(vh)
                    iv = 0
                    self.appendlistdata_f3xyzf3nf4rgba(key,
                                                       p[0], p[1], p[2],
                                                       n[0], n[1], n[2],
                                                       c1[0], c1[1], c1[2], c1[3])
                isGeometrySelected = False
            else:
                if mesh.has_face_colors():
                    c = mesh.color(fh)
                for vh in mesh.fv(fh):  # vertex handle
                    vit = mesh.vv(vh)  # iterator
                    p = mesh.point(vh)
                    if mesh.has_vertex_colors():
                        c = mesh.color(vh)
                    iv = 0
                    self.appendlistdata_f3xyzf3nf4rgba(key,
                                                       p[0], p[1], p[2],
                                                       n[0], n[1], n[2],
                                                       c[0], c[1], c[2], c[3])
        return

    @Slot()
    def onSelected(self, si: SelectionInfo):
        if self.selType == 0:  # whole geometry selection
            if self._si.haveSelection() and si.haveSelection():
                if self._si.geometry._guid != si.geometry._guid:
                    self._geo2Remove.append(si.geometry)
                    self._geo2Remove.append(self._si.geometry)
                    self._geo2Add.append(self._si.geometry)
                    self._geo2Add.append(si.geometry)
                    self.requestGLUpdate()
            elif si.haveSelection():
                self._geo2Remove.append(si.geometry)
                self._geo2Add.append(si.geometry)
                self.requestGLUpdate()
            elif self._si.haveSelection():
                self._geo2Remove.append(self._si.geometry)
                self._geo2Add.append(self._si.geometry)
                self.requestGLUpdate()
            self._si = si
        else:
            self._doSelection = True
            self._si = si
            self.requestGLUpdate()
        pass
