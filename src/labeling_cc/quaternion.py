# -*- coding: utf-8 -*-
import math
class Quaternion():
    def __init__(self, iw = 1.0, ix = 0.0, iy = 0.0, iz = 0.0):
        self.w = iw
        self.x = ix
        self.y = iy
        self.z = iz
    def __mul__(self, p):
        q = Quaternion()
        q.x = self.w * p.x - self.z * p.y + self.y * p.z + self.x * p.w
        q.y = self.z * p.x + self.w * p.y - self.x * p.z + self.y * p.w
        q.z =-self.y * p.x + self.x * p.y + self.w * p.z + self.z * p.w
        q.w =-self.x * p.x - self.y * p.y - self.z * p.z + self.w * p.w
        return q
    def inv(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    def toMat(self):
        xx = self.x * self.x * 2.0
        yy = self.y * self.y * 2.0
        zz = self.z * self.z * 2.0
        xy = self.x * self.y * 2.0
        yz = self.y * self.z * 2.0
        zx = self.z * self.x * 2.0
        xw = self.x * self.w * 2.0
        yw = self.y * self.w * 2.0
        zw = self.z * self.w * 2.0
        return [
            [1.0 - yy - zz, xy + zw, zx - yw, 0.0],
            [xy - zw, 1.0 - zz - xx, yz + xw, 0.0],
            [zx + yw, yz - xw, 1.0 - xx - yy, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    def offset(self, o):
        return Quaternion(self.w, self.x + o[0], self.y + o[1], self.z + o[2])
    def copyTo(self, q):
        q.w = self.w
        q.x = self.x
        q.y = self.y
        q.z = self.z
    def rotX(theta):
        return Quaternion(math.cos(theta / 2.0), math.sin(theta / 2.0), 0.0, 0.0)
    def rotY(theta):
        return Quaternion(math.cos(theta / 2.0), 0.0, math.sin(theta / 2.0), 0.0)
    def rotZ(theta):
        return Quaternion(math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0))
    def rotEulerYXZ(ry, rx, rz):
        return Quaternion.rotZ(rz) * Quaternion.rotX(rx) * Quaternion.rotY(ry)
    def rotEuler(rx, ry, rz):
        return Quaternion.rotZ(rz) * Quaternion.rotY(ry) * Quaternion.rotX(rx)
    def rotAxis(axis, rot):
        return Quaternion(math.cos(rot / 2.0), math.sin(rot / 2.0) * axis[0], math.sin(rot / 2.0) * axis[1], math.sin(rot / 2.0) * axis[2])
