
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import sys
from quaternion import *
from tester import *
from labeling import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
#from OpenGL.GLUT.freeglut import *

width_ = 800
height_ = 800

mouse_x_ = 0
mouse_y_ = 0
scale_ = 0.002

select_id_ = 2
target_ = Quaternion()
current_ = Quaternion()
rotate_ = [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]

char_voxel = []
labels = []
vcs = []
vtx = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
vl = 3

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE | GLUT_DEPTH)
    glutInitWindowSize(width_, height_)     # window size
    glutInitWindowPosition(100, 100) # window position
    glutCreateWindow("tester")      # show window
    glutDisplayFunc(display)         # draw callback function
    glutReshapeFunc(reshape)         # resize callback function
    glutMouseFunc(mouse)
    glutMotionFunc(mouseMove)
    glutKeyboardFunc(keyBoard)
    #glutMouseWheelFunc(mouseWheel)
    #glutSpecialFunc(specialKey)
    glutIdleFunc(idle)
    #glutTimerFunc(1000, loadModel, 0)
    init(width_, height_)
    glutMainLoop()
def init(width, height):
    """ initialize """
    glClearColor(0.0, 0.0, 0.0, 1.0)
    #glEnable(GL_DEPTH_TEST) # enable shading

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    ##set perspective
    gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)

def display():
    """ display """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    ##set camera
    gluLookAt(-1.0, 0.0, 0., 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    glMultMatrixd(sum(rotate_, []))
    glMultMatrixd([scale_, 0, 0, 0, 0, scale_, 0, 0, 0, 0, scale_, 0, 0, 0.0, 0, 1])
    glColor3f(1.0, 0.0, 0.0)

    glEnableClientState( GL_VERTEX_ARRAY )
    glVertexPointer( 3, GL_FLOAT, 0, vtx )
    glEnableClientState( GL_COLOR_ARRAY )
    glColorPointer( 3, GL_FLOAT, 0, vcs )
    glDrawArrays( GL_POINTS, 0, vl )

    glFlush()  # enforce OpenGL command

def reshape(width, height):
    """callback function resize window"""
    width_ = width
    height_ = height
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
def idle():
    glutPostRedisplay()
def keyBoard(key, x, y):
    print(key)
    global scale_
    global select_id_
    if key == b'a':
        scale_ *= 1.1
    if key == b's':
        scale_ *= 0.9
    if key == b'z':
        select_id_ -= 1
        set_voxels()
    if key == b'x':
        select_id_ += 1
        set_voxels()

def mouseMove(x, y):
    dx = (x - mouse_x_) * 3.14 / width_
    dy = (y - mouse_y_) * 3.14 / height_
    length = math.sqrt(dx*dx+dy*dy)
    if length > 0.0:
        s = math.sin(length) / length
        after = Quaternion(math.cos(length), -dy*s, 0.0, dx*s)
        global target_
        q = after * current_
        q.copyTo(target_)
        listCopy(rotate_, target_.toMat())
def mouse(button, state, x, y):
    if state==GLUT_DOWN:
        global mouse_x_
        global mouse_y_
        mouse_x_ = x
        mouse_y_ = y
    elif state == GLUT_UP:
        global current_
        target_.copyTo(current_)

def mouseWheel(wheel_number, direction, x, y):
    global scale_
    if direction == 1:
        scale_ = scale_ * 1.1
    else:
        scale_ /= 1.1
def listCopy(l1, l2):
    for i in range(4):
        for j in range(4):
            l1[i][j] = l2[i][j]
def set_voxel_test():
    global vtx
    global vl
    vl=10
    vtx = np.random.rand(3, 10)

def set_voxels():
    print("select voxel {}".format(select_id_))
    print("voxel data loading...")
    global vtx
    global vcs
    global vl
    voxel_size = labels.shape
    target = np.array(np.nonzero(labels == select_id_)).transpose()
    vl = target.shape[0]
    vtx = target.astype(np.float32)
    vtx[:,0] -= 180
    vtx[:,1] -= 137.5
    vtx[:,2] -= 135.5

    vcs = np.zeros((vl, 3), np.float32)

    normals = get_normals(char_voxel, target)
    vcs = normals*0.5+0.5
    print("voxel load finished. length: {}".format(vl))

if __name__ == "__main__":
    char_voxel = create_downsampled_voxel_sample()
    #bool_boxel = create_bool_voxel_d2(char_voxel)
    #labels = labeling(char_voxel, bool_boxel)
    bool_voxel = create_th_voxel(char_voxel)
    labels = labeling_bool(bool_voxel)
    select_id_ = np.max(labels)
    set_voxels()
    main()
