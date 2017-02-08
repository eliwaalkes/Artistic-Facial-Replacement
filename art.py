from __future__ import print_function
import os, shutil, tempfile, subprocess, glob
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import cv2, openface
from sklearn.mixture import GMM
from skimage import io

faceDetector = "/Users/chilom/Desktop/ml_final/openface/models/dlib/shape_predictor_68_face_landmarks.dat"
faceNN = "/Users/chilom/Desktop/ml_final/openface/models/openface/vgg-face.def.lua"
dir_neuralStyle = "/Users/chilom/Desktop/ml_final/neural-style/"
path_faceImg = "/Users/chilom/Desktop/ml_final/project/face.jpg"
path_artfaceImg = "/Users/chilom/Desktop/ml_final/project/artwork.jpg"
""" path_styleImg = "/opt/neural-style/examples/inputs/starry_night.jpg" """

fname_face = "face.jpg"
fname_artface = "face2.jpg"

class SmoothConvexHull:
    def dist_smooth(self, x, y):
        #Taken from http://stackoverflow.com/questions/14344099/numpy-scipy-smooth-spline-representation-of-an-arbitrary-contour-flength
        nt = np.linspace(0, 1, 100)
        t = np.zeros(x.shape)
        t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        t = np.cumsum(t)
        t /= t[-1]
        x2 = interpolate.spline(t, x, nt)
        y2 = interpolate.spline(t, y, nt)

        return x2, y2
    
    def __init__(self, points):
        sharpHull = ConvexHull(points)
        xs = np.array([point[0] for point in points[sharpHull.vertices]])
        ys = np.array([point[1] for point in points[sharpHull.vertices]])
        xs2, ys2 = self.dist_smooth(xs, ys)
        self.points = np.array([(xs2[i], ys2[i]) for i in range(len(xs2))])
        self.scipyObject = ConvexHull(self.points)
        self.path = Path(self.points[self.scipyObject.vertices])

    def contains(self, point):
        return self.path.contains_point(point)

class FaceDetection:
    def __init__(self, region, hull, size, start, end, backtransform):
        self.region = region
        self.hull = hull
        self.size = size
        self.start = start
        self.end = end
        self.backtransform = backtransform

def loadImage(imgPath, imgDim=(0, 0)):
    rawImg = cv2.imread(imgPath)
    if imgDim != (0, 0):
        rawImg = cv2.resize(rawImg, imgDim)
    else:
        imgDim = (len(rawImg[0]), len(rawImg))
    rgbImg = cv2.cvtColor(rawImg, cv2.COLOR_BGR2RGB)
    return rgbImg, imgDim

def prepareFaceDetection(imgDim):
    global faceNN, faceDetector
    align = openface.AlignDlib(faceDetector)
    return align

def getFaces(img, align):

    """ Find all faces and iterate:
    bb = align.getAllFaceBoundingBoxes(rgbImg)
    """
    FACE_OUTER_BOUNDARY = [i for i in range(27)]
    
    faceBox = align.getLargestFaceBoundingBox(img)
    landmarks = align.findLandmarks(img, faceBox)
    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE
    npLandmarks = np.float32(landmarks)
    npLandmarkIndices = np.array(landmarkIndices)
    H = cv2.getAffineTransform( npLandmarks[npLandmarkIndices],
                                faceBox.height() *
                                openface.align_dlib.MINMAX_TEMPLATE[npLandmarkIndices])
    alignedFace = cv2.warpAffine(img, H, (faceBox.width(), faceBox.height()))
    inverseH = cv2.invertAffineTransform(H)
    alignedFaceVector = inverseH.dot([0, 0, 1]).astype(int)
    hull = SmoothConvexHull(npLandmarks[FACE_OUTER_BOUNDARY])
    print(npLandmarks[FACE_OUTER_BOUNDARY])
    faceDetectionObj = FaceDetection(alignedFace,\
            hull,\
            (int(faceBox.width()), int(faceBox.height())),\
            alignedFaceVector,\
            (alignedFaceVector + np.array([faceBox.width(), faceBox.height()]).astype(int)),\
            inverseH)
    
    return faceDetectionObj

""" -------------------------------------------------------------------------"""

def prepareStyleExtraction(face, artface):
    global fname_face, fname_artface
    dir_tmp = tempfile.mkdtemp()
    os.chdir(dir_tmp)
    io.imsave(dir_tmp+"/"+fname_face, face.region)
    io.imsave(dir_tmp+"/"+fname_artface, artface.region)
    return dir_tmp

def performStyleExtraction(dir, stack_size, imgDim):
    # global path_content, dir_neuralStyle, fname_artface
    # num_iter = 500
    # save_iter = num_iter // stack_size
    # path_output = dir + "/" + "art.jpg"
    # os.chdir(dir_neuralStyle)
    # cmd_th = ["th", "neural_style.lua"]
    # cmd_th += ["-style_image", dir+"/"+fname_artface]
    # cmd_th += ["-content_image", dir+"/"+fname_face]
    # cmd_th += ["-output_image", path_output]
    # cmd_th += ["-num_iterations", str(num_iter)]
    # cmd_th += ["-save_iter", str(save_iter)]
    # cmd_th += ["-gpu", str(-1)]
    # print(" ".join(cmd_th))
    # popen = subprocess.Popen(cmd_th, stdout=subprocess.PIPE, universal_newlines=True)
    # for stdout_line in iter(popen.stdout.readline, ""):
    #     print(stdout_line, end="")
    # popen.stdout.close()
    # return_code = popen.wait()

    return loadImage("/Users/chilom/Desktop/ml_final/project/art.jpg", imgDim)[0]

""" -------------------------------------------------------------------------"""

faceImg, faceImgDim = loadImage(path_faceImg)
artfaceImg, artfaceImgDim = loadImage(path_artfaceImg)

align = prepareFaceDetection(faceImgDim)
face = getFaces(faceImg, align)
artface = getFaces(artfaceImg, align)

dir = prepareStyleExtraction(face, artface)
art_out = performStyleExtraction(dir, 10, artface.size)
#styledface = getFaces(art_out, align)


# finalFace = artface.region
# for y in range(artface.size[1]):
#     for x in range(artface.size[0]):
#         if face.hull.contains((x, y)):
#             finalFace[y, x, :] = art_out[y, x, :]
# finalImg = artfaceImg
warpImg = cv2.warpAffine(art_out, artface.backtransform, artfaceImgDim)
io.imsave("/Users/chilom/Desktop/ml_final/project/test.jpg", warpImg)
# warpImg_gray = cv2.cvtColor(warpImg, cv2.COLOR_RGB2GRAY)
# grayMask = np.not_equal(warpImg_gray, 0)
# structure = np.array([ [0, 1, 0], [1, 1, 1], [0, 1, 0] ])
# erodedMask = ndimage.binary_erosion(grayMask, structure=structure)
# io.imsave("/Users/chilom/Desktop/ml_final/project/binary.jpg", erodedMask)
# rgbMask = np.repeat(np.expand_dims(erodedMask, axis=2), 3, axis=2)
# np.copyto(finalImg, warpImg, where=rgbMask)



# out = "/Users/chilom/Desktop/ml_final/project/artface_out.jpg"
# io.imsave(out, finalImg)

#print("Output saved to {}.".format(out))

""" shutil.rmtree(dir) """
