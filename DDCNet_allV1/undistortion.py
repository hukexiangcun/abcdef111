import cv2
import numpy as np
import glob

def image_undistortion(img):
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  h,w = img.shape[:2]
  
  objp = np.zeros((w*h,3), np.float32)
  objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
  
  objpoints = [] 
  imgpoints = [] 
#  
#  images =  glob.glob('/root/data/datasets/VOC/VOCdevkit/VOC2012/JPEGImages_my/*.jpg')
#  #img = images
#  for fname in images:
#    img = cv2.imread(fname)
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    # �ҵ����̸�ǵ�
#    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
#    # ����ҵ��㹻��ԣ�����洢����
#    if ret == True:
#        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#        objpoints.append(objp)
#        imgpoints.append(corners)
#        # ���ǵ���ͼ������ʾ
#        cv2.drawChessboardCorners(img, (w,h), corners, ret)
#        cv2.imshow('findCorners',img)
#        cv2.waitKey(1)
#  cv2.destroyAllWindows()
#  
#  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
  img2 = img
  mtx = np.loadtxt('calib.txt')
  dist = np.loadtxt('dist.txt')
#  h,  w = img2.shape[:2]
  newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h)) # ���ɱ�������
  dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
  
  cv2.imwrite('calibresult.png',dst)
  
#  total_error = 0
#  for i in xrange(len(objpoints)):
#      imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#      error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#      total_error += error
#  print ("total error: ", total_error/len(objpoints))
  return dst