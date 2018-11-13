#! /usr/bin/python
# -*-coding:utf-8-*-
import numpy as np
import cv2
import os
from skimage import transform,img_as_ubyte
import pydicom
import shutil
import pydensecrf.densecrf as dcrf
import glob


def _dense_crf(img, out_probs):
    h = out_probs.shape[0]
    w = out_probs.shape[1]

    out_probs = np.expand_dims(out_probs, 0)
    out_probs = np.append(1 - out_probs, out_probs, axis=0)
    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(out_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U, np.float32)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=40, compat=8, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)
    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
    return Q

def _GetDataFromFile(dir1, dir2, newDir):
    for file in os.listdir(dir1):
        filename = os.path.splitext(file)[0]
        filename = filename + ".png"
        filedir = os.path.join(dir2, filename)
        newfileDir = os.path.join(newDir, filename)
        shutil.copy(filedir, newfileDir)

# 提取到文件夹中的文件名，将其保存到txt中
def _GetFileNameFromFolder(dir, MasktxtDir, fileName):
    filedir = os.path.join(MasktxtDir, fileName) + ".txt"
    if not os.path.isfile(filedir):
        # 创建文件
        os.mknod(filedir)
    with open(filedir, 'w') as f:
        for file in os.listdir(dir):
            f.write(file)
            f.write("\n")

def _MoveMasktoFolder(dir, dirTo):
    print (dir)
    for file in os.listdir(dir):
        fileName = os.path.splitext(file)[0]
        Folder = os.path.join(dirTo, fileName)
        filedir = os.path.join(dir, file)
        print(Folder)
        if not os.path.isdir(Folder):
            os.mkdir(Folder)
            shutil.copy(filedir, Folder)
        _GetFileNameFromFolder(Folder, dirTo, fileName)


# if __name__ == "__main__":
#     dir ="/home/andy/data/sg_mask_resize/"
#     dirTo   ="/home/andy/data/sg_mask/"
#     _MoveMasktoFolder(dir, dirTo)


# if __name__ == "__main__":
#     dir ="/home/andy/data/HasToDealPNGResize_1"
#     dirTo   ="/home/andy/data/ribMask/"
#     _MoveMasktoFolder(dir, dirTo)

def _GetFileFromFolder(dir1, dir2, dir3, dir4):
    for file in os.listdir(dir1):
        fileName = os.path.splitext(file)[0]
        if fileName in os.listdir(dir2):
            folderDir = os.path.join(os.path.join(dir2, fileName), "png")
            dstfolderDir = os.path.join(dir4, fileName)
            shutil.copytree(folderDir, dstfolderDir)
            _GetFileNameFromFolder(folderDir, dir4, fileName)
            fileDir = os.path.join(dir1, fileName) + ".dcm"
        elif fileName in os.listdir(dir3):
            folderDir = os.path.join(os.path.join(dir3, fileName), "png")
            dstfolderDir = os.path.join(dir4, fileName)
            shutil.copytree(folderDir, dstfolderDir)
            _GetFileNameFromFolder(folderDir, dir4, fileName)

# if __name__ == "__main__":
#     dir1= "/home/andy/data/do_again_result/do_more_sg"
#     dir2 = "/home/andy/test/rib_label_8_17/not_repeat_mask/sg_con/sg_dicom/"
#     dir3 = "/home/andy/test/rib_label_8_17/327_mask/sg_con/sg_dicom/"
#     dir4 ="/home/andy/data/do_again_result/do_more_sg_mask/"
#     _GetFileFromFolder(dir1, dir2, dir3, dir4)


# if __name__ == "__main__":
#     dir1= "/home/andy/data/do_again_result/do_more"
#     dir2 = "/home/andy/test/rib_label_8_17/not_repeat_mask/mask_con/dicom/"
#     dir3 = "/home/andy/test/rib_label_8_17/327_mask/mask_con/dicom/"
#     dir4 ="/home/andy/data/do_again_result/do_more_rib_mask/"
#     _GetFileFromFolder(dir1, dir2, dir3, dir4)


# if __name__ == "__main__":
#     dir1= "/home/andy/data/do_again_result/do_more"
#     dir2 = "/home/andy/test/rib_label_8_17/not_repeat_mask/sg_con/sg_dicom/"
#     dir3 = "/home/andy/test/rib_label_8_17/327_mask/sg_con/sg_dicom/"
#     dir4 ="/home/andy/data/do_again_result/do_more_mask/"
#     _GetFileFromFolder(dir1, dir2, dir3, dir4)


def _GetFileFromFolder(dir1, dir2, dir3, dir4, dir5):
    for file in os.listdir(dir1):
        fileName = os.path.splitext(file)[0]
        if fileName in os.listdir(dir2):
            folderDir = os.path.join(os.path.join(dir2, fileName), "png")
            dstfolderDir = os.path.join(dir4, fileName)
            shutil.copytree(folderDir, dstfolderDir)
            _GetFileNameFromFolder(folderDir, dir4, fileName)
            fileDir = os.path.join(dir1, fileName) + ".dcm"
            shutil.copy(fileDir, dir5)
        elif fileName in os.listdir(dir3):
            folderDir = os.path.join(os.path.join(dir3, fileName), "png")
            dstfolderDir = os.path.join(dir4, fileName)
            shutil.copytree(folderDir, dstfolderDir)
            _GetFileNameFromFolder(folderDir, dir4, fileName)
            fileDir = os.path.join(dir1, fileName) + ".dcm"
            shutil.copy(fileDir, dir5)



# if __name__ == "__main__":
#     dir1 = "/home/andy/data/HasToDealDcm_1"
#     dir2 = "/home/andy/test/rib_label_8_17/not_repeat_mask/sg_con/sg_dicom/"
#     dir3 = "/home/andy/test/rib_label_8_17/327_mask/sg_con/sg_dicom/"
#     dir4 = "/home/andy/data/Mask/"
#     dir5 = "/home/andy/data/sgDataDeal/"
#     _GetFileFromFolder(dir1, dir2, dir3, dir4, dir5)

# 将dir2中属于dir1的数据放到newDir中
def _GetDataFromFile(dir1, dir2, newDir):
    for file in os.listdir(dir1):
        filedir = os.path.join(dir2, file)
        newfileDir = os.path.join(newDir, file)
        shutil.copy(filedir, newfileDir)


# if __name__ == "__main__":
#     dir1 = "/home/andy/data/sgMarker"
#     dir2 = "/home/andy/data/dcm_src"
#     newDir= "/home/andy/data/sgMarker_"
#     _GetDataFromFile(dir1, dir2, newDir)

# 将dir2,dir3中属于dir1的放到dir4中
def _GetDataFromFolder(dir1, dir2, dir3, dir4):
    for file in os.listdir(dir1):
        filename = os.path.splitext(file)[0]
        file1 = filename + ".png"
        if file1 in os.listdir(dir2):
            filedir = os.path.join(dir2, file1)
            shutil.copy(filedir, dir4)
        elif file1 in os.listdir(dir3):
            filedir = os.path.join(dir3, file1)
            shutil.copy(filedir, dir4)

# 将dir2,dir3中属于dir1的放到dir4中
def _GetDataFromFolder_1(dir1, dir2, dir3, dir4):
    for file in os.listdir(dir1):
        file1 = os.path.splitext(file)[0] + ".dcm"
        if file1 in os.listdir(dir2):
            filedir = os.path.join(dir2, file1)
            shutil.copy(filedir, dir4)
        elif file1 in os.listdir(dir3):
            filedir = os.path.join(dir3, file1)
            shutil.copy(filedir, dir4)

# if __name__ == "__main__":
#     dir1 = "/home/andy/data/sg_mask/"
#     dir2 = "/home/andy/data/do_again_result/do_again_result"
#     dir3 = "/home/andy/data/do_again_result/GroundTruth_revise"
#     dir4 = "/home/andy/data/sgMarker"
#     _GetDataFromFolder_1(dir1, dir2, dir3, dir4)


# if __name__ =="__main__":
#     dir1="/home/andy/data/HasToDealDcm_1"
#     dir2 ="/home/andy/data/肋骨mask"
#     dir3 ="/home/andy/data/肋骨mask1"
#     dir4 = "/home/andy/data/HasToDealPNG_1/"
#     _GetDataFromFolder(dir1, dir2, dir3, dir4)


# if __name__ == "__main__":
#     dir1 = "/home/andy/data/sgDataDeal/"
#     dir2 = "/home/andy/test/TrainMaskResize/"
#     dir3 = "/home/andy/test/TestMaskResize/"
#     dir4 = "/home/andy/data/MaskAll"
#     _GetDataFromFolder(dir1, dir2, dir3, dir4)

#读取dcm数据转换成jepg的数据格式
def _TransformDCMToPNG(dir1, dir2):
    for file in os.listdir(dir1):
        fileName = os.path.splitext(file)[0]
        fileDir = os.path.join(dir1, file)
        file1 = fileName + ".jpeg"
        file1Dir = os.path.join(dir2, file1)
        ds = pydicom.read_file(fileDir)
        pixel_bytes = ds.PixelData
        array = ds.pixel_array
        array = array.astype(np.float64)
        array = (array - np.min(array))/(np.max(array)-np.min(array))*255.0
        array = array.astype(int)
        cv2.imwrite(file1Dir, array)


# if __name__ == "__main__":
#     dir1 = "/home/andy/data/HasToDealDcm_1/"
#     dir2 = "/home/andy/data/HasToDealPNG_1/"
#     _TransformDCMToPNG(dir1, dir2)



# if __name__ == "__main__":
#     dir1 = "/home/andy/data/sgMarker_/"
#     dir2 = "/home/andy/data/sgMarkerPNG/"
#     _TransformDCMToPNG(dir1, dir2)

# 从文件夹dir1中获取不属于dir2的数据保存到dir3中
def _GetDataFromFolder(dir1, dir2, dir3):
    for file in os.listdir(dir1):
        if file not in os.listdir(dir2):
            fileDir = os.path.join(dir1, file)
            shutil.copy(fileDir, dir3)

# if __name__ == "__main__":
#     dir1 ="/home/andy/data/do_again"
#     dir2="/home/andy/data/do_again_result/do_again_result"
#     dir3="/home/andy/data/do_again_result/do_more"
#     _GetDataFromFolder(dir1, dir2, dir3)

#从文件夹dir1中获取不属于dir2和dir3的数据保存到dir4中
def _GetDataFromFolder(dir1, dir2, dir3, dir4):
    for file in os.listdir(dir1):
        fileName = os.path.splitext(file)[0]
        print(fileName)
        if (fileName not in os.listdir(dir2)) and (fileName not in os.listdir(dir3)):
            fileDir = os.path.join(dir1, file)
            shutil.copy(fileDir, dir4)


# if __name__ =="__main__":
#     dir1 = "/home/andy/data/HasToDealDcm"
#     dir2 = "/home/andy/test/rib_label_8_17/not_repeat_mask/sg_con/sg_dicom/"
#     dir3 = "/home/andy/test/rib_label_8_17/327_mask/sg_con/sg_dicom/"
#     dir4 ="/home/andy/data/HasToDealDCM_1"
#     _GetDataFromFolder(dir1, dir2, dir3, dir4)

#从文件夹dir1中获取不属于dir2和dir3的数据保存到dir4中
def _GetDataFromFolder1(dir1, dir2, dir3, dir4):
    for file in os.listdir(dir1):
        if (file not in os.listdir(dir2)) and (file not in os.listdir(dir3)):
            fileDir = os.path.join(dir1, file)
            shutil.copy(fileDir, dir4)

# if __name__ == "__main__":
#     dir1 = "/home/andy/data/do_again/"
#     dir2 = "/home/andy/data/do_again_result/do_again_result_1"
#     dir3 = "/home/andy/data/do_again_result/do_again_result2"
#     dir4 = "/home/andy/data/HasToDealDcm/"
#     _GetDataFromFolder1(dir1, dir2, dir3, dir4)


# if __name__ == "__main__":
#     dir1 = "/home/andy/test/doctor_xie/"
#     dir2 = "/home/andy/test/rib_label_8_17/not_repeat_mask/sg_con/sg_dicom/"
#     dir3 = "/home/andy/test/rib_label_8_17/327_mask/sg_con/sg_dicom/"
#     dir4 = "/home/andy/data/sgMarker"
#     _GetDataFromFolder(dir1, dir2, dir3, dir4)

# if __name__ == "__main__":
#     dir1 = "/home/andy/data/sgData/"
#     dir2 = "/home/andy/test/rib_label_8_17/not_repeat_mask/sg_con/sg_dicom/"
#     dir3 = "/home/andy/test/rib_label_8_17/327_mask/sg_con/sg_dicom/"
#     dir4 = "/home/andy/data/Mask/"
#     dir5 = "/home/andy/data/sgDataDeal/"
#     _GetFileFromFolder(dir1, dir2, dir3, dir4, dir5)

# if __name__ == "__main__":
#     dir1 = "/home/andy/data/qian/"
#     dir2 = "/home/andy/test/do_again/"
#     dir3 = "/home/andy/data/sgData/"
#     _GetDataFromFolder(dir1, dir2, dir3)




# if __name__ == "__main__":
#     dir1 = "/home/andy/data/do_again/"
#     dir2 = "//home/andy/do_again_again/"
#     dir3 = "/home/andy/test/do_again_result/do_again_result_1/"
#     _GetDataFromFolder(dir1, dir2, dir3)

# if __name__ == "__main__":
#     dir1 = "/home/andy/ribRMData/data/do_again_again/"
#     dir2 = "/home/andy/test/do_again_result/do_again_result2/"
#     dir3 = "/home/andy/test/do_again_again_again"
#     _GetDataFromFolder(dir1, dir2, dir3)


# if __name__ == "__main__":
#     dir1 = "/home/andy/do_again_again"
#     dir2 = "/home/andy/test/rib_label_8_17/not_repeat_mask/mask_con/dicom"
#     dir3 = "/home/andy/test/rib_label_8_17/327_mask/mask_con/dicom"
#     dir4 = "/home/andy/test/Mask/"
#     dir5 = "/home/andy/test/do_again_again/"
#     _GetFileFromFolder(dir1, dir2, dir3, dir4, dir5)



# if __name__ == "__main__":
#     _GetFileNameFromFolder("/home/andy/ribRMData/ribrmwrapper/X15224379_rib/", "X15224379.txt")

# if __name__ =="__main__":
#     dir1 = "/home/andy/test/do_again"
#     dir2 = "/home/andy/test/doctor_xie_Lungmask"
#     newdir2 = "/home/andy/data/LungMask_do_again"
#     dir3 = "/home/andy/test/doctor_xie_ribmask"
#     newdir3 = "/home/andy/data/Ribmask_do_again"
#     _GetDataFromFile(dir1, dir2, newdir2)
#     _GetDataFromFile(dir1, dir3, newdir3)

def _maskResize(pathfile, pathmask, fileDir):

    img = cv2.imread(pathfile)
    origin = img.copy()
    img = img.astype(np.uint8)
    label = cv2.imread(pathmask, cv2.IMREAD_GRAYSCALE)
    if np.max(label) <= 1:
        label = label * 255
    # label = label[:, :, 0]
    # 实现将图像缩放和数值放缩到[0,1]之间
    label = transform.resize(label, img.shape[0:2])
    origin_label = label
    label = label.astype(np.float32)

    Q = _dense_crf(img.copy(), label)
    origin_label[origin_label > 0] = 255
    origin_label = origin_label.astype(np.uint8)
    Q[Q > 0] = 255
    # Q = Q.astype(np.float64)
    # Q = img_as_ubyte(Q)
    Q = Q.astype(np.uint8)
    cv2.imwrite(fileDir, Q)
    # ret_, binary_ = cv2.threshold(origin_label, 1, 255, cv2.THRESH_BINARY)
    # _, contours, hierarchy_ = cv2.findContours(binary_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # origin = cv2.drawContours(origin, contours, -1, (0, 0, 255), 1)
    # cv2.imwrite("B.jpg", origin)
    # ret_, binary_1 = cv2.threshold(Q, 1, 255, cv2.THRESH_BINARY)
    # _, contours_1, hierarchy_ = cv2.findContours(binary_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # origin = cv2.drawContours(origin, contours_1, -1, (0, 255, 0), 1)
    # cv2.imwrite("A.jpg", origin)

def _maskResizeFolder(pathfile, pathmask, saveDir):
    for file in os.listdir(pathfile):
        prefileDir = os.path.join(pathfile, file)
        file1 = os.path.splitext(file)[0] + ".jpeg"
        file2 = os.path.splitext(file)[0] + ".png"
        maskfileDir = os.path.join(pathmask, file1)
        savefileDir = os.path.join(saveDir, file2)
        print(prefileDir)
        print(maskfileDir)
        print(savefileDir)
        _maskResize(prefileDir, maskfileDir, savefileDir)

# if __name__ == "__main__":
#     pathfile = "/home/andy/data/sgMarkerPNG"
#     pathmask = "/home/andy/data/sg_mask"
#     fileDir = "/home/andy/data/sg_mask_resize"
#     _maskResizeFolder(pathfile, pathmask, fileDir)


# if __name__ == "__main__":
#     pathfile = "/home/andy/data/sgMarkerPNG/X15209314.jpeg"
#     pathmask = "/home/andy/data/sg_mask/X15209314.jpeg"
#     fileDir = "/home/andy/data/sg_mask_resize/X15209314.jpeg"
#     _maskResize(pathfile, pathmask, fileDir)

# if __name__ == "__main__":
#     dir1 = "/home/andy/ribRMData/Algo-server/src/ribRM/CPP/ribrmcore/Mask/X15228197/X15224379.png"
#     img = cv2.imread(dir1)
#     label = transform.resize(img, [1025, 1025])
#     cv2.imwrite("/home/andy/A.png", label)

# 测试将mask的精标的Ground Truth的左右两边的肋骨不一致
def _RetSetMask(filedir1, filedir2):
    imgLeft  = cv2.imread(filedir1, cv2.IMREAD_GRAYSCALE)
    imgRight = cv2.imread(filedir2, cv2.IMREAD_GRAYSCALE)
    imgLeft[imgLeft == 255] = 128
    img = imgLeft + imgRight
    return img

# if __name__ == "__main__":
#     filedir1 = "/home/shouliang/test/png/X15207198-1.png"
#     filedir2 = "/home/shouliang/test/png/X15207198-2.png"
#     img = _RetSetMask(filedir1, filedir2)

#获取图片，并将图片合成拷贝到某个文件夹中
def _SearchImg(file, dirRef, dirDst):
    fileName = os.path.splitext(file)[0]
    filedir = os.path.join(dirRef, fileName, "png")
    filedir1 = glob.glob(os.path.join(filedir, "*-1.png"))
    filedir2 = glob.glob(os.path.join(filedir, "*-2.png"))
    if len(filedir1) == 0 or len(filedir2) == 0:
        filedir1 = glob.glob(os.path.join(filedir, "*_1.png"))
        filedir2 = glob.glob(os.path.join(filedir, "*_2.png"))
        if len(filedir1) == 0 or len(filedir2) == 0:
            filedir1 = glob.glob(os.path.join(filedir, "*_1_*"))
            filedir2 = glob.glob(os.path.join(filedir, "*_2_*"))
            if len(filedir1) == 0 or len(filedir2) == 0:
                print(filedir)
                return
    filedir1 = filedir1[0]
    filedir2 = filedir2[0]
    img = _RetSetMask(filedir1, filedir2)
    saveFileDir = os.path.join(dirDst, file)
    cv2.imwrite(saveFileDir, img)

def _GetSeq(bFlag,filedir, pos):
    fileseq = []
    for dir in filedir:
        fileName = os.path.split(dir)[1]
        if bFlag == 0:
            if pos == 0:
                indstart = fileName.index('-1-')
            else:
                indstart = fileName.index('-2-')
        if bFlag == 1:
            if pos == 0:
                indstart = fileName.index('_1_')
            else:
                indstart = fileName.index('_2_')
        idxend = fileName.index('.')
        sub_s = fileName[(indstart + 3):idxend]
        fileseq.append(sub_s)
    return fileseq

# 测试将mask的精标的Ground Truth的左右两边的肋骨不一致
def _RetSetRibMask(filedir1=None, filedir2=None):
    imgLeft = np.empty(0)
    imgRight = np.empty(0)
    if not filedir1 == None:
        imgLeft  = cv2.imread(filedir1, cv2.IMREAD_GRAYSCALE)
        imgLeft[imgLeft > 0] = 255
    if not filedir2 == None:
        imgRight = cv2.imread(filedir2, cv2.IMREAD_GRAYSCALE)
        imgRight[imgRight > 0] = 255
    if imgLeft.shape[0] > 0 and imgRight.shape[0] > 0:
        img = imgLeft + imgRight
    elif imgLeft.shape[0] > 0:
        img = imgLeft
    else:
        img = imgRight
    return img


#获取制定的图片，将对称的肋骨合并到制定的文件夹中
def _SearchRibImg(file, dirRef, dirDst):
    fileName = os.path.splitext(file)[0]
    filedir = os.path.join(dirRef, fileName, "png")
    # 先分类所有-1,-2的文件，然后匹配
    filedir1 = glob.glob(os.path.join(filedir, "*-1-*.png"))
    filedir2 = glob.glob(os.path.join(filedir, "*-2-*.png"))
    bFlag = 0
    if len(filedir1) == 0 or len(filedir2) == 0:
        filedir1 = glob.glob(os.path.join(filedir, "*_1_*.png"))
        filedir2 = glob.glob(os.path.join(filedir, "*_2_*.png"))
        bFlag = 1
        if len(filedir1) == 0 or len(filedir2) == 0:
            bFlag = 2
            print(filedir)

    filedir1Seq = _GetSeq(bFlag, filedir1, 0)
    filedir2Seq = _GetSeq(bFlag, filedir2, 1)
    filedirSeq = filedir1Seq + filedir2Seq
    filedirSeq = list(set(filedirSeq))
    filedirSeq.sort()

    for i in filedirSeq:
        idx1 = filedir1Seq.index(i) if i in filedir1Seq else -1
        idx2 = filedir2Seq.index(i) if i in filedir2Seq else -1
        dir1 = filedir1[idx1] if idx1 > -1 else None
        dir2 = filedir2[idx2] if idx2 > -1 else None
        img = _RetSetRibMask(dir1, dir2)
        folderIdx = 'rib' + i
        dirDst_ = os.path.join(dirDst, folderIdx)
        if not os.path.isdir(dirDst_):
            os.mkdir(dirDst_)
        saveFileDir = os.path.join(dirDst_, file)
        cv2.imwrite(saveFileDir, img)

# 将训练的文件中的mask的数据的单支合成拷贝到指定的文件夹中
def _getSyn(dirsrc, dirRef1, dirRef2, dirDst):
    for file in os.listdir(dirsrc):
        fileName = os.path.splitext(file)[0]
        if fileName in os.listdir(dirRef1):
            _SearchImg(file, dirRef1, dirDst)
        elif fileName in os.listdir(dirRef2):
            _SearchImg(file, dirRef2, dirDst)


def _getSyn(dirsrc, dirRef1, dirRef2, dirDst):
    for file in os.listdir(dirsrc):
        fileName = os.path.splitext(file)[0]
        if fileName in os.listdir(dirRef1):
            _SearchRibImg(file, dirRef1, dirDst)
        elif fileName in os.listdir(dirRef2):
            _SearchRibImg(file, dirRef2, dirDst)


# 将2,3肋骨合成输出
# 将8，9,10,11,12肋骨合成输出
# 4,5,6,7肋骨分开输出
def _SetRib1Syn(ribSynList, ribSynDir):
    for file in os.listdir(ribSynList[0]):
        ribpath = os.path.join(ribSynList[0], file)
        rib = cv2.imread(ribpath, -1)
        for i in range(1, len(ribSynList)):
            ribpath = os.path.join(ribSynList[i], file)
            rib1 = cv2.imread(ribpath, -1)
            rib += rib1

        rib[rib > 255] = 255
        savefileDir = os.path.join(ribSynDir, file)
        cv2.imwrite(savefileDir, rib)

if __name__ == "__main__":
    dir = "/home/shouliang/test/ribMaskVal"
    ribSynList1 = []
    ribSynList2 = []
    ribSynList3 = []
    for i in range(2, 4):
        ribDir = os.path.join(dir, "rib")+str(i)
        ribSynList1.append(ribDir)

    for i in range(4, 8):
        ribDir = os.path.join(dir, "rib") + str(i)
        ribSynList2.append(ribDir)

    for i in range(8, 12):
        ribDir = os.path.join(dir, "rib") + str(i)
        ribSynList3.append(ribDir)

    ribSynDir1 = os.path.join(dir, "ribSyn1")
    ribSynDir2 = os.path.join(dir, "ribSyn2")
    ribSynDir3 = os.path.join(dir, "ribSyn3")
    if not os.path.isdir(ribSynDir1):
        os.mkdir(ribSynDir1)

    if not os.path.isdir(ribSynDir2):
        os.mkdir(ribSynDir2)

    if not os.path.isdir(ribSynDir3):
        os.mkdir(ribSynDir3)



    _SetRib1Syn(ribSynList1, ribSynDir1)
    _SetRib1Syn(ribSynList2, ribSynDir2)
    _SetRib1Syn(ribSynList3, ribSynDir3)

def _SetRib2Syn(rib2Dir,rib3Dir, rib1Syn):
    for file in os.listdir(rib2Dir):
        rib2path = os.path.join(rib2Dir, file)
        rib3path = os.path.join(rib3Dir,file)
        rib2 = cv2.imread(rib2path)
        rib3 = cv2.imread(rib3path)
        rib = rib2 + rib3
        rib[rib > 255] = 255
        savefileDir = os.path.join(rib1Syn,file)
        cv2.imwrite(savefileDir, rib)

# if __name__ == "__main__":
#     dirsrc  = "/home/shouliang/test/sg_label_val/"
#     dirRef1 = "/home/shouliang/test/rib_label_8_17/327_mask/mask_con/dicom/"
#     dirRef2 = "/home/shouliang/test/rib_label_8_17/not_repeat_mask/mask_con/dicom/"
#     dirDst = "/home/shouliang/test/ribMaskVal/"
#     _getSyn(dirsrc, dirRef1, dirRef2, dirDst)


#将图像中大于0的像素的灰度值变到255
def _ReSetImg(srcDir,dstDir):
    for file in os.listdir(srcDir):
        filedir = os.path.join(srcDir, file)
        img = cv2.imread(filedir, cv2.IMREAD_GRAYSCALE)
        img[img == 128] = 255
        dstfiledir = os.path.join(dstDir, file)
        cv2.imwrite(dstfiledir, img)

# if __name__ == "__main__":
#     srcDir = "/home/shouliang/test/sg_label_val/"
#     dstDir = "/home/shouliang/test/sg_label_reset/"
#     _ReSetImg(srcDir, dstDir)