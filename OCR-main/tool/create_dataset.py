import os
import lmdb 
import cv2 
import numpy as np
import argparse

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
    except:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if isinstance(v, bytes):
                txn.put(k.encode('utf-8'), v)
            else:
                txn.put(k.encode('utf-8'), v.encode('utf-8'))

def createDataset(outputPath, imagePathList, labelList, checkValid=True):
    """
    Tạo LMDB dataset cho CRNN
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    
    # Tạo thư mục nếu chưa có
    os.makedirs(outputPath, exist_ok=True)
    
    env = lmdb.open(outputPath, map_size=10737418240)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print(f'Không tìm thấy ảnh: {imagePath}')
            continue
            
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            
        if checkValid:
            if not checkImageIsValid(imageBin):
                print(f'Ảnh bị lỗi, bỏ qua: {imagePath}')
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print(f'Đã đóng gói {cnt} / {nSamples} ảnh...')
        cnt += 1
        
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print(f'Hoàn tất! Đã tạo bộ dữ liệu LMDB với {nSamples} ảnh.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', required=True, help='Đường dẫn xuất file LMDB')
    parser.add_argument('--labelpath', required=True, help='Đường dẫn tới file train.txt')
    args = parser.parse_args()

    with open(args.labelpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    imagePathList = []
    labelList = []
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            imagePathList.append(parts[0])
            labelList.append(parts[1])
            
    createDataset(args.outpath, imagePathList, labelList)