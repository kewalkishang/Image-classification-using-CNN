import numpy as np
import os

breed=['cat','dog']

newpath = r'F:\kaggle\dogcat\sorted' 
for br in breed:
        dirn = os.path.join(newpath,br)
        os.makedirs(dirn)
        
oriPath= r"F:\kaggle\dogcat\train"
cat='cat'
dog='dog'
for img in glob.glob(os.path.join(oriPath,'*.jpg')): 
    filen=os.path.splitext(os.path.basename(img))[0]
    filj=os.path.splitext(os.path.basename(img))[0]+".jpg"
    if "cat" in filen:
        os.rename(os.path.join(oriPath,filj),os.path.join(newpath,cat,filj))
    else:
         os.rename(os.path.join(oriPath,filj),os.path.join(newpath,dog,filj))