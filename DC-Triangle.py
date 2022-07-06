import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import pdb
import glob
from scipy.stats import gaussian_kde
from scipy import linalg

##原理简介：输入棋盘图原图和变形棋盘图，得到畸变模式。用畸变模式，矫正变形模式为变形棋盘图的目标图片，至棋盘图原图的模式。
##使用说明：输入任意格式棋盘图原图、变形棋盘图，手动输入棋盘图横纵格数w和h，输入目标图片，等待输出即可。

##root = tk.Tk()
##root.withdraw()
##file = tk.filedialog.askopenfilename(parent=root)
file = 'simple512.jpg'
file_Trans = 'simple512-n.jpg'
file_target = 'lena.jpg'
file_background = 'bg.jpg'

img = cv2.imread(file)
imgTrans = cv2.imread(file_Trans)
img_target = cv2.imread(file_target)
img_bg = cv2.imread(file_background)

img_t_arr = img_target.flatten()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

w = int(input('棋盘格横向点数 w: '))
h = int(input('棋盘格纵向点数 h: '))

########### find corners in origin & trans  #############
corners_origin = []     #角点数据（原）
corners_trans = []      #角点数据（变形）

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(imgTrans,cv2.COLOR_BGR2GRAY)
##gray_target = cv2.cvtColor(img_target,cv2.COLOR_BGR2GRAY)
##print(img_target[0][0])
##
##print('gray',gray,len(gray))
##pdb.set_trace()

ret,corners = cv2.findChessboardCorners(gray,(w,h),None)
if ret == True:
    corners = cv2.cornerSubPix(gray,corners,(1,1),(-1,-1),criteria)
    cv2.drawChessboardCorners(img,(w,h),corners,ret)
    print('corners',type(corners),len(corners),corners)
    corners_origin = corners
##    plt.imshow(img,cmap='gray',interpolation='bicubic')
##    cv2.imwrite('img.jpg',img)
##    plt.show()
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
else:
    print('ret:',ret)
    print( 'Can not find corners in origin file')


ret,corners = cv2.findChessboardCorners(gray1,(w,h),None)
if ret == True:
    corners = cv2.cornerSubPix(gray,corners,(1,1),(-1,-1),criteria)
    cv2.drawChessboardCorners(imgTrans,(w,h),corners,ret)
    print('corners',type(corners),len(corners),corners)
    corners_trans = corners
##    plt.imshow(img,cmap='gray',interpolation='bicubic')
##    cv2.imwrite('img.jpg',img)
##    plt.show()
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
else:
    print('ret:',ret)
    print( 'Can not find corners in trans file')


print('corners_orgin',len(corners_origin),corners_origin)
print('corners_trans',len(corners_trans),corners_trans)

print('corners_trans: ',corners_trans[0],corners_trans[195])

cv2.imshow('imgTrans',imgTrans)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
###############  divide into groups  ##########################
# x' = f1 + f2*x + f3*y + f4*xy
# y' = f5 + f6*x + f7*y + f8*xy
f_list = []
points_list = []
point2search = (w*(h-1))
for i in range(point2search):
    if((i%w == (w-1)) or (i == (w-1))): #每到横向边缘最后一个点 跳过
        continue
    else:
        x1,y1 = corners_trans[i].ravel()
        x_1,y_1 = corners_origin[i].ravel()
        x2,y2 = corners_trans[i+1].ravel()
        x_2,y_2 = corners_origin[i+1].ravel()
        x3,y3 = corners_trans[i+14].ravel()
        x_3,y_3 = corners_origin[i+14].ravel()
        x4,y4 = corners_trans[i+15].ravel()
        x_4,y_4 = corners_origin[i+15].ravel()
    
        A2_x = np.array([[1,x1,y1],[1,x2,y2],[1,x4,y4]])
        b2_x = np.array([x_1,x_2,x_4])
        A2_y = np.array([[1,x1,y1],[1,x2,y2],[1,x4,y4]])
        b2_y = np.array([y_1,y_2,y_4])

        A1_x = np.array([[1,x1,y1],[1,x3,y3],[1,x4,y4]])
        b1_x = np.array([x_1,x_3,x_4])
        A1_y = np.array([[1,x1,y1],[1,x3,y3],[1,x4,y4]])
        b1_y = np.array([y_1,y_3,y_4])
    
        f1_x = linalg.solve(A1_x,b1_x)
        f1_y = linalg.solve(A1_y,b1_y)
        f2_x = linalg.solve(A2_x,b2_x)
        f2_y = linalg.solve(A2_y,b2_y)

        points_list.append((x_1,y_1,x_3,y_3,x_4,y_4)) #三角形134
        points_list.append((x_1,y_1,x_2,y_2,x_4,y_4)) #三角形124
        
        f_list.append((f1_x,f1_y))
        f_list.append((f2_x,f2_y))
        
f_array = np.array(f_list)   #预畸变参数组结果
points_array = np.array(points_list)
print('参数组总数：',len(f_array))
##print(f_array)
##print(points_array)


#############################################主要操作gray_target
x0,y0 = corners_trans[0].ravel() #起始点坐标
h_target,w_target,channel = img_target.shape
x_start,y_start = corners_trans[0].ravel()
x_end,y_end = corners_trans[-1].ravel()
x0grid,y0grid = corners_origin[0].ravel()
x1grid,y1grid = corners_origin[1].ravel()
xwgrid,ywgrid = corners_origin[w].ravel()
w_search = int(x1grid-x0grid)
h_search = int(ywgrid-y0grid)
h_grid = int(y_end - y_start)
w_grid = int(x_end - x_start)
grid = []
pointer_list = []
scan_area_xmax = int(w_search*1.8)  #执行每一个分组模糊检测的范围（判断合适与否影响分组的分离度）
scan_area_xmin = int(w_search*0.3)
scan_area_ymax = int(h_search*1.8)
scan_area_ymin = int(h_search*0.3)

p_h_g2t = (h_grid/h_target)
p_w_g2t = (w_grid/w_target)


for i in range(h_target):
    y_grid = i * p_h_g2t + y0
    for j in range(w_target):
        x_grid = j * p_w_g2t + x0
        grid.append((x_grid,y_grid))
print(len(grid))

for i in range(len(points_array)):
    xa,ya,xb,yb,xc,yc = points_array[i]
    
    for j in range(len(grid)):
        x,y = grid[j] # 访问标准网格第xy个点 查看分组信息 并链接target

        if( ( x>=(xa-scan_area_xmin) and x<(xa + scan_area_xmax) ) and ( (y>=(ya-scan_area_ymin)) and (y<(ya + scan_area_ymax) ) )):#检索的时候先设置一个模糊范围减少计算量
            s1 =0.5*abs(xb*y + xa*yb + x*ya - x*yb - xb*ya - xa*y)
            s2 =0.5*abs(x*yc + xa*y + xc*ya - xc*y - x*ya - xa*yc)
            s3 =0.5*abs(xb*yc + x*yb + xc*y - xc*yb - xb*y - x*yc)
            s_0 = 0.5*abs(xb*yc + xa*yb + xc*ya - xc*yb - xb*ya - xa*yc)
            
            if  ( s_0-(s1+s2+s3) >=-107  ):
                f1,f2,f3,f4,f5,f6 = f_array[i].ravel() #访问分组参数 准备预畸变

                x_target = (f1 + f2*x + f3*y - x0)/p_w_g2t + 200
                y_target = (f4 + f5*x + f6*y - y0)/p_h_g2t + 200

                x_max = math.ceil(x_target) #向下、上取整 由一个像素对应4个像素
                y_max = math.ceil(y_target)
                x_min = int(x_target)
                y_min = int(y_target)
                
                color = (img_t_arr[j*3],img_t_arr[1+j*3],img_t_arr[2+j*3])

                img_bg[y_min][x_min] = color
                img_bg[y_min][x_max] = color
                img_bg[y_max][x_min] = color
                img_bg[y_max][x_max] = color

            else: continue
        else: continue
    print(i+1,'/',len(points_array))

cv2.imshow('imgTrans',img_bg)
cv2.imwrite('demo25-f.jpg',img_bg)
cv2.waitKey(-1)
cv2.destroyAllWindows()
