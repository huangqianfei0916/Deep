import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
import pylab

import matplotlib
from itertools import cycle
from sklearn import svm
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(12,8.5), dpi=300,facecolor=(1, 1, 1))#平铺画布

y1 = [0.876,0.9,0.852,0.753]#设置 Acc,Sn,Sp,MCC
y2 = [0.894,0.92,0.868,0.789]
y3 = [0.904,0.96,0.848,0.813]

ind = np.arange(4)+2                # the x locations for the groups
width = 0.25
plt.bar(ind,y1,width,color = '#EEC900',label = 'NB',edgecolor = 'black')  #绘制直方图，ind表示哪些位置，y1表示高度，width表示宽度
plt.bar(ind+width,y2,width,color = '#EE8262',label = 'RF',edgecolor = 'black') # ind+width adjusts the left start location of the bar.
plt.bar(ind+2*width,y3,width,color = '#36648B',label = "SVM",edgecolor = 'black')
plt.xticks(np.arange(4) + 9*width, ('ACC','SP','SE','MCC'))#x轴下表，第一个参数为位置

# plt.legend(loc="upper right",bbox_to_anchor=(1, 0.5))
# plt.legend(loc="upper right",ncol=2)
plt.legend(loc="upper right",fontsize=12)
# plt.legend(loc="upper right",ncol=2,handletextpad=0.2,fontsize=9.05,columnspacing=-4)
plt.title('Ten-fold cross validation',fontsize=12)#写剃头了
plt.text(1.15,0.99,'A',fontsize='xx-large')#图片中写字
# Independent Test
plt.ylim(0.45, 1)#设置y轴长度
# plt.savefig('F:/draw_protein/hist_ROC_one.tif',bbox_inches='tight')
plt.show()