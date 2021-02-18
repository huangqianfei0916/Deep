import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

font={'family':'serif',
    'weight':'normal',
      'size':10
}
plt.figure(figsize=(16,10),dpi=300)
ax1 = plt.subplot(211)
plt.subplots_adjust(wspace =0.3, hspace =0.35)
label_list = ['Sn', 'Sp', 'ACC', 'MCC']     # 横坐标刻度显示值
num_list1 = [84.89, 89.66, 87.27, 75.0]      # 纵坐标值1
num_list2 = [86.48, 89.77, 88.1, 76.0]      # 纵坐标值2
num_list3 = [85.8, 88.4, 87.1, 74.0]      # 纵坐标值1
num_list4 = [85.8, 90.0, 87.9, 76.0]      # 纵坐标值2

import numpy as np
x = np.arange(len(label_list))
width = 0.20
rects1 = plt.bar(x=x - width*2, height=num_list1, width=width, edgecolor = 'black', label="SVM")
rects2 = plt.bar(x=x - width, height=num_list2, width=width,edgecolor = 'black', label="XGBoost")
rects3 = plt.bar(x=x , height=num_list3, width=width, edgecolor = 'black', label="GBDT")
rects4 = plt.bar(x=x + width, height=num_list4, width=width,edgecolor = 'black', label="vote")

plt.ylim(50, 120)     # y轴取值范围
plt.ylabel("")

plt.xticks(x, label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注
plt.text(-0.75,125,"A",fontdict=font)
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
for rect in rects3:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")
for rect in rects4:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
#####################################
ax2 = plt.subplot(212)
plt.subplots_adjust(wspace =0.3, hspace =0.35)
label_list = ['Sn', 'Sp', 'ACC', 'MCC']    # 横坐标刻度显示值
num_list1 = [95.97, 75.33, 85.65, 73.0]      # 纵坐标值1
num_list2 = [92.86, 71.13, 82.0, 66.0]      # 纵坐标值2
num_list3 = [92.18, 71.43, 81.8, 65.0]      # 纵坐标值1
num_list4 = [93.99, 73.1, 83.55, 69.0]      # 纵坐标值2

import numpy as np
x = np.arange(len(label_list))
width = 0.20
rects1 = plt.bar(x=x - width*2, height=num_list1, width=width, edgecolor = 'black', label="SVM")
rects2 = plt.bar(x=x - width, height=num_list2, width=width,edgecolor = 'black', label="XGBoost")
rects3 = plt.bar(x=x , height=num_list3, width=width, edgecolor = 'black', label="GBDT")
rects4 = plt.bar(x=x + width, height=num_list4, width=width,edgecolor = 'black', label="vote")

plt.ylim(50, 120)     # y轴取值范围
plt.ylabel("")

plt.xticks(x, label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注
plt.text(-0.75,125,"B",fontdict=font)
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
for rect in rects3:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")
for rect in rects4:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
plt.savefig("C://Users//hqf//Desktop//paper/Fig6.png", bbox_inches='tight')
plt.show()