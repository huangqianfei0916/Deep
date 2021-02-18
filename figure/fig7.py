import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


plt.figure(figsize=(9,3),dpi=320)
ax1 = plt.subplot(131)
plt.subplots_adjust(wspace =0.3, hspace =0.35)
label_list = ['1', '2', '3', '4']     # 横坐标刻度显示值
num_list1 = [85.9, 90.0, 93.11, 85.96]      # 纵坐标值1

import numpy as np
x = np.arange(len(label_list))
width = 0.45
rects1 = plt.bar(x=x, height=num_list1, width=width, edgecolor = 'black', label="6mA-Pred")

plt.ylim(50, 120)     # y轴取值范围
plt.ylabel("ACC",fontsize=8)

plt.xticks(x, label_list)
plt.xlabel("KMER的K值",fontsize=8)
plt.title("老鼠",fontsize=8)
plt.legend()     # 设置题注
plt.text(-1,125,"A")
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")

#####################################
ax2 = plt.subplot(132)
plt.subplots_adjust(wspace =0.3, hspace =0.35)
label_list = ['1', '2', '3', '4']    # 横坐标刻度显示值
num_list1 = [92.1, 87.83, 93.56, 92.23]      # 纵坐标值1

import numpy as np
x = np.arange(len(label_list))
width = 0.45
rects1 = plt.bar(x=x, height=num_list1, width=width, edgecolor = 'black', label="6mA-Pred")


plt.ylim(50, 120)     # y轴取值范围
plt.ylabel("ACC",fontsize=8)

plt.xticks(x, label_list)
plt.xlabel("KMER的k值",fontsize=8)
plt.title("水稻",fontsize=8)
plt.legend()     # 设置题注
plt.text(-1,125,"B")

# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")
#############################
ax3 = plt.subplot(133)
plt.subplots_adjust(wspace =0.3, hspace =0.35)
label_list = ['1', '2', '3', '4']    # 横坐标刻度显示值
num_list1 = [93.3, 93.2, 93.3, 92.9]      # 纵坐标值1

import numpy as np
x = np.arange(len(label_list))
width = 0.45
rects1 = plt.bar(x=x, height=num_list1, width=width, edgecolor = 'black', label="6mA-Pred")

plt.ylim(50, 120)     # y轴取值范围
plt.ylabel("ACC",fontsize=8)

plt.xticks(x, label_list)
plt.xlabel("KMER的K值",fontsize=8)
plt.title("人类",fontsize=8)
plt.legend()     # 设置题注
plt.text(-1,125,"C")

# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")
plt.savefig("C://Users//hqf//Desktop//paper/Fig7.png", bbox_inches='tight')
plt.show()