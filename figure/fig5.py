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
plt.figure(figsize=(12,8),dpi=300)
ax1 = plt.subplot(111)
plt.subplots_adjust(wspace =0.3, hspace =0.35)
label_list = ['chi2', 'ANOVA', 'Lasso', 'XGBoost']    # 横坐标刻度显示值
num_list1 = [85.36, 85.47, 83.14, 85.65]      # 纵坐标值1
num_list2 = [72.18, 72.36, 66.95, 72.87]      # 纵坐标值2

import numpy as np
x = np.arange(len(label_list))
width = 0.35
rects1 = plt.bar(x=x - width/2, height=num_list1, width=width, edgecolor = 'black', label="ACC")
rects2 = plt.bar(x=x + width/2, height=num_list2, width=width,edgecolor = 'black', label="MCC")

plt.ylim(0, 120)     # y轴取值范围
plt.ylabel("")

plt.xticks(x, label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注

# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")



plt.savefig("C://Users//hqf//Desktop//paper/Fig5.png", bbox_inches='tight')
plt.show()