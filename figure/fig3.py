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
plt.figure(figsize=(16,12),dpi=350)
ax1 = plt.subplot(221)
plt.subplots_adjust(wspace =0.3, hspace =0.35)
label_list = ['Sn', 'Sp', 'Acc', 'AUC']    # 横坐标刻度显示值
num_list1 = [91.4, 70.9, 81.1, 90.4]      # 纵坐标值1
num_list2 = [94.6, 91.7, 93.1, 97.4]      # 纵坐标值2

x = [0,1.5,3,4.5]
rects1 = plt.bar(x=x, height=num_list1, width=0.5,  color='#aecdc2',edgecolor = 'black', label="iDNA6mA-rice")
rects2 = plt.bar(x=[i + 0.5 for i in x], height=num_list2, width=0.5,edgecolor = 'black', color='#f0b8b8', label="6mA-Pred")

plt.ylim(0, 120)     # y轴取值范围
plt.ylabel("")

plt.xticks([index + 0.3 for index in x], label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注
plt.text(-1,125,"A:(5:5)",fontdict=font)
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

#################################################################################
ax2 = plt.subplot(222)

label_list = ['Sn', 'Sp', 'Acc', 'AUC']    # 横坐标刻度显示值
num_list1 = [92.2, 90.6, 91.4, 96.3]  # 纵坐标值1
num_list2 = [95.1, 91.6, 93.4, 97.7]  # 纵坐标值2

x = [0,1.5,3,4.5]


rects1 = plt.bar(x=x, height=num_list1, width=0.5,  edgecolor = 'black',color='#aecdc2', label="iDNA6mA-rice")
rects2 = plt.bar(x=[i + 0.5 for i in x], height=num_list2, width=0.5,edgecolor = 'black', color='#f0b8b8', label="6mA-Pred")

plt.ylim(0, 120)     # y轴取值范围
plt.ylabel("")

"""
设置x轴刻度显示值
参数一：中点坐标
参数二：显示值
"""
plt.xticks([index + 0.3 for index in x], label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注

plt.text(-1,125,"B:(7:3)",fontdict=font)
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

#######################
ax3= plt.subplot(223)

label_list = ['Sn', 'Sp', 'Acc', 'AUC']    # 横坐标刻度显示值
num_list1 = [92.4, 91.7, 92.1, 96.7]  # 纵坐标值1
num_list2 = [94.2, 91.3, 92.7, 97.3]  # 纵坐标值2

x = [0,1.5,3,4.5]

rects1 = plt.bar(x=x, height=num_list1, width=0.5,  edgecolor = 'black',color='#aecdc2', label="iDNA6mA-rice")
rects2 = plt.bar(x=[i + 0.5 for i in x], height=num_list2, width=0.5,edgecolor = 'black', color='#f0b8b8', label="6mA-Pred")

plt.ylim(0, 120)     # y轴取值范围
plt.ylabel("")

"""
设置x轴刻度显示值
参数一：中点坐标
参数二：显示值
"""
plt.xticks([index + 0.3 for index in x], label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注

plt.text(-1,125,"C:(8:2)",fontdict=font)
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

##################
ax4= plt.subplot(224)

label_list = ['Sn', 'Sp', 'Acc', 'AUC']    # 横坐标刻度显示值
num_list1 = [92.7, 92.1, 92.2, 96.9]  # 纵坐标值1
num_list2 = [94.7, 92.6, 93.6, 97.7]  # 纵坐标值2

x = [0,1.5,3,4.5]


rects1 = plt.bar(x=x, height=num_list1, width=0.5,  edgecolor = 'black',color='#aecdc2', label="iDNA6mA-rice")
rects2 = plt.bar(x=[i + 0.5 for i in x], height=num_list2, width=0.5, edgecolor = 'black',color='#f0b8b8', label="6mA-Pred")

plt.ylim(0, 120)     # y轴取值范围
plt.ylabel("")

"""
设置x轴刻度显示值
参数一：中点坐标
参数二：显示值
"""
plt.xticks([index + 0.3 for index in x], label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注

plt.text(-1,125,"D:(9:1)",fontdict=font)
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")


plt.savefig("C://Users//hqf//Desktop//paper/Fig35.svg", bbox_inches='tight')
plt.show()