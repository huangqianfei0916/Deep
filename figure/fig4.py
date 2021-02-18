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
plt.figure(figsize=(16,10),dpi=350)
ax1 = plt.subplot(231)
plt.subplots_adjust(wspace =0.3, hspace =0.35)
label_list = ['SVM', 'XGBoost', 'RF', '6mA_Pred']    # 横坐标刻度显示值
num_list1 = [74.88, 77.0, 75.11, 93.8]      # 纵坐标值1

x = [1,2,3,4]

rects1 = plt.bar(x=x, height=num_list1, width=0.5,  color='#ffa600')

plt.ylim(0, 120)     # y轴取值范围
plt.ylabel("Acc")

plt.xticks([index  for index in x], label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注
plt.text(0,130,"A",fontdict=font)
plt.text(2,125,"Mouse-KMER",fontdict=font)
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3.5, height+1, str(height), ha="center", va="bottom")


#################################################################################
ax2 = plt.subplot(232)

label_list = ['SVM', 'XGBoost', 'RF', '6mA_Pred']    # 横坐标刻度显示值
num_list2 = [69.1, 68.8, 70.4, 93.6]  # 纵坐标值2
x = [1,2,3,4]

rects2 = plt.bar(x=[i for i in x], height=num_list2, width=0.5, color='#ff6e54')

plt.ylim(0, 120)     # y轴取值范围
plt.ylabel("Acc")

plt.xticks([index  for index in x], label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注
plt.text(0,130,"B",fontdict=font)
plt.text(2,125,"Rice-KMER",fontdict=font)

for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

##########################################################
ax3= plt.subplot(233)

label_list = ['SVM', 'XGBoost', 'RF', '6mA_Pred']   # 横坐标刻度显示值
num_list1 = [75.3, 76.7, 76.3, 93.34]  # 纵坐标值1
x = [1,2,3,4]
rects1 = plt.bar(x=x, height=num_list1, width=0.5,  color='#dd5182')

plt.ylim(0, 120)     # y轴取值范围
plt.ylabel("Acc")

plt.xticks([index for index in x], label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注
plt.text(0,130,"C",fontdict=font)
plt.text(2,125,"Human-KMER",fontdict=font)
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")

ax4 = plt.subplot(234)
plt.subplots_adjust(wspace =0.3, hspace =0.35)
label_list = ['SVM', 'XGBoost', 'RF', '6mA_Pred']    # 横坐标刻度显示值
num_list1 = [96.1, 96.1, 96.1, 93.8]      # 纵坐标值1
x = [1,2,3,4]
rects1 = plt.bar(x=x, height=num_list1, width=0.5,  color='#955196')
plt.ylim(0, 120)     # y轴取值范围
plt.ylabel("Acc")

plt.xticks([index  for index in x], label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注
plt.text(0,130,"D",fontdict=font)
plt.text(2,125,"Mouse-NCP",fontdict=font)
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3.5, height+1, str(height), ha="center", va="bottom")


#################################################################################
ax5 = plt.subplot(235)

label_list = ['SVM', 'XGBoost', 'RF', '6mA_Pred']    # 横坐标刻度显示值
num_list2 = [92.1, 91.2, 92.0, 93.6]  # 纵坐标值2

x = [1,2,3,4]
rects2 = plt.bar(x=[i for i in x], height=num_list2, width=0.5, color='#444e86')

plt.ylim(0, 120)     # y轴取值范围
plt.ylabel("Acc")

plt.xticks([index  for index in x], label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注
plt.text(0,130,"E",fontdict=font)
plt.text(2,125,"Rice-NCP",fontdict=font)
# 编辑文本

for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

####################################################
ax6= plt.subplot(236)

label_list = ['SVM', 'XGBoost', 'RF', '6mA_Pred']    # 横坐标刻度显示值
num_list1 = [92.2, 92.8, 91.2, 93.34]  # 纵坐标值1
x = [1,2,3,4]

rects1 = plt.bar(x=x, height=num_list1, width=0.5,  color='#003f5c')

plt.ylim(0, 120)     # y轴取值范围
plt.ylabel("Acc")

plt.xticks([index for index in x], label_list)
plt.xlabel("")
plt.title("")
plt.legend()     # 设置题注
plt.text(0,130,"F",fontdict=font)
plt.text(2,125,"Human-NCP",fontdict=font)
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 3, height+1, str(height), ha="center", va="bottom")

plt.savefig("C://Users//hqf//Desktop//paper/Fig4.png", bbox_inches='tight')
plt.show()