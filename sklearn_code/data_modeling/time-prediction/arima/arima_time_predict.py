#use arima model to predict stock data

#############################################################################################################################################################
#利用arima模型进行时序分析预测的过程：

#----->>>>>>1.拿到时序数据，画出序列图，观察序列是否有周期性与趋势性，判断是否平稳，也可通过画出自相关图与偏自相关图

#----->>>>>>2.进行季节性分解，提取分解出来的SAF，SAS，观察序列是否有周期性与趋势性，判断是否平稳，也可通过画出自相关图与偏自相关图

#------>>>>>3.对提取出来的子序列进行平稳性分析，将计算得到的p-value与显著性水平数值比较

#------>>>>>4.若不是平稳性序列，进行差分处理，直到变为平稳性序列，记录差分次数d

#------>>>>>5.拿到平稳序列后，进行白噪声检验，将计算得到的p-value与显著水平数值比较

#------>>>>>6.若平稳序列不是白噪声，则进行arima参数p，q确定分析，也可通过画出自相关图与偏自相关图，人为判断p，q参数

#------>>>>>7.获得模型后，计算获得模型的残值，对这个残值进行白噪声分析，若是白噪声序列，则说明该模型可以用来预测分析，否则说明模型还需要改进
############################################################################################################################################################

#注意：有时平稳性检查会出现偏差，需要进行观察序列图进行辅助判断！！！
#注意：有时差分后平稳的数据会被判断为白噪声，原则上即没有分析的价值了，但是还可以继续用信息准则求出p，q用以建立arima模型，只要模型通过残值检验，就可以用来预测，起到参考作用
#注意：对模型的检验，即获取模型的残值序列，然后进行该序列的白噪声检验，利用QQ图进行正态分布检验，以及Ljung-BOX检验，即Q检验
######################################################################################################################################


#################################################################################################################################
#画出原始数据的序列图
#时序图function
def plot_data(data):
	import matplotlib.pyplot as plt
	plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
	plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
	print('通过数据的序列图来观测数据，观察数据序列是否有趋势性以及周期性，也就是序列的平均值以及方差的数值是否保持稳定！！，通过对序列图的观察大致判断数据是否为平稳序列')
	data.plot()
	plt.show()



def check_steady(data):
	#平稳性检测function
	from statsmodels.tsa.stattools import adfuller as ADF
	print(u'原始序列的ADF检验结果为：')
	print('返回所有的信息：',ADF(data))
	#返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore
	print('返回P_VALUE：',ADF(data)[1])
	print('---->>>>将计算获得的p-value与显著性水平数值0.05比较，大于该数值说明该序列不是平稳序列，反之是平稳序列！')
	if ADF(data)[1]<0.05:
		print('----->>>>>STEADY!')
	else:
		print('----->>>>>NOT STEADY!')

def check_self(data):
	from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
	plot_acf(data).show()#自相关图
	plot_pacf(data).show() #偏自相关图



def diff(data):
	#差分后的结果
	D_data = data.diff().dropna()
	return D_data


def check_whitenoise(data):
	#白噪声检验
	from statsmodels.stats.diagnostic import acorr_ljungbox
	print(u'序列的白噪声检验结果为：', acorr_ljungbox(data, lags=12)) #返回统计量和p值
	print('检查该序列是否为随机白噪声序列，观察输出的P值与假设显著值的比较，若大于显著值则说明该序列是白噪声序列！！')
	if acorr_ljungbox(data, lags=12)[1][0]>0.05:
		print('------>>>>>>it is white noise !! ')
	else:
		print('------>>>>>>not white noise !!')


def find_p_q(data):
	from statsmodels.tsa.arima_model import ARIMA
	#定阶
	pmax = int(len(data)/10)#一般阶数不超过length/10
	qmax = int(len(data)/10)  #一般阶数不超过length/10
	bic_matrix = [] #bic矩阵
	data=np.array(data,dtype=np.float)
	for p in range(pmax+1):
  		tmp = []
  		for q in range(qmax+1):
  			try:
  				tmp.append(ARIMA(data, (p,1,q)).fit().bic)
  			except:
  				tmp.append(None)
  				print('fault')
  		bic_matrix.append(tmp)
	bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值
	print(bic_matrix)
	p,q = bic_matrix.stack().idxmin() 
	#先用stack展平，然后用idxmin找出最小值位置。
	print(u'BIC最小的p值和q值为：%s、%s' %(p,q)) 


def bulid_arima_predict(data,p,d,q,predict_days):
	#根据给定的数据与参数来构建arima模型
	from statsmodels.tsa.arima_model import ARIMA
	model = ARIMA(data, order=(p,d,q)).fit()
	S=model.summary2() #给出一份模型报告
	if predict_days>0:
		P=model.forecast(predict_days) #作为期5天的预测，返回预测结果、标准误差、置信区间。
		print('================================================report========================================================')
		print(S)
		print('=================================================predict of the 5 days in future=============================================================')
		print(pd.Series(list(P)).iloc[0])

		predict_data=[]#构造一个预测序列列表，其中包括历史数据（经过d次查分）以及最终预测出来的数据
		for i in range(len(data)):
			predict_data.append(data[i])
		for i in range(predict_days):
			predict_data.append(pd.Series(list(P)).iloc[0][i])

	return predict_data


def bulid_arima_give_model(data,p,d,q):
	#根据给定的数据与参数来构建arima模型,give the model out to check its error 
	from statsmodels.tsa.arima_model import ARIMA
	model = ARIMA(data, (p,d,q)).fit() 
	return model


def get_error_arima_model(arima_model): 
	#计算得到arima模型的残值
	error=arima_model.resid
	return error


########################################################操作区域#########################################################################################
import pandas as pd
import numpy as np
data=pd.read_excel('E:\Master\PPDAMcode\system_WARN_INFO\system_info.xls',head=None,index_col = u'日期')
#注意：输入的时序数据需要日期数据列，并作为导入数据的index_col，否则arima建模会出现问题
data=data.iloc[200:,:]
data=data[u'日志类告警']
data=np.array(data,dtype=np.float)
print(data)

#data1=diff(data)
#plot_data(data)
check_steady(data)

#check_self(data)
#check_whitenoise(data)
#find_p_q(data)
bulid_arima_predict(data,0,0,1,10)
#check_whitenoise(get_error_arima_model(bulid_arima_give_model(data,0,0,1)))
#29.4 32.35 32 32.75 34.1

