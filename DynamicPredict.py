'''
@Brief predict future 5 hours pm2.5 value
@Auther: Shawn Yang, Matthew Liu
@Date: 2018/06/26
'''
import urllib.request
import json 
import ssl
import numpy as np
import pandas as pd
###################################
import config
#PastHours = config.PastHours #Set the number of past hours data for Yang predict model
PastHours = 6
#PastHoursJson = config.PastHoursJson  # Path to hourly average data 
PastHoursJson = '/home/ubuntu/www/yang_past_6hours.json'
#predict_hour = config.PredictHours 
predict_hour = 5
## Read past n hours data
with open(PastHoursJson , 'r') as reader:
    data = json.loads(reader.read())

def computeM_hour(Ref_time, result, lamda):
    '''
    Predict one hour pm2.5 by linear regression
    '''
    leftpart = np.zeros(shape=(len(result),len(result)))
    rightpart = np.zeros(shape=(len(result),len(result)))    
    for i in range(0,Ref_time):
        leftpart += pd.DataFrame(pd.DataFrame(result[i+1]).values.dot(pd.DataFrame(result[i]).T.values))
    for i in range(0,Ref_time):
        rightpart += pd.DataFrame(pd.DataFrame(result[i]).dot(pd.DataFrame(result[i]).T.values))
    array = lamda * np.identity(len(result)) + rightpart
    array_inv = np.linalg.pinv(array)
    new_M = leftpart.dot(array_inv)   
    answer = new_M.dot(result[0].values)
    return pd.DataFrame(answer)

def lamda_loop(count=1, head=1503, tail=30005):
    lamda = 0
    for i in range(120):
        initial_A = computeM_hour(3, pd.DataFrame(CityDeviec.iloc[:,1:].values, 
                                      columns=range(0,PastHours-1), 
                                      index=CityDeviec.index),head)
        initial_B = computeM_hour(3, pd.DataFrame(CityDeviec.iloc[:,1:].values, 
                                      columns=range(0,PastHours-1), 
                                      index=CityDeviec.index),tail)
        A_10 = computeM_hour(3, pd.DataFrame(CityDeviec.iloc[:,1:].values, 
                                      columns=range(0,PastHours-1), 
                                      index=CityDeviec.index),head-10)
        B_10 = computeM_hour(3, pd.DataFrame(CityDeviec.iloc[:,1:].values, 
                                      columns=range(0,PastHours-1), 
                                      index=CityDeviec.index),tail+10)
        err_A = pd.DataFrame(abs(CityDeviec[0].values - initial_A[0].values)/(CityDeviec[0].values+1)).mean(axis=0)[0]
        err_B = pd.DataFrame(abs(CityDeviec[0].values - initial_B[0].values)/(CityDeviec[0].values+1)).mean(axis=0)[0]
        err_A_10 = pd.DataFrame(abs(CityDeviec[0].values - A_10[0].values)/(CityDeviec[0].values+1)).mean(axis=0)[0]
        err_B_10 = pd.DataFrame(abs(CityDeviec[0].values - B_10[0].values)/(CityDeviec[0].values+1)).mean(axis=0)[0]
        A = head
        B = tail
        if err_A >= err_B:
            if err_B <= err_B_10:
                if err_A <= err_B_10:
                    tail = (head + tail)/2
                    lamda = tail
                else:
                    head = (head + tail)/2
                    lamda = head
            else:
                tail -= 10
                lamda = tail
        else:
            if err_A <= err_A_10:
                if err_A_10 <= err_B:
                    tail = (head + tail)/2
                    lamda = tail
                else:
                    head = (head + tail)/2
                    lamda = head
            else:
                head -= 10
                lamda = head
    return lamda

 
CityDeviec = pd.DataFrame.from_dict(data, orient='columns').T
CityDeviec.columns =[n for n in range(PastHours)]
PredictMatrix = pd.DataFrame(np.zeros(shape=(len(CityDeviec),PastHours-1)), columns=range(0,PastHours-1), index=CityDeviec.index)
lamda = lamda_loop()
err_r = []

'''
Find optimized time parameter by PastHoursJson
'''
for i in range(1, predict_hour):
    one = computeM_hour(i,pd.DataFrame(CityDeviec.iloc[:,1:].values, 
                                           columns=range(0,PastHours-1), 
                                           index=CityDeviec.index),
                       lamda)
    one = pd.DataFrame(one.values, columns=[i-1])
    err_r.append(pd.DataFrame(abs(CityDeviec[0].values - one[i-1].values)/(CityDeviec[0].values+1)).mean(axis=0)[0])
time = err_r.index(min(err_r)) + 1


'''
Predict n hours pm2.5 by real data and predicted data.
New predict value is equal to predict value * 0.3 + last predict value * 0.7 (data assimilation)
'''
for i in range(1,predict_hour+1):
    if i == 1:
        PredictMatrix[i-1] = computeM_hour(time, CityDeviec, lamda).values*0.3 + CityDeviec[0].values*0.7
    else:
        PredictMatrix[i-1] = computeM_hour(time, CityDeviec, lamda).values*0.3 + PredictMatrix[i-2].values*0.7
    del CityDeviec[PastHours-1]
    CityDeviec = pd.DataFrame(CityDeviec.values, columns=range(1,PastHours), index=CityDeviec.index)
    CityDeviec[0] = PredictMatrix[i-1].values
'''
Output to Json format
'''
final_predict = PredictMatrix.to_json(orient='index')
f = open('yang_Taiwan.json','w') ## Write predict result to yang_Taiwan.json
f.write(final_predict)
