import numpy as np
import codecs, json

f1 = 'variableDic.json'
f2 = 'variableDic46250.json'

file1 = codecs.open(f1, 'r', encoding = 'utf-8').read()
file2 = codecs.open(f2, 'r',encoding = 'utf-8').read()
data1 = json.loads(file1)
data2 = json.loads(file2)

#para1 = 'lstm/BasicLSTMCell/w2:0'
#para2 = 'seq_embedding/map:0'
para2 = 'lstm/BasicLSTMCell/Linear/Matrix:0'

diff = 0
num = 0
for i in range(len(data1[para2])):
  for j in range(len(data1[para2][0])):
    change = abs(data1[para2][i][j]-data2[para2][i][j])
    if change != 0:
      diff += change
      num +=1
print(para2)
print('diff', diff, num)  


