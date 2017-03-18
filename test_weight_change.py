import numpy as np
import codecs, json

#f1 = 'variableDic.json'
f2 = 'variableDic46250.json'

#file1 = codecs.open(f1, 'r', encoding = 'utf-8').read()
file2 = codecs.open(f2, 'r',encoding = 'utf-8').read()
#data1 = json.loads(file1)
data2 = json.loads(file2)

para1 = 'lstm/BasicLSTMCell/w2:0'
#para2 = 'seq_embedding/map:0'
para2 = 'lstm/BasicLSTMCell/Linear/Matrix:0'
para3 = 'lstm/BasicLSTMCell/w1:0'

diff = 0
num = 0

for i in range(4):
  print('matrix',i)
  we = data2[para2][0:512][i*512:(i+1)*512]
  print(np.linalg.norm(we,2)) 
  h = data2[para2][512:1024][i*512:(i+1)*512]
  print(np.linalg.norm(h,2))
  zi = data2[para2][1024:][i*512:(i+1)*512]
  print(np.linalg.norm(zi,2))

print('lstm_cell_w2 l2 norm', np.linalg.norm(data2[para1]))
print('lstm_cell_w1 l2 norm', np.linalg.norm(data2[para3]))

#for i in range(len(data1[para2])):
# for j in range(len(data1[para2][0])):
#    change = abs(data1[para2][i][j]-data2[para2][i][j])
#    if change != 0:
#      diff += change
#      num +=1
#print(para2)
#print('diff', diff, num)  


