import numpy as np
import pickle
model = pickle.load(open('profitpred.pkl', 'rb'))

rd = float(input())
adm = float(input())
mrk = float(input())
d1 = float(input())
d2 = float(input())

int_features = [rd,adm,mrk,d1,d2]
final_features = [np.array(int_features)]
prediction = model.predict(final_features)
output = round(prediction[0], 2)
print(output)

