import numpy as np

coef = [-0.00222292, 0.02062339, 0.02574563]
intercept = [5.90809568]

student=[601,25,42]
print(np.dot(coef,student)+intercept[0])