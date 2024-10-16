import os
from texteller.inference import execute

if "__main__" == __name__:
    diract = os.path.dirname(__file__)
    # print(diract)
    # path = os.path.join(diract, "test\\test.png")
    res = ''
    for i in range(0,6):
        path = os.path.join(diract, f"test\\data\\line_{i}.jpg")
        res += execute(path)+'\n\n'
    
    # path = os.path.join(diract, "test\\image5.jpg")
    # res = execute(path)
    print(res)