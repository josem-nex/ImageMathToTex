import os
from texteller.inference import execute

if "__main__" == __name__:
    diract = os.path.dirname(__file__)
    # print(diract)
    path = os.path.join(diract, "test/test.png")
    
    res = execute(path)
    
    print(res)