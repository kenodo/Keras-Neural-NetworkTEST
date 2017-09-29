from keras.preprocessing import image
import numpy as np
import os
import train.py as model
import test.py





def cls():
    os.system('cls' if os.name=='nt' else 'clear')




model.load_weights('weights.h5')

cls()

print("Vvedite, chto sdelat' dalshe: ")
print("1 Natrenirovat' set'")
print("2 Proverit' set' na gotovix vesah")

cheDelat=str(raw_input("vvod: "))


if (str(cheDelat=="1")):
    model.launch()
elif (str(cheDelat=="2")):
    test.launch()