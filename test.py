from keras.preprocessing import image
import numpy as np
import train.py as model



def launch():
    while True:
        try:
            img_path = str(raw_input("jpg file path: "))
            img = image.load_img(img_path, target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            classes = model.predict(x, 16, 0)
            ts = classes.tostring()
            tx = np.fromstring(ts, dtype=int)
            # print (tx)
            # print(str(classes))

            if str(tx) == '[0]':
                print("Chekaem " + str(img_path))
                print('')
                print("Koteika")
                print('')

            elif str(tx) == '[1065353216]':
                print("Chekaem " + str(img_path))
                print('')
                print("Sobaka spidozniy")
                print('')
            else:
                print("X3 4e eto takoe")
        except:
            print("4eto keknulos'")


