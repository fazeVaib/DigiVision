import cv2 as cv
import warnings
warnings.filterwarnings("ignore")
import p_part
import f_part
from caption_tune import modcap, face_found_cap, face_not_found_cap
from gensoundgtts import generate_sound
import tkinter as tk


def saveface():
    x1 = entry1.get()
    print(x1 + ' face saved')
    root.destroy()
    cv.imwrite(r"images//" +
        str(x1) + ".jpg", frame)
    data = {x1: f_part.img_to_encoding(
        "images//" + str(x1) + ".jpg").tolist()}
    f_part.digi_db.insert_one(data)


def ignoreface():
    print("Not saved")
    root.destroy()


cap = cv.VideoCapture('Sample Videos/test.mp4')

while True:
    ret, frame = cap.read()

    if ret:
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.imshow("Video", frame)

        if cv.waitKey(5) == ord('p'):
            #             print(K.image_data_format())
            cv.imwrite('./test.jpg', frame)
            final_caption = p_part.generate_caption('./test.jpg')  # create caption
            final_caption = modcap(final_caption)  # remove tags
            print(final_caption)
            generate_sound(final_caption)  # convert to audio

        if cv.waitKey(5) == ord('f'):
            cv.imwrite('./test.jpg', frame)
            try:
                dis, name = f_part.who_is_it('./test.jpg')
                print(str(dis)+","+name)
                temp = face_found_cap(name)
                generate_sound(temp)
                if(name == 'unknown'):
                    temp = face_not_found_cap()
                    generate_sound(temp)

                    root = tk.Tk()

                    large_font = ('Times New Roman', 14)

                    canvas1 = tk.Canvas(root, width=300, height=200)
                    canvas1.pack()
                    label = tk.Label(root, text='Enter the Name')
                    canvas1.create_window(140, 50, window=label)
                    entry1Var = tk.StringVar(value='')
                    entry1 = tk.Entry(root, textvariable=entry1Var, font=large_font)
                    canvas1.create_window(150, 90, window=entry1)
                    button1 = tk.Button(text='SAVE', command=saveface)
                    button2 = tk.Button(text='IGNORE', command=ignoreface)
                    canvas1.create_window(100, 150, window=button1)
                    canvas1.create_window(180, 150, window=button2)

                    root.mainloop()
            except:
                print("No recognizable face detected")
                generate_sound("No recognizable face detected")

        if cv.waitKey(1) & 0xFF == 27:  # ASCII for Esc Key
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
