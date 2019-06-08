import cv2 as cv
import warnings
warnings.filterwarnings("ignore")
import p_part
import f_part
from caption_tune import modcap, face_found_cap, face_not_found_cap
from gensound import generate_sound
import tkinter as tk
from faceadd import addn,speech


def saveface():
 #   generate_sound("Tell me the name")
    x1 = speech("What is this human called?")
    print(x1 + ' face saved')
    cv.imwrite(r"images//" +
               str(x1) + ".jpg", save)
    data = {x1: f_part.img_to_encoding(
        "images//" + str(x1) + ".jpg").tolist()}
    f_part.digi_db.insert_one(data)


def ignoreface():
    generate_sound("Not saved")


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    facedetect = cv.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    if ret:
        # font = cv.FONT_HERSHEY_SIMPLEX
        cv.imshow("Video", frame)

        if cv.waitKey(1) == ord('p'):

            cv.imwrite('./test.jpg', frame)
            final_caption = p_part.generate_caption(
                './test.jpg')  # create caption
            final_caption = modcap(final_caption)  # remove tags
            print(final_caption)
            generate_sound(final_caption)  # convert to audio

        if cv.waitKey(1) == ord('f'):
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            cv.imwrite('./test.jpg', frame)
            known_detected = 0
            unknown_detected = 0
            known_face_list = []
            known_face_dist = []
            try:
                for x, y, w, h in faces:
                    #cv2.imwrite("dset//User."+str(user)+"."+str(sample)+".jpg",gray[y:y+h,x:x+w])
                    save = frame[y:y+h, x:x+w]
                    cv.imwrite('./test.jpg', save)
                    dis, name = f_part.who_is_it('./test.jpg')
                    print(str(dis)+","+name)
                    if name != 'unknown':
                        known_face_list.append(name)
                        known_face_dist.append(dis)
                        known_detected += 1

                    else:
                        unknown_detected += 1

                if known_detected > 0:
                    print("known: " + str(known_detected))
                    for i in range(known_detected):
                        print('i=' + str(i))
                        print(
                            known_face_list[i] + " at dist of: " + str(known_face_dist[i]))
                        temp = face_found_cap(str(known_face_list[i]))
                        generate_sound(temp)
                elif unknown_detected == 1:
                    temp = face_not_found_cap()
                    generate_sound(temp)
                    generate_sound("Do you want to add this face in your database")
                    addn(save)

                elif known_detected == 0 and unknown_detected == 0:
                    print("No person found")
                    generate_sound("No person found!")

                else:
                    print("Too many people")
                    generate_sound("Too many people.")
            except Exception as e:
                generate_sound("No recognisable face found!")
                print(e)

        if cv.waitKey(1) & 0xFF == 27:  # ASCII for Esc Key
            break
    else:
        break
cap.release()
cv.destroyAllWindows()