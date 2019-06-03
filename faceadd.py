import speech_recognition as sr
from gensound import generate_sound
import f_part
import cv2 as cv
def speech(abc):

# obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Pls say something....")
        generate_sound(abc)  
        audio = r.listen(source)

    # recognize speech 
    try:
        print("Google Audio:" + r.recognize_google(audio))
        return(r.recognize_google(audio))
      #  print("Sphinx:" + r.recognize_sphinx(audio))
    except sr.UnknownValueError:
        generate_sound("Could not understand your response Speak again")
        speech(abc)
    except sr.RequestError as e:
        print("error: {0}".format(e))
        generate_sound("Connection Error")


def addn(save):
    ans = str(speech("Yes or No"))
    if(ans == "yes" or ans == "Yes"):
        saveface(save)
    elif(ans == "no" or ans == "No"):
        ignoreface()
    else:
        generate_sound("Could not understand your response Answer again")
        addn(save)

def saveface(save):
 #   generate_sound("Tell me the name")
    x1 = speech("Tell me the name")
    print(x1 + ' face saved')
    cv.imwrite(r"images//" +
               str(x1) + ".jpg", save)
    data = {x1: f_part.img_to_encoding(
        "images//" + str(x1) + ".jpg").tolist()}
    f_part.digi_db.insert_one(data)


def ignoreface():
    generate_sound("Not saved")