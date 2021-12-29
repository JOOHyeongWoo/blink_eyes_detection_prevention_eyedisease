# -*- coding: utf-8 -*-

import kivy
import sys

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.clock import Clock
from kivy.uix.camera import Camera
from kivy.lang import Builder
from kivy.core.text import LabelBase
import cv2


import numpy as np
from keras.models import load_model
import os
import time
import threading


import mediapipe as mp
from testmedia import Eye_check      
from kivy.garden.notification import Notification



def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


fontName = resource_path('NanumGothic.ttf')


class Setting_window(Screen):

    pass

class Running_window(Screen):
    def changeLabel2(self,closedEyenum):
        self.get_screen('running').ids.check_eye_num.text= str(closedEyenum)
        return

    
    pass

class WindowManager(ScreenManager):
    
    def on_startpage(self,  instance):
        self.get_screen('running').ids.check_eye_num.text= '개인 눈 세팅중'

    def on_blinkEyeprogram(self,  instance):
        
        Eye_check.programstate =True
        eyecheck_thread = threading.Thread(target=Eye_check.checkblink, args=(self,))
        eyecheck_thread.setDaemon(True)
        eyecheck_thread.start()

        

    def close_blinkEyeprogram(self, instace):
        Eye_check.programstate =False
        
    def notification(self):
        Notification().open(
            title=" test notification",
            message= " test notification",
            timeout = 2
        )
    pass



sm =WindowManager()
sm.add_widget(Setting_window(name='setting'))
sm.add_widget(Running_window(name='running'))
rw=Running_window()

class Upper_bar(BoxLayout):
    def on_change_checknum(self):
        text = self.ids.checknum_testinput.text
        self.ids.check_num_slider.value= int(text)
        Eye_check.max_blinknum_permin= int(text)
        
    def on_slider_change(self,*args):
        self.ids.checknum_testinput.text= str(int(args[1]))
        Eye_check.max_blinknum_permin= int(args[1])
        
        
    pass

class slider_bar(BoxLayout):  
    pass

class Check_minute(BoxLayout):
    def on_checkbox(self, instance, value, permin):
        Eye_check.check_permin= int(permin)
        
    pass

class Notification_bar(BoxLayout):
    def on_notfication_check(self, instance, value, notification):
        print()
    pass
       
class Button_bar(BoxLayout):

    pass


class Running_blinkeye_check_bar(BoxLayout):
    
    pass

class Running_blinkeye_button_bar(BoxLayout):

    pass

class MainApp(App):
    LabelBase.register(name='NanumGothic.ttf', fn_regular=resource_path('NanumGothic.ttf'))
    
    def build(self):
        with open(resource_path('main3.kv'), encoding='utf-8') as f:
            return Builder.load_string(f.read())
            


if __name__ == '__main__':
    
    MainApp().run()

