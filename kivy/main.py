#from re import M
import kivy
#kivy.require('1.0.6') # replace with your current kivy version !

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
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import os
from scipy.spatial import distance
import time
from testmedia import Eye_check

from kivy.garden.notification import Notification


fontName = 'NanumGothic.ttf'
class Screen2(GridLayout):
    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)
        self.rows =5

        
        self.check_blink_num= BoxLayout(orientation='horizontal')
        self.add_widget(self.check_blink_num)
        self.check_blink_num.add_widget(Label(text='눈 깜박임수', padding=(30,20), size_hint=(0.3, 1), font_name=fontName ))
        self.check_blink_num.add_widget(Label(text='20'))
        
        ## slider
        self.slider= Slider(orientation= 'horizontal', value_track=True, value_track_color=(1,0,0,1), size_hint=(0.3, 0.8))
        self.slider.bind(value= self.on_slider_changed)
        self.add_widget(self.slider)


        ## 분체크 레이아웃
        self.min_checkbox= BoxLayout(orientation='horizontal', padding=(70, 40))
        self.add_widget(self.min_checkbox)

     
        self.min_checkbox.add_widget(Label(text='1분', font_name=fontName ))
        self.min_checkbox.one_min = CheckBox(size_hint=(0.3, 1))
        self.min_checkbox.one_min.bind(active= self.on_checkbox)
        self.min_checkbox.add_widget(self.min_checkbox.one_min)


        self.min_checkbox.add_widget(Label(text='3분', font_name=fontName ))
        self.min_checkbox.three_min = CheckBox()
        self.min_checkbox.three_min.bind(active= self.on_checkbox)
        self.min_checkbox.add_widget(self.min_checkbox.three_min)


        self.min_checkbox.add_widget(Label(text='5분', font_name=fontName ))
        self.min_checkbox.five_min = CheckBox()
        self.min_checkbox.five_min.bind(active= self.on_checkbox)
        self.min_checkbox.add_widget(self.min_checkbox.five_min)

        self.min_checkbox.add_widget(Label(text='10분', font_name=fontName ))
        self.min_checkbox.ten_min = CheckBox()
        self.min_checkbox.ten_min.bind(active= self.on_checkbox)
        self.min_checkbox.add_widget(self.min_checkbox.ten_min)


        ## 알림 설정 레이아웃

        self.notice_layout= BoxLayout(orientation='vertical')
        self.add_widget(self.notice_layout)

        
        self.notice_layout.add_widget(Label(text='1분', font_name=fontName ))
        self.notice_layout.alarm_check = CheckBox(size_hint=(0.3, 1))
        self.notice_layout.alarm_check.bind(active= self.on_checkbox)
        self.notice_layout.add_widget(self.notice_layout.alarm_check)

        self.notice_layout.add_widget(Label(text='1분', font_name=fontName ))
        self.notice_layout.add_widget(Label(text='1분', font_name=fontName ))
        self.notice_layout.add_widget(Label(text='1분', font_name=fontName ))
        

        self.button= Button(text="Start",font_size=40 , padding=[90,40] )
        self.button.bind(on_press= self.on_pressed)
        self.add_widget(self.button)


    def on_pressed(self, instance):
        print("pressed button")

    def on_checkbox(self, instance, value):
        if value:
            print("checked")
        else:
            print("unchecke")

    def on_slider_changed(self,instance,value):
        print(value)
        #self.ids.checknum_testinput.text= str(int(value))


class Setting_window(Screen):
    pass

class Running_window(Screen):
    pass

class WindowManager(ScreenManager):
    pass

class Upper_bar(BoxLayout):
    def on_change_checknum(self):
        text = self.ids.checknum_testinput.text
        self.ids.check_num_slider.value= int(text)
        print(text)
    def on_slider_change(self,*args):
        self.ids.checknum_testinput.text= str(int(args[1]))
    pass

class slider_bar(BoxLayout):  
    pass

class Check_minute(BoxLayout):
    def on_checkbox(self, instance, value):
        if value:
            print("checked")
        else:
            print("unchecke")
    pass

class Notification_bar(BoxLayout):
    def on_notfication_check(self, instance, value, notification):
        print("check")
    pass
       
class Button_bar(BoxLayout):
    def notification(self):
        Notification().open(
            title=" test notification",
            message= " test notification",
            timeout = 2
        )
    pass


#class Running_blinkeye_check_bar(BoxLayout):
#    pass

#class Running_blinkeye_button_bar(BoxLayout):
#    pass

class MainApp(App):
    pass


class MyApp(App):

    def build(self):
        #Eye_check.checkblink()
        return  Screen2()


if __name__ == '__main__':
    #MyApp().run()
    MainApp().run()