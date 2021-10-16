from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np



class MainnApp(App):

    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.image = Image()
        layout.add_widget(self.image)
        layout.add_widget ( Button(text="CLICK HERE",pos_hint={'center_x': .5, 'center_y': .5}, size_hint=(None, None)))

        
        self.capture = cv2.VideoCapture(0)
        self.load_video(self)
        #Clock.schedule_interval(self.load_video, 1.0/3.0)
        return layout

    def load_video(self, *args):
        
        while(True):
            
            ret, frame = self.capture.read()
            # Frame initialize
            self.image_frame = frame
            buffer = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture= texture

        


if __name__ == '__main__':
    MainnApp().run()