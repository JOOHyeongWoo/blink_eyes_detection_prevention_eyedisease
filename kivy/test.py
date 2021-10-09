import kivy

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.garden.notification import Notification

fontname= 'NanumGothic.ttf'

class testscreen(GridLayout):
    def __init__(self, **kwargs):
        super(testscreen, self).__init__(**kwargs)
        self.button= Button(text="Start",font_size=40 , padding=[90,40], font_name=fontname )
        self.button.bind(on_release= self.testnotification)
        self.add_widget(self.button)

    def testnotification(self, instance):
        Notification().open(
            title=" test notification",
            message= " test notification",
            timeout = 2
        )


class TestApp(App):
    
    def build(self):
        return testscreen()


if __name__ == '__main__':
    TestApp().run()
