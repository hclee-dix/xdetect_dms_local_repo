from dev.ai.xBase.style import XStyle
##
from dev.schema.style import IStyle,IFontStyle

class XFireStyle(XStyle):
    def __init__(self, meta):
        super().__init__(meta)
        self.alert_container= IStyle(
            background_color=self.rgba2hex((0,0,0,0.1)),
            padding_left=self.w(200),
            padding_top=self.h(20),
            padding_right = self.w(200),
            padding_bottom=self.h(200)
        )
        self.alert_box = IStyle(
            width=self.w(600),
            height=self.h(200),
            padding_left=self.w(20),
            padding_top=self.h(20),
            padding_right=self.w(20),
            padding_bottom=self.h(20),
        )
        self.warn_text = IFontStyle(
            font_color=self.rgba2hex((0,200,255,1.0))
        )
        self.alert_text = IFontStyle(
            font_color=self.rgba2hex((0,0,255,1.0))
        )