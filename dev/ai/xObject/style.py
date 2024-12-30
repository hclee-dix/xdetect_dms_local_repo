from dev.ai.xBase.style import XStyle
##
from dev.schema.style import IStyle

class XObjectStyle(XStyle):
    def __init__(self, meta):
        super().__init__(meta)
        self.bboxContainer = IStyle(
            background_color=self.rgba2hex((200,40,255,0.7)),
            border_color=self.rgba2hex((200,40,255,0.7)),
            border_width=self.w(10),
            border_radius=self.w(10)
        )
        self.bboxLabel = IStyle(
            width=self.w(150),
            height=self.h(50),
            padding_left=self.w(10),
            padding_top=self.h(10),
            padding_right=self.w(10),
            padding_bottom=self.h(0),
        )
    