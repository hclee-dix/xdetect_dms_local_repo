import math
from typing import Union,Tuple
from types import SimpleNamespace
from dev.util.util import *
from PIL import Image,ImageFont
import numpy as np
from operator import add as _add,sub as _sub,mul as _mul

class XStyle:
    def __init__(self,meta):
        self.ref_width = 1920 # ref_width
        self.ref_height = 1080 # ref_height
        self.im_width = meta.width # image_width
        self.im_height = meta.height # image_height
        self.width_ratio = self.im_width/self.ref_width
        self.height_ratio = self.im_height/self.ref_height
        self.ratio = math.sqrt((self.im_width/self.ref_width)**2+(self.im_height/self.ref_height)**2)
        
    def w(self,v):
        assert 0<=v<=self.ref_width
        return int(v*self.width_ratio)
    
    def h(self,v):
        assert 0<=v<=self.ref_height
        return int(v*self.height_ratio)
    
    def fs(self,v, min_size = 12):
        return v*self.ratio or max(round(sum((self.ref_width,self.ref_height)) /2 * 0.035),min_size)*self.ratio
    
    def getOriginCoord(self,v):
        ratioMap = [1/self.width_ratio,1/self.height_ratio]*2
        return mul(v,ratioMap)
    
    def getBoxCoord(self,v):
        ratioMap = [1/self.width_ratio,1/self.height_ratio]*3
        w = v[2]-v[0]
        h = v[3]-v[1]
        v = v+[w,h]
        return mul(v,ratioMap)

    
class InfoStyle(XStyle):
    def __init__(self,meta):
        super().__init__(meta)
        self.container = SimpleNamespace(**{
            'width': meta.width,
            'height': meta.height,
            'background_color':rgba2hex((0,0,0,0)) 
        })
        self.background= SimpleNamespace(**{
            'width': self.w(118), # px
            'height': self.h(60), # px
            'left': 0, # px
            'top': 0, # px
            'bottom': 0, # px
            'right': 0, # px
            'margin_left': self.w(10), # px
            'margin_top': self.h(10), # px
            'padding_left': self.w(10), # px
            'padding_top': self.h(10), # px
            'padding_right': self.w(10), # px
            'padding_bottom': self.h(10), # px
            'background_color': rgba2hex((20,20,20,0.6)), #rgba
            'border_width': self.w(4), # px
            'border_color': rgba2hex((255,255,255,1)), #rgba
            'border_radius': self.w(10), # px
        })
        self.countTextView = SimpleNamespace(**{
            'margin_left': self.w(20),
        })
        self.text = SimpleNamespace(**{
            'label_font_size': self.fs(20),
            'font_size': self.fs(20), # px
            'font_color': rgba2hex((255,255,255,1.0)), # px
            'font_family': 'dev/core/asset/font/NanumSquareNeoB.ttf',
            'label_font':ImageFont.load_default(),
            'font':ImageFont.load_default(),
        })
        self.default_icon = SimpleNamespace(**{
            'width': self.w(32),
            'height': self.h(32),
            'color': (255,255,255),
            'levelColor':(0,240,50),
        })
        self.user_icon = SimpleNamespace(**{
            
        })
        self.arrowLabelContainer = SimpleNamespace(**{
            'width': self.w(140),
            'height': self.h(60),
            'margin_bottom': self.h(20),
            'padding_left': self.w(10), # px
            'padding_top': self.h(10), # px
            'padding_right': self.w(10), # px
            'padding_bottom': self.h(10), # px
            'background_color': rgba2hex((0,210,240,0.3)),
            'border_color': rgba2hex((255,255,255,0.7)),
            'border_width': self.w(4), # px
            'border_radius': self.w(10), # px
        })
        self.arrowPoint = SimpleNamespace(**{
            'width': self.w(20),
            'height': self.h(40),
        })
        self.sexuality_icon = SimpleNamespace(**{
            'margin_right': self.w(10),
        })

        self.smokingContainer = SimpleNamespace(**{
            'background_color': rgba2hex((0,30,240,0.5)),
            'border_color': rgba2hex((0,0,255,0.5)),
        })
        self.bboxContainer = SimpleNamespace(**{
            'background_colors': SimpleNamespace(**{
                'fight':rgba2hex((200,40,255,0.7)),
                'faint':rgba2hex((0,200,255,0.7)),
            }),
            'border_colors': SimpleNamespace(**{
                'fight':rgba2hex((200,40,255,0.7)),
                'faint':rgba2hex((0,200,255,0.7)),
            }),
            'border_width': self.w(10), # px
            'border_radius': self.w(10), # px
        })
        self.bboxLabel = SimpleNamespace(**{
            'width': self.w(150),
            'height': self.h(50),
            'padding_left': self.w(10), # px
            'padding_top': self.h(10), # px
            'padding_right': self.w(10), # px
            'padding_bottom': self.h(0), # px
        })
        self.alertFont = SimpleNamespace(**{
            'font_family':'dev/core/asset/font/NanumSquareNeoEB.ttf',
            'font_size': self.fs(40),
            'font_color': rgba2hex((0,0,255,1.0)), # px
            'font':ImageFont.load_default()
        })
        self.warnFont = SimpleNamespace(**{
            'font_color': rgba2hex((0,200,255,1.0)), # px
        })
        self.alertContainer = SimpleNamespace(**{
            'border_width': self.w(10),
            'border_color': rgba2hex((0,0,255,0.7)),
            'background_color': rgba2hex((0,0,0,0.1)),
        })
        self.alertBox = SimpleNamespace(**{
            'width': self.w(300),
            'height': self.h(100),
            'padding_left': self.w(20), # px
            'padding_top': self.h(20), # px
            'padding_right': self.w(20), # px
            'padding_bottom': self.h(20), # px
            'border_width': self.w(10),
            'border_color': rgba2hex((0,0,255,0.7)),
            'background_color': rgba2hex((50,50,210,0.2)),
            
        })
        self.alert_icon = SimpleNamespace(**{
            'width': self.w(64),
            'height': self.h(64),
            'color': (0,0,255),
        })
        self.fire = SimpleNamespace(**{
            'default':SimpleNamespace(**{
                'border_width': self.w(10),
                'background_color': rgba2hex((0,0,0,0.1)),
                'padding_left': self.w(200), # px
                'padding_top': self.h(20), # px
                'padding_right': self.w(200), # px
                'padding_bottom': self.h(200), # px
            }),
            'default_icon':SimpleNamespace(**{
                'width': self.w(64),
                'height': self.h(64),
            }),
            'default_box':SimpleNamespace(**{
                'width': self.w(600),
                'height': self.h(200),
                'padding_left': self.w(20), # px
                'padding_top': self.h(20), # px
                'padding_right': self.w(20), # px
                'padding_bottom': self.h(20), # px
            }),
            'alert':SimpleNamespace(**{
                'default':SimpleNamespace(**{
                   'border_color': rgba2hex((0,0,255,0.7)), 
                }),
                'container':SimpleNamespace(**{
                    
                }),
                'box':SimpleNamespace(**{
                    'background_color': rgba2hex((50,50,210,0.2)),
                }),
                'icon':SimpleNamespace(**{
                    'color': (0,0,255)
                }),
                'text':SimpleNamespace(**{
                    'color': rgba2hex((0,0,255,1.0))
                })
            }),
            'warn':SimpleNamespace(**{
                'default':SimpleNamespace(**{
                   'border_color': rgba2hex((0,200,255,0.7)), 
                }),
                'container':SimpleNamespace(**{
                    
                }),
                'box':SimpleNamespace(**{
                    'background_color': rgba2hex((30,200,255,0.2)),
                }),
                'icon':SimpleNamespace(**{
                    'color': (0,200,255)
                }),
                'text':SimpleNamespace(**{
                    'color': rgba2hex((0,200,255,1.0))
                })
            }),
        })


def getColorLevel(image):
    import matplotlib.pyplot as plt
    # NumPy 배열로 변환
    image_array = np.array(image)
    
    # 각 채널 분리 (R, G, B, A)
    alpha = image_array[...,3]

    # 분석: RGB 값 범위와 밝기 값 분포
    r, g, b = image_array[..., 0], image_array[..., 1], image_array[..., 2]
    weights = np.array([0.2989,0.5870,0.1140])  # 밝기 기준 계산 (Luma 변환)
    
    intensity = np.dot(image_array[..., :3], weights)

    # 픽셀 값의 범위 출력
    print(f"R min: {r.min()}, R max: {r.max()}")
    print(f"G min: {g.min()}, G max: {g.max()}")
    print(f"B min: {b.min()}, B max: {b.max()}")
    print(f"Intensity min: {intensity.min()}, Intensity max: {intensity.max()}")
    print(f"Alpha min: {alpha.min()}, Alpha max: {alpha.max()}")

    # 밝기 값 분포 시각화
    plt.hist(intensity.flatten(), bins=50, color="gray", alpha=0.7, label="Intensity")
    plt.title("Brightness Distribution")
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.legend()
    plt.show()

def setColorLevel(image,color, levelColor=None,levelValue=0):
    image_array = np.array(image)
    black_mask = (image_array[:, :, :3].sum(axis=-1) == 0)  # 검은색 요소 마스크
    if levelValue > 0 and levelColor is not None:
        height_mask = np.arange(image_array.shape[0])[:, None] > levelValue
        image_array[black_mask & ~height_mask,:3] = color
        image_array[black_mask & height_mask, :3] = levelColor
    else:
        image_array[black_mask,:3] = color
    # intensity = np.einsum('ijk,k->ij', image_array[..., :3], [0.2989, 0.5870, 0.1140])
    # alpha = image_array[...,3]
    # image_array[(intensity < 50) & (alpha > 0),:3] = color
    
    new_image = Image.fromarray(image_array, "RGBA")
    return new_image


def rgb2hex(value:Union[Tuple[int,int,int],Tuple[int,int,int,float]]):
    if len(value) == 4:
        value = value[:3]
    assert 0<=value[0]<=255, 'should be 0~255'
    assert 0<=value[1]<=255, 'should be 0~255'
    assert 0<=value[2]<=255, 'should be 0~255'
    return "#{:02X}{:02X}{:02X}".format(value[0], value[1], value[2]).lower()
    
def rgba2hex(value:Tuple[int,int,int,float]):
    assert 0<=value[0]<=255, 'should be 0~255'
    assert 0<=value[1]<=255, 'should be 0~255'
    assert 0<=value[2]<=255, 'should be 0~255'
    assert 0<=value[3]<=1.0, 'should be 0~1'
    alpha = int(value[3]*255)
    return "#{:02X}{:02X}{:02X}{:02X}".format(value[0], value[1], value[2], alpha).lower()

def hex2rgba(hex_code: str):
    assert len(hex_code) == 9 and hex_code.startswith("#"), "Invalid HEX code format"
    r = int(hex_code[1:3], 16)  # 16진수 -> 10진수
    g = int(hex_code[3:5], 16)
    b = int(hex_code[5:7], 16)
    a = int(hex_code[7:9], 16) / 255.0  # A 값은 0~255를 0~1로 변환
    
    return r, g, b, round(a, 2)  # RGBA 반환 (A는 소수점 둘째 자리)

def add(l1,l2):
    return list(map(_add,l1,l2))

def sub(l1,l2):
    return list(map(_sub,l1,l2))

def mul(l1,l2):
    return list(map(_mul,l1,l2))

def getVerticalAlign(l1,l2):
    h = int((l1[3]+l1[1]-l2[1])/2)
    return [l1[0],h]