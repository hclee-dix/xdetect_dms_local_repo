import glob,os
#
import math
from typing import Union,Tuple
from operator import add as _add,sub as _sub,mul as _mul
#
from PIL import Image,ImageFont
import matplotlib.pyplot as plt
import numpy as np
#
from dev.schema.style import IStyle,IFontStyle,IIconStyle,IIcon


#
BASE_ASSET_PATH = 'dev/core/asset/'
BASE_FONT_PATH = BASE_ASSET_PATH+'font/'
BASE_ICON_PATH = BASE_ASSET_PATH+'icon/'
class XStyle:
    def __init__(self,meta):
        self.ref_width = 1920 # ref_width
        self.ref_height = 1080 # ref_height
        self.im_width = meta.width # image_width
        self.im_height = meta.height # image_height
        self.width_ratio = self.im_width/self.ref_width
        self.height_ratio = self.im_height/self.ref_height
        self.ratio = math.sqrt((self.im_width/self.ref_width)**2+(self.im_height/self.ref_height)**2)
        ####
        self.init_default_style()
        self.init_font()
        self.init_icon()
    
    def init_default_style(self):
        ## container
        self.default_container = IStyle(
            width=self.im_width,
            height=self.im_height,
            background_color=self.rgba2hex((0,0,0,0))
        )
        self.default_alert_container = IStyle(
            border_width=self.w(10),
            border_color=self.rgba2hex((0,0,255,0.7)),
            background_color=self.rgba2hex((0,0,0,0.1))
        )
        self.default_alert_box = IStyle(
            width=self.w(300),
            height=self.h(100),
            padding_left=self.w(20),
            padding_top=self.h(20),
            padding_right=self.w(20),
            padding_bottom=self.h(20),
            border_width=self.w(10),
            border_color=self.rgba2hex((0,0,255,0.7)),
            background_color=self.rgba2hex((50,50,210,0.2))
        )
        ## text font
        self.default_text_style = IFontStyle(
            font_color=self.rgba2hex((255,255,255,1.0)),
            font_family=self.getFontPath('NanumSquareNeoB.ttf'),
            font_size=self.fs(20)
        )
        self.default_text_warn_style = IFontStyle(
            font_color=self.rgba2hex((0,200,255,1.0)),
            font_family=self.getFontPath('NanumSquareNeoEB.ttf'),
            font_size=self.fs(40)
        )
        self.default_text_alert_style = IFontStyle(
            font_color=self.rgba2hex((0,0,255,1.0)),
            font_family=self.getFontPath('NanumSquareNeoEB.ttf'),
            font_size=self.fs(40)
        )
        ## icon
        self.default_icon_style = IIconStyle(
            width=self.w(32),
            height=self.h(32),
        )
        self.default_warn_icon_style = IIconStyle(
            width=self.w(64),
            height=self.h(64),
            color=(0,200,255),
        )
        self.default_alert_icon_style = IIconStyle(
            width=self.w(64),
            height=self.h(64),
            color=(0,0,255),
        )
        ## userCount
        self.user_count_style = IStyle(
            width=self.w(118),
            height=self.h(60),
            margin_left=self.w(10),
            margin_top=self.h(10),
            padding_left=self.w(10),
            padding_top=self.h(10),
            padding_right=self.w(10),
            padding_bottom=self.h(10),
            background_color=self.rgba2hex((20,20,20,0.6)),
            border_width=self.w(4),
            border_color=self.rgba2hex((255,255,255,1)),
            border_radius=self.w(10),
        )
    
    def init_font(self):
        self.default_text = ImageFont.truetype(self.default_text_style.font_family,self.default_text_style.font_size)
        self.default_warn_text = ImageFont.truetype(self.default_text_warn_style.font_family,self.default_text_warn_style.font_size)
        self.default_alert_text = ImageFont.truetype(self.default_text_alert_style.font_family,self.default_text_alert_style.font_size)

    def init_icon(self):
        resourcePathList = glob.glob(BASE_ICON_PATH+'*.png')
        icon_style = self.default_icon_style
        resourceMap = {}
        for resourcePath in resourcePathList:
            resourceName = os.path.splitext(os.path.basename(resourcePath))[0]
            resource = Image.open(resourcePath)
            if resourceName in ['warning']:
                resource = resource.resize((self.default_alert_icon_style.width,self.default_alert_icon_style.height))
            else:
                resource = resource.resize((icon_style.width,icon_style.height))
            resourceMap.update({os.path.splitext(os.path.basename(resourcePath))[0]:resource})
        self.icon = IIcon(src=resourceMap)
    
    def w(self,v):
        assert 0<=v<=self.ref_width
        return int(v*self.width_ratio)
    
    def h(self,v):
        assert 0<=v<=self.ref_height
        return int(v*self.height_ratio)
    
    def fs(self,v:int, min_size = 12):
        return int(v*self.ratio) or int(max(round(sum((self.ref_width,self.ref_height)) /2 * 0.035),min_size)*self.ratio)
    
    def add(self,l1,l2):
        return list(map(_add,l1,l2))

    def sub(self,l1,l2):
        return list(map(_sub,l1,l2))

    def mul(self,l1,l2):
        return list(map(_mul,l1,l2))
    
    def getFontPath(self,title:str):
        return BASE_FONT_PATH+title
    
    def getIconPath(self,title:str):
        return BASE_ICON_PATH+title
    
    def getOriginCoord(self,v):
        ratioMap = [1/self.width_ratio,1/self.height_ratio]*2
        return self.mul(v,ratioMap)
    
    def getBoxCoord(self,v):
        ratioMap = [1/self.width_ratio,1/self.height_ratio]*3
        w = v[2]-v[0]
        h = v[3]-v[1]
        v = v+[w,h]
        return self.mul(v,ratioMap)
    
    def getVerticalAlign(self,l1,l2):
        h = int((l1[3]+l1[1]-l2[1])/2)
        return [l1[0],h]
    
    def getTextSize(self,textbox):
        return [textbox[2]-textbox[0],textbox[3]-textbox[1]]
    
    def getColorLevel(self,image):
        
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

    def setColorLevel(self,image,color, levelColor=None,levelValue=0):
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


    def rgb2hex(self,value:Union[Tuple[int,int,int],Tuple[int,int,int,float]]):
        if len(value) == 4:
            value = value[:3]
        assert 0<=value[0]<=255, 'should be 0~255'
        assert 0<=value[1]<=255, 'should be 0~255'
        assert 0<=value[2]<=255, 'should be 0~255'
        return "#{:02X}{:02X}{:02X}".format(value[0], value[1], value[2]).lower()
        
    def rgba2hex(self,value:Tuple[int,int,int,float]):
        assert 0<=value[0]<=255, 'should be 0~255'
        assert 0<=value[1]<=255, 'should be 0~255'
        assert 0<=value[2]<=255, 'should be 0~255'
        assert 0<=value[3]<=1.0, 'should be 0~1'
        alpha = int(value[3]*255)
        return "#{:02X}{:02X}{:02X}{:02X}".format(value[0], value[1], value[2], alpha).lower()

    def hex2rgba(self,hex_code: str):
        assert len(hex_code) == 9 and hex_code.startswith("#"), "Invalid HEX code format"
        r = int(hex_code[1:3], 16)  # 16진수 -> 10진수
        g = int(hex_code[3:5], 16)
        b = int(hex_code[5:7], 16)
        a = int(hex_code[7:9], 16) / 255.0  # A 값은 0~255를 0~1로 변환
        
        return r, g, b, round(a, 2)  # RGBA 반환 (A는 소수점 둘째 자리)