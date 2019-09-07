#-*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm
reload(sys)
sys.setdefaultencoding("utf-8")

FLAGS = None


def draw_char_bitmap(ch, font, char_size, x_offset, y_offset):
    image = Image.new("RGB", (char_size, char_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    offset = font.getsize(ch)
    draw.text(((224-offset[0])/2, (224-offset[1])/2), ch, (0, 0, 0), font=font)
    gray = image.convert('RGB')
    bitmap = np.asarray(gray)
    return gray


def generate_font_bitmaps(chars, font_path, char_size, canvas_size, x_offset, y_offset):
    font_obj = ImageFont.truetype(font_path, char_size)
    bitmaps = list()
    if not os.path.exists(os.path.join('font2',font_path.split('/')[4][:-4])):
        os.mkdir(os.path.join('font2',font_path.split('/')[4][:-4]))
    for c in chars:
        bm = draw_char_bitmap(c, font_obj, canvas_size, x_offset, y_offset)
        path = os.path.join('font2',font_path.split('/')[4][:-4],c)
        bm.save(path+'.png')
    return np.array(bitmaps)


def process_font(chars, font_paths, x_offset=5, y_offset=5, mode='target'):
    char_size = 180
    canvas = 224
    if mode == 'source':
        char_size *= 2
        canvas *= 2
    cnt = 0
    for font_path in font_paths:
        print(font_path,cnt)
        font_bitmaps = generate_font_bitmaps(chars, font_path, char_size,
                                             canvas, x_offset, y_offset)
        _, ext = os.path.splitext(font_path)
        if not ext.lower() in [".otf", ".ttf",".ttc"]:
            raise RuntimeError("unknown font type found %s. only TrueType or OpenType is supported" % ext)
        _, tail = os.path.split(font_path)
        cnt = cnt +1


def get_chars_set(path):
    """
    Expect a text file that each line is a char
    """
    chars = list()
    with open(path) as f:
        for line in f:
            line = u"%s" % line
            char = line.split()[0]
            chars.append(char)
    return chars

def load_global_charset():
    global CN_CHARSET, JP_CHARSET, KR_CHARSET, CN_T_CHARSET
    cjk = json.load(open('cjk.json'))
    CN_CHARSET = cjk["gbk"]
    JP_CHARSET = cjk["jp"]
    KR_CHARSET = cjk["kr"]
    CN_T_CHARSET = cjk["gb2312_t"]


if __name__ == '__main__':
    fonts_list = os.listdir('/home/cvml/한글 TTF')
    fonts_list = [os.path.join('/home/cvml/한글 TTF',font) for font in fonts_list]
    load_global_charset()
    process_font(KR_CHARSET[33:],fonts_list)

# load_global_charset()
# parser = argparse.ArgumentParser(description='Convert font to images')
# parser.add_argument('--src_font', dest='src_font', required=True, help='path of the source font')
# parser.add_argument('--dst_font', dest='dst_font', required=True, help='path of the target font')
# parser.add_argument('--filter', dest='filter', type=int, default=0, help='filter recurring characters')
# parser.add_argument('--charset', dest='charset', type=str, default='CN',
#                     help='charset, can be either: CN, JP, KR or a one line file')
# parser.add_argument('--shuffle', dest='shuffle', type=int, default=0, help='shuffle a charset before processings')
# parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
# parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
# parser.add_argument('--x_offset', dest='x_offset', type=int, default=20, help='x offset')
# parser.add_argument('--y_offset', dest='y_offset', type=int, default=20, help='y_offset')
# parser.add_argument('--sample_count', dest='sample_count', type=int, default=1000, help='number of characters to draw')
# parser.add_argument('--sample_dir', dest='sample_dir', help='directory to save examples')
# parser.add_argument('--label', dest='label', type=int, default=0, help='label as the prefix of examples')
#
# args = parser.parse_args()
#
# if __name__ == "__main__":
#     if args.charset in ['CN', 'JP', 'KR', 'CN_T']:
#         charset = locals().get("%s_CHARSET" % args.charset)
#     else:
#         charset = [c for c in open(args.charset).readline()[:-1].decode("utf-8")]
#     if args.shuffle:
#         np.random.shuffle(charset)
#     font2img(args.src_font, args.dst_font, charset, args.char_size,
#              args.canvas_size, args.x_offset, args.y_offset,
#              args.sample_count, args.sample_dir, args.label, args.filter)