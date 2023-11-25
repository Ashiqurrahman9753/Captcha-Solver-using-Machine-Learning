#!/usr/bin/env python3

import os
import random
import argparse
import numpy
import cv2
from captcha.image import ImageCaptcha
from PIL import ImageFont

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int, default=128)
    parser.add_argument('--height', help='Height of captcha image', type=int, default=64)
    parser.add_argument('--min-length', help='Minimum length of captchas in characters', type=int, default=1)
    parser.add_argument('--max-length', help='Maximum length of captchas in characters', type=int, default=6)
    parser.add_argument('--count', help='How many captchas to generate', type=int, default=150000)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str, default='C:/Scalable/project_2/trainingcap1')
    parser.add_argument('--symbols', help='Characters to use in captchas', type=str, default='symbols.txt')
    parser.add_argument('--font', help='Path to a custom font file (woff)', type=str, default='C:\Scalable\project_2\EamonU.ttf')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    captcha_generator = ImageCaptcha(width=args.width, height=args.height, font_sizes=(40, 45, 50), fonts=[args.font])

    if args.font:
        custom_font = ImageFont.truetype(args.font, 40)
        captcha_generator.fonts = [custom_font]

    with open(args.symbols, 'r') as symbols_file:
        symbols = symbols_file.readline().strip()

    for i in range(args.count):
        captcha_length = random.randint(args.min_length, args.max_length)
        random_str = ''.join([random.choice(symbols) for _ in range(captcha_length)])
        image_path = os.path.join(args.output_dir, f'{random_str}.png')

        image = numpy.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)

if __name__ == '__main__':
    main()
