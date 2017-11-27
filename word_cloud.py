import sys
import wordcloud
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter


def read_mask(mask_file_name):
    """주어진 이름의 파일에서 마스킹 이미지 데이터를 읽어서 돌려준다."""
    mask = np.array(Image.open(mask_file_name))

    return mask


def draw_word_clouds(word_count, mask):
    font_path = get_font_path()
    cloud = build_cloud(word_count, font_path, mask)
    show_cloud(cloud, mask)


def get_font_path():
    """플랫폼에 따라 글꼴 경로를 설정한다."""

    if sys.platform == "win32" or sys.platform == "win64":
        font_path = "C:/Windows/Fonts/malgun.ttf"
    elif sys.platform == "darwin":
        font_path = "/Library/Fonts/AppleGothic.ttf"

    return font_path


def build_cloud(word_count, font_path, mask):
    """주어진 어휘 계수 결과와 글꼴 경로를 이용하여 워드 클라우드를 생성하여 돌려준다."""
    NUM_WORDS = 200
    cloud_gen = wordcloud.WordCloud(background_color="white",
                                    font_path=font_path, max_words=NUM_WORDS,
                                    mask=mask, collocations=False)
    cloud = cloud_gen.generate_from_frequencies(word_count)

    return cloud


def show_cloud(cloud, mask):
    """워드 클라우드를 화면에 표시한다."""

    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def WC(word_counts, mask_file_path, option_pretty=True):
    mask_file_name = mask_file_path
    mask = read_mask(mask_file_name)
    if option_pretty:
        word_count = Counter({l.split('/')[0]: c for l, c in word_counts.items()})
    else:
        word_count = Counter({l: c for l, c in word_counts.items()})
    draw_word_clouds(word_count, mask)