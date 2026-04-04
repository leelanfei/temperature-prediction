#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据爬虫模块 - 郑州市气温数据爬取
从天气网站爬取指定年份的气温数据
"""

import requests
import csv
from lxml import etree


def get_weather(url):
    """
    爬取指定URL的天气数据

    Args:
        url: 天气数据URL

    Returns:
        list: 包含每日天气信息的字典列表
    """
    weather_info = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    resp = requests.get(url, headers=headers)
    resp_html = etree.HTML(resp.text)
    resp_list = resp_html.xpath("//ul[@class='thrui']/li")

    for li in resp_list:
        day_weather_info = {
            'date_time': li.xpath("./div[1]/text()")[0].split(' ')[0],
            'high': li.xpath("./div[2]/text()")[0].replace('℃', ''),
            'low': li.xpath("./div[3]/text()")[0].replace('℃', ''),
            'weather': li.xpath("./div[4]/text()")[0],
            'wind': li.xpath("./div[5]/text()")[0]
        }
        weather_info.append(day_weather_info)

    return weather_info


def crawl_weather_data(year=2024, city='zhengzhou', output_file='weather.csv'):
    """
    爬取全年天气数据并保存到CSV文件

    Args:
        year: 要爬取的年份
        city: 城市代码
        output_file: 输出文件名

    Returns:
        list: 包含全年天气数据的列表
    """
    weathers = []

    for month in range(1, 13):
        weather_time = f'{year}{month:02}'
        url = f'https://lishi.tianqi.com/{city}/{weather_time}.html'
        weather = get_weather(url)
        weathers.append(weather)
        print(f'已爬取 {year}年{month}月 数据')

    print('数据爬取完成!')

    # 数据写入CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['日期', '最高气温', '最低气温', '天气', '风向'])
        for month_weather in weathers:
            for day_weather_dict in month_weather:
                writer.writerow(list(day_weather_dict.values()))

    print(f'数据已保存到 {output_file}')
    return weathers


if __name__ == '__main__':
    # 爬取2024年郑州天气数据
    crawl_weather_data(year=2024, output_file='weather.csv')