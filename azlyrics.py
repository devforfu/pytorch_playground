import os
import time
import argparse
import configparser
from urllib.parse import urljoin
from string import ascii_letters, digits

import requests
import numpy as np
from bs4 import BeautifulSoup


# noinspection PyBroadException
class AZLyricsParser:
    """
    A simple scrapper to parse content of AZLyrics site.

    The scrapper tries to gather texts without increasing server's load too
    much, and makes effort to prevent getting into black list by making random
    long delays between requests.
    """
    base_url = 'https://www.azlyrics.com'

    def __init__(self, throttling=0.5, proxy=None):
        self.throttling = throttling
        self.proxy = proxy

    def build_songs_list(self, artist: str):
        name = normalize(artist)
        first_letter = name[0]
        albums_url = f'{self.base_url}/{first_letter}/{name}.html'
        try:
            r = requests.get(albums_url, proxies=self.proxy)
        except:
            print('Error: Cannot build songs list! Connection rejected')
            return

        page_content = r.text
        tree = BeautifulSoup(page_content, 'html.parser')
        albums = tree.find('div', id='listAlbum')
        if albums is None:
            print('Albums were not found')
            return

        songs = []
        for tag in albums.find_all('a'):
            link = tag.attrs.get('href')
            if not link:
                continue
            songs.append((tag.text, urljoin(self.base_url, link)))

        return songs

    def parse_songs(self, songs: list):
        print(f'Number of songs to parse: {len(songs):d}')
        texts = []
        for i, (title, url) in enumerate(songs):
            print(f'Parsing song: {url}')
            wait_time = 10
            while True:
                try:
                    text = self.parse_song(url)
                    break
                except:
                    print('Cannot parse song! Connection rejected')
                    print(f'Trying again after a period of delay equal '
                          f'to {wait_time:d} seconds')
                    time.sleep(wait_time)
                    wait_time *= 2
            texts.append((title, text))
            wait_time = np.random.normal(self.throttling, 3)
            print(f'Waiting for {wait_time:2.2f} seconds...')
            time.sleep(wait_time)
        return texts

    def parse_song(self, url):
        r = requests.get(url, proxies=self.proxy)
        page_content = r.text
        tree = BeautifulSoup(page_content, 'html.parser')
        lyrics = tree.find('div', {'id': None, 'class': None})
        if lyrics is None:
            return None
        text = lyrics.text.strip().replace('\r\n', '\n')
        return text


def normalize(string, domain=ascii_letters+digits):
    return ''.join([char for char in string if char in domain]).lower()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--artist',
        default='Black Sabbath',
        help='an artist whose songs to parse'
    )
    parser.add_argument(
        '-p', '--proxy',
        default=None,
        help='proxy configuration (if required)'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='path to folder with downloaded songs'
    )
    parser.add_argument(
        '-t', '--throttling',
        default=10.0,
        type=float,
        help='base throttling value used to define delay between requests'
    )

    args = parser.parse_args()

    if args.proxy is not None:
        conf = configparser.ConfigParser()
        conf.read(args.proxy)
        proxy = dict(conf['proxy'])
        url = 'socks5://{username}:{password}@{host}:{port}'.format(**proxy)
        args.proxy = {'http': url, 'https': url}

    if args.output is None:
        args.output = os.path.expanduser(f'~/data/azlyrics/{args.artist}')

    return args


def main():
    args = parse_args()

    parser = AZLyricsParser(throttling=args.throttling, proxy=args.proxy)
    songs = parser.build_songs_list(args.artist)

    if not songs:
        return

    folder = args.output
    texts = parser.parse_songs(songs)
    if not os.path.exists(folder):
        os.makedirs(folder)

    index_path = os.path.join(folder, 'songs.csv')
    with open(index_path, 'w') as index_file:
        for i, (title, text) in enumerate(texts):
            index_file.write(f'{i},{title}\n')
            filename = os.path.join(folder, f'{i}.txt')
            with open(filename, 'w') as text_file:
                text_file.write(text + '\n')

    print(f'Completed! Index path: {index_path}')


if __name__ == '__main__':
    main()