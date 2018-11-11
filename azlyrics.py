"""
An AZLyrics parsing tool.

The tool downloads lyrics from the website and saves each song into separate
file. It also creates a CSV file with song titles.

Note that the downloaded texts can be used only for educational and personal
purposes. Please visit the website to get familiar with license and privacy
policy: https://www.azlyrics.com
"""
import os
import time
import argparse
import configparser
from pathlib import Path
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

    Parameters:
        throttling: Mean value of normal distribution used to generate random
            delays between HTTP requests.
        proxy: Optional dictionary with SOCKS proxy definition.

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
        help='an artist whose songs to parse; is used only if -f is missing'
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
    parser.add_argument(
        '-f', '--file',
        default=None,
        help='path to file with artist names'
    )
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help='load songs texts even if folder with artist name already exists'
    )

    args = parser.parse_args()

    if args.proxy is not None:
        conf = configparser.ConfigParser()
        conf.read(args.proxy)
        proxy = dict(conf['proxy'])
        url = 'socks5://{username}:{password}@{host}:{port}'.format(**proxy)
        args.proxy = {'http': url, 'https': url}

    args.output = Path(args.output or '~/data/azlyrics').expanduser()

    if args.file is None:
        artists = [args.artist]

    else:
        path = Path(args.file)
        if not path.exists():
            parser.error(f'File does not exist: {args.file}')

        artists = {line.strip() for line in Path(args.file).open()}

        if not args.force_reload:
            for dirname in args.output.iterdir():
                artist = dirname.stem
                if artist in artists:
                    print(f'Artist folder already exists: {artist}')
                    artists.remove(artist)

    args.artists = sorted(artists)

    return args


def main():
    print('Instantiating lyrics parser')

    args = parse_args()
    parser = AZLyricsParser(throttling=args.throttling, proxy=args.proxy)
    artists = args.artists

    for i, artist in enumerate(artists, 1):
        print(f'Building list of songs URLs for artist {artist}',
              f'({i} of {len(artists)})')

        songs = parser.build_songs_list(artist)
        if not songs:
            print('Songs not found. Skipping...')
            continue

        print(f'Parsing collected songs ({len(songs)} total)')
        folder = args.output/artist
        texts = parser.parse_songs(songs)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)

        index_path = Path(folder).joinpath('songs.csv')
        with index_path.open('w') as index_file:
            for j, (title, text) in enumerate(texts):
                index_file.write(f'{j},{title}\n')
                with (folder/f'{j}.txt').open('w') as text_file:
                    text_file.write(text + '\n')

        print(f'Completed! Index path: {index_path}')


if __name__ == '__main__':
    main()
