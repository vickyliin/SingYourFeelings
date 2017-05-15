#!/usr/bin/env python
from bs4 import BeautifulSoup
import requests
import midi
import logging
fmt = '%(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=fmt, filename='crawler.log', filemode='w')
logger = logging.getLogger('MIDIDownloader')
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(fmt))
logger.addHandler(sh)

import sys
if len(sys.argv) > 1:
    PATH = sys.argv[1]
else:
    PATH = '../data/raw'

def mdownload(path):
    for pagelink in pagelinks:
        titlelinks=midi.songlink(pagelink)
        for songurl in titlelinks:
            try:
                midi.songtext(songurl,path)
            except Exception as e:
                logger.error(songurl)
                logger.error(e)

result=requests.get('http://sql.jaes.ntpc.edu.tw/javaroom/midi/alas/Ch/ch.htm')
page=result.text
doc=BeautifulSoup(page,"html.parser")
pagelist=doc.find_all('td')
pagelinks=[page.find('a') for page in pagelist]
while None in pagelinks:
    pagelinks.remove(None)
pagelinks=[page.get('href') for page in pagelinks]
pagelinks=['http://sql.jaes.ntpc.edu.tw/javaroom/midi/alas/Ch/'\
           +page for page in pagelinks]

mdownload(PATH)
