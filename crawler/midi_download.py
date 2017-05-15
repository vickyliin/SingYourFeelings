#!/usr/bin/env python
from bs4 import BeautifulSoup
import requests
import midi
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
for pagelink in pagelinks:
    titlelinks=midi.pagelink(pagelink)
    for songurl in titlelinks:
        try:
            midi.songtext(songurl)
        except Exception as e:
            print(songurl,'\n has some problems')
            print(e)
            
