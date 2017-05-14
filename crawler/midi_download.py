from bs4 import BeautifulSoup
import requests
import midi
result=requests.get('http://sql.jaes.ntpc.edu.tw/javaroom/midi/alas/Ch/ch.htm')
page=result.text
doc=BeautifulSoup(page,"html.parser")
pagelist=doc.find_all('td')
pagelinks=[page.find('a') for page in pagelist]

pagelinks=[page.get('href') for page in pagelist]
for pagelink in pagelinks:
    titlelinks=midi.titlelink(pagelink)
    for songurl in titlelinks:
        try:
            midi.songtext(songurl)
        except:
            print(Exception)
