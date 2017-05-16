from bs4 import BeautifulSoup
import requests
import re
import urllib
import os
import logging
logger = logging.getLogger('MIDI')
def songlink(url):
    result=requests.get(url)
    page=result.text
    doc=BeautifulSoup(page,"html.parser")
    titlelist=doc.find_all('a')
    titlelinks=[title.get('href') for title in titlelist]
    titlelinks=['http://sql.jaes.ntpc.edu.tw/javaroom/midi/alas/Ch/'+\
                title for title in titlelinks]
    titlelinks=[re.sub('ch\d','Ch',title) for title in titlelinks]
    return titlelinks
def songtext(url,path):
    result=requests.get(url)
    try:
        page=result.text.encode('ISO-8859-1').decode('big5')
    except Exception:
        page=result.text.encode('ISO-8859-1').decode('hkscs')
    doc=BeautifulSoup(page,"html.parser")
    test1=doc.find_all('td')
    titlyr=[t.text for t in test1]
    title=''
    while title=='':
        title=titlyr.pop(0)
        title=re.sub('[":]','-',title)
        title = ''.join(title.split())
    titletxt=title+'.txt'
    lyr='\n'.join(titlyr)
    lyr=re.sub('\(轉載請標明出處，否則請勿引用\)','',lyr)
    lyr=lyr.strip('\n')
    lyr=re.sub('\xa0','',lyr)
    fout=open(os.path.join(path,titletxt),'wt')
    fout.write(lyr)
    fout.close()
    song=doc.find('bgsound')
    try:
        songurl=song.get('src')
        songurl='http://sql.jaes.ntpc.edu.tw/javaroom/midi/alas/Ch/'+songurl
        titlemid=title+'.mid'
        urllib.request.urlretrieve(songurl,os.path.join(path,titlemid))
    except Exception:
        try:
            test1=doc.find_all('a')
            b=test1[1].get('href')
            songurl='http://sql.jaes.ntpc.edu.tw/javaroom/midi/alas/Ch/'+b
            titlemid=title+'.mid'
            urllib.request.urlretrieve(songurl,os.path.join(path,titlemid))
        except Exception as pro:
            logger.error(title)
            logger.error(pro)
        
