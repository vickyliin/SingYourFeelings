from bs4 import BeautifulSoup
import requests
import re
def pagelink(url):
    result=requests.get(url)
    page=result.text
    doc=BeautifulSoup(page,"html.parser")
    titlelist=doc.find_all('a')
    titlelinks=[title.get('tppabs') for title in titlelist]
    titlelinks=[title.replace('www.tacocity.com.tw/ala/lyrics/lych',\
                              'sql.jaes.ntpc.edu.tw/javaroom/midi/alas')\
                for title in titlelinks]
    titlelinks=[re.sub('ch\d','Ch',title) for title in titlelinks]
    return titlelinks
def songtext(url):
    result=requests.get(url)
    page=result.text.encode('ISO-8859-1').decode('big5')
    doc=BeautifulSoup(page,"html.parser")
    test1=doc.find_all('td')
    titlyr=[t.text for t in test1]
    title=titlyr.pop(0)
    title=re.sub('\u3000','',title)
    titletxt=title+'.txt'
    lyr='\n'.join(titlyr)
    lyr=lyr.strip('\n')
    fout=open(titletxt,'wt')
    fout.write(lyr)
    fout.close()
    song=doc.find('bgsound')
    songurl=song.get('src')
    songurl='http://www.tacocity.com.tw/ala/ch'+songurl
    titlemid=title+'.mid'
    urllib.request.urlretrieve(songurl,titlemid)
    
    
