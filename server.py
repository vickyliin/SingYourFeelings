#!/usr/bin/env python
import os
import uuid
from subprocess import Popen, DEVNULL
import logging

import asyncio
import websockets

import model

logging.getLogger('websockets.protocol').setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.ERROR)
logging.getLogger('jseg.jieba').setLevel(logging.ERROR)

model.use_cuda = False
outpath = 'gui/midifiles'
para = 'model/test.para'

cmd = 'watch -n 300 find {}/ -mmin +5 -delete'.format(outpath)
cleaner = Popen(cmd.split(), stdout=DEVNULL, stderr=DEVNULL)
async def midigen(websocket, path):
    while True:
        text = await websocket.recv()
        print("< {}".format(text))
        midi = tr.translate(text)
        filename = '%s/%s.mid' % (outpath, uuid.uuid1())
        os.makedirs(outpath, exist_ok=True)
        with open(filename, 'wb') as f:
            midi.writeFile(f)

        await websocket.send(os.path.basename(filename))
        print("> {}".format(filename))


tr = model.Translator()
sd = model.load(para)
tr.load_state_dict(sd)

start_server = websockets.serve(midigen, '0.0.0.0', 5678, timeout=86400)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
