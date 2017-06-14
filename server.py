#!/usr/bin/env python
import os
import uuid
from subprocess import Popen

import asyncio
import websockets

import model

model.use_cuda = False
outpath = 'midifiles'
para = 'model/test.para'

cleaner = Popen('watch -n 3600 find midifiles/ -mmin +60 -delete'.split())

async def hello(websocket, path):
    while True:
        text = await websocket.recv()
        print("< {}".format(text))
        midi = tr.translate(text)
        filename = '%s/%s.mid' % (outpath, uuid.uuid1())
        with open(filename, 'wb') as f:
            midi.writeFile(f)

        await websocket.send(os.path.basename(filename))
        print("> {}".format(filename))


tr = model.Translator()
sd = model.load(para)
tr.load_state_dict(sd)

os.makedirs(outpath, exist_ok=True)

start_server = websockets.serve(hello, '0.0.0.0', 5678)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
