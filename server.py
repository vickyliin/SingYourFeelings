#!/usr/bin/env python
import os
import uuid

import asyncio
import websockets

import model

model.use_cuda = False
outpath = 'midifiles'
para = '/home/cwtsai/class/computational-information/test.para'

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

start_server = websockets.serve(hello, 'localhost', 5678)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
