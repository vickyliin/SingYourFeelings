#!/usr/bin/env python
import io
import logging
logging.getLogger('jseg.jieba').setLevel(logging.ERROR)

from flask import Flask, request, send_file

import model

para = 'model/test.para'
model.use_cuda = False
port = 80

static = 'gui'
app = Flask('syf-server', static_url_path='', static_folder=static)

def midigen(text):
    app.logger.debug("< {}".format(text))
    midi = tr.translate(text)

    buff = io.BytesIO()
    midi.writeFile(buff)
    return buff

def root():
    if 'text' in request.args:
        buff = midigen(request.args['text'])
        buff.seek(0)
        return send_file(buff, attachment_filename='melody.mid')
    else:
        return app.send_static_file('index.html')

tr = model.Translator()
sd = model.load(para)
tr.load_state_dict(sd)

app.route('/')(root)
app.route('/index.html')(root)
app.run(debug=True, port=port, threaded=True, host='0.0.0.0')
