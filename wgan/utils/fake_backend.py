from aiohttp import web
from PIL import Image
from io import BytesIO
from datetime import datetime
from os.path import join, exists, split, splitext

import numpy as np
import base64
import os
import io
import json
from misc import to_hint_layer

f_json = "../samples/jsonExample.json"
f_flat = "../samples/flat png/0004_back_flat.png"
# load img
flat = np.array(Image.open(f_flat))
_, name = split(f_flat)
name, _ = splitext(name)
# load json
with open(f_json, 'r') as f:
    labels = json.load(f)
# find label
label = None
for l in labels:
    if l["file"] == name:
        label = l
        break

routes = web.RouteTableDef()

@routes.get('/')
# seems the function name is not that important?
async def hello(request):
    return web.Response(text="ShadowMagic API server is running")

## Add more API entry points
@routes.post('/add_shadow')
async def add_shadow( request ):
    data = await request.json()
    try:
        data = json.loads(data)
    except:
        print("got dict directly")
    
    # convert to json
    print("Log:\treceive input")
    
    shad = Image.open("../samples/flat png/0004_back_shadow.png")
    result = {}
    result['shadow'] = to_base64(shad)
    result['hint'] = to_base64(to_hint_layer(flat, label))

    return web.json_response( result )

def to_base64(array):
    '''
    A helper function to convert numpy array to png in base64 format
    '''
    with io.BytesIO() as output:
        if type(array) == np.ndarray:
            Image.fromarray(array).save(output, format='png')
        else:
            array.save(output, format='png')
        img = output.getvalue()
    img = base64.encodebytes(img).decode("utf-8")
    return img


def main():
    app = web.Application(client_max_size = 1024 * 1024 ** 2)
    app.add_routes(routes)
    web.run_app(app)

if __name__ == '__main__':
    main()
