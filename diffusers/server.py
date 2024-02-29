import numpy as np
import base64
import io
from io import BytesIO
from PIL import Image
from aiohttp import web
from shadow_magic_api_async import run_single

routes = web.RouteTableDef()
# https://github.com/pytorch/pytorch/issues/1494

@routes.get('/')
# seems the function name is not that important?
async def hello(request):
    return web.Response(text="Shadowing API server is running")

@routes.post('/shadowsingle')
async def shadowsingle( request ):
    data = await request.json()
    try:
        data = json.loads(data)
    except:
        print("got dict directly")
    
    # json to data
    flat = np.array(to_pil(data['flat']))
    line = np.array(to_pil(data['line']))
    color = np.array(to_pil(data['color']))

    direction = str(data['direction'])
    user = str(data['user'])
    name = str(data['name'])

    shadows = await run_single(user, flat, line, color, name, direction)

    result = {}
    result['user'] = user
    result['direction'] = direction
    result['name'] = name
    for i in range(len(shadows)):
        result['shadow_%d'%i] = to_base64(shadows[i])
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

def to_pil(byte):
    '''
    A helper function to convert byte png to PIL.Image
    '''
    byte = base64.b64decode(byte)
    return Image.open(BytesIO(byte))

def main():
    app = web.Application(client_max_size = 1024 * 1024 ** 2)
    app.add_routes(routes)
    web.run_app(app, port=8000, keepalive_timeout = 500, shutdown_timeout= 500)

if __name__ == '__main__':
    main()
