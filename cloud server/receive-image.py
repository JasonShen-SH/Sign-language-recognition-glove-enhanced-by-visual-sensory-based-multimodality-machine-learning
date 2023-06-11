# prepare for receiving image

from flask_restful import Api, Resource
from flask import Flask, request

app = Flask(__name__)
api = Api(app)

class receive_pic(Resource):
    def put(self):
        img = request.get_data()
        with open("test.png", "wb") as f:
            f.write(img)
        return 0
        
api.add_resource(receive_pic,'/test')

if __name__ == '__main__':
    app_port = 7100
    app.run(host="0.0.0.0", port=app_port, debug=False)
    
# stop the kernel
