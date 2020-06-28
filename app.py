from flask import Flask,jsonify,request, json
from flask_restful import Resource,Api
from Activity import Activity
import threading


app=Flask(__name__)
api=Api(app)

data = open("test.txt", "r")
content = data.read()
users = content.split(",")

class Index(Resource):
    def get (self):
        activity=Activity()
        from Authentication import Authentication
        authentication=Authentication()
        print(activity)
        return ({"activity":str(activity),"user":str(authentication),"location":"home"})

    def post(self):
        new_user = request.get_json()
        with open("test.txt","a") as fo:
            fo.write(new_user['user'])
            fo.write("\n")

        return new_user['user']

api.add_resource(Index,'/get')


if __name__=='__main__':
    app.run(debug=True)
   