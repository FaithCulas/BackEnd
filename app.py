from flask import Flask,jsonify,request
from flask_restful import Resource,Api
from Activity import Activity
from Authentication import Authentication

app=Flask(__name__)
api=Api(app)

class Index(Resource):
    def get (self):
        activity=Activity()
        authentication=Authentication()
        print(activity)
        return ({"activity":str(activity),"user":str(authentication),"location":"home"})



api.add_resource(Index,'/get')

if __name__=='__main__':
    app.run(debug=True)