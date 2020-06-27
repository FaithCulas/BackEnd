from flask import Flask,jsonify,request
from flask_restful import Resource,Api
from Activity import Activity

app=Flask(__name__)
api=Api(app)

class Index(Resource):
    def get (self):
        activity=Activity()
        print(activity)
        return ({"activity":str(activity),"user":"faith","location":"home"})



api.add_resource(Index,'/get')

if __name__=='__main__':
    app.run(debug=True)