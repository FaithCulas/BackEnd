from flask import Flask,jsonify,request,redirect,url_for
from flask_restful import Resource,Api
from flask import Blueprint
from Activity import Activity
app=Flask(__name__)
api=Api(app)
 

class Index(Resource):
    def get (self):
        activity=Activity()
        print(activity)
        return ({"activity":str(activity),"user":"faith","location":"home"})
    
        """     WORDS = []
            with open("A:\\GIT\\Backend\\env1\\test.csv2", "r") as file:
                for line in file.readlines():
                    WORDS.append(line.rstrip())
            print (WORDS)
            return WORDS """
    



api.add_resource(Index,'/get')

if __name__=='__main__':
    app.run(debug=True)
    