#!/env/bin python3
from flask import Flask,jsonify,request, json
from flask_restful import Resource,Api
from flask import Blueprint
from flask_cors import CORS
from Activity import Activity
import numpy as np
import pandas as pd

app=Flask(__name__)

# activate VE: source ./env/bin/activate
#run backend: /home/lahiru/Documents/Git/BackEnd/env/bin/python3 /home/lahiru/Documents/Git/BackEnd/app.py

# Cross-Origin Resource Sharing <- allows react to access the API
CORS(app)
cors=CORS(app,resources={
    r"/*":{
        "origins":"localhost"
    }
})

api=Api(app)
 
data = open("test.txt", "r")
content = data.read()
users = content.splitlines() #have to change it to CSI data
data.close()

#specify user data file location
userDataFile='/home/lahiru/Documents/Git/Data/userData.npy'

class Index(Resource):
    def get(self):
        current_state = pd.read_csv('/home/lahiru/Documents/Git/BackEnd/test2.csv')
        return ({"activity":str(current_state['Activity'][0]),"user":str(current_state['User'][0]),"location":str(current_state['Location'][0])})

    def post(self):
        from Authentication import addUser
        new_user = request.get_json()
        with open("test.txt","a") as fo:
            fo.write(new_user['user'])
            fo.write("\n")
        addUser(new_user['user'], users)
        return new_user['user']

api.add_resource(Index,'/get')


if __name__=='__main__':
    app.run(debug=True)
