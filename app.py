import os
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
from flask_marshmallow import Marshmallow
from flask_sqlalchemy import SQLAlchemy
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from sqlalchemy.sql import func

import constants

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

# chat_history = []

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = TextLoader("data/data.txt")  # Use this line if you only need data.txt
    # loader = DirectoryLoader("data/") # Use this line if you need all the files in the data folder
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

# Creation the Flask app here

app = Flask(__name__)
CORS(app)
basedir = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = \
    'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(100), nullable=False)
    lastname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    phone = db.Column(db.String(20))
    age = db.Column(db.Integer)
    address = db.Column(db.String(100))
    bloodgroup = db.Column(db.String(10))
    created_at = db.Column(db.DateTime(timezone=True),
                           server_default=func.now())
    bio = db.Column(db.Text)

    def __repr__(self):
        return f'<Patient {self.firstname}>'


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text)
    time = db.Column(db.String(50))
    source = db.Column(db.String(10), nullable=False)
    patient_email = db.Column(db.String(80), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True),
                           server_default=func.now())

    def __repr__(self):
        return f'<Message {self.message}>'


ma = Marshmallow(app)


class PatientSchema(ma.Schema):
    class Meta:
        fields = ('id', 'firstname', 'lastname', 'email', 'password', 'phone', 'age', 'address', 'bloodgroup', 'bio', 'created_at')


patient_schema = PatientSchema()
patients_schema = PatientSchema(many=True)


class MessageSchema(ma.Schema):
    class Meta:
        fields = ('id', 'text', 'time', 'source', 'patient_email', 'created_at')


message_schema = MessageSchema()
messages_schema = MessageSchema(many=True)

chat_history = []


# Define the /api route to handle POST requests
@app.route("/api/chatbot", methods=["POST"])
def api():
    # Get the message from the POST request
    data = request.json
    messageReceived = request.json.get("text")
    messages = db.session.execute(db.select(Message).where(Message.patient_email == data['patient_email']).order_by(Message.id)).scalars().all()
    messages = [message_schema.dump(item) for item in messages]
    print("-----------------------------------------------------")
    print("messages size = ", len(messages))
    print("-----------------------------------------------------")
    if len(messages) > 16:
        messages = messages[-16:]
        print("-----------------------------------------------------")
        print("new messages size = ", len(messages))
        print("-----------------------------------------------------")
    cpt = 0
    while cpt < len(messages):
        if messages[cpt]['source'] == "user":
            chat_history.append((messages[cpt]['text'], messages[cpt+1]['text']))
            cpt += 2
        else:
            cpt += 1
    print("-----------------------------------------------------")
    print("chat_history size => ", len(chat_history))
    print("-----------------------------------------------------")
    print("chat_history in db => ", chat_history)
    print("Nouveau message => ", messageReceived)
    result = chain({"question": messageReceived, "chat_history": chat_history})
    print("-----------------------------------------------------")
    print("result => ", result)

    if result['answer']:
        # chat_history.append((messageReceived, result['answer']))
        db.session.add(Message(text=messageReceived, time=datetime.now(), source="user", patient_email=data['patient_email']))
        db.session.add(Message(text=result['answer'], time=datetime.now(), source="ia", patient_email=data['patient_email']))
        db.session.commit()
        # return the response as JSON
        return {"message": result['answer']}
    else:
        return 'Failed to Generate response!'


@app.route("/patients", methods=["GET"])
def user_list():
    patients = db.session.execute(db.select(Patient).order_by(Patient.id)).scalars().all()
    return {"patients": [patient_schema.dump(item) for item in patients], "code": 200}


@app.route("/register", methods=["POST"])
def user_create():
    data = request.json
    print("data => ", data)
    patient = Patient(firstname=data['firstname'], lastname=data['lastname'], email=data['email'],password=data['password'])
    db.session.add(patient)
    db.session.commit()
    return {"patient": patient_schema.dump(patient), "code": 200}


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    print("data => ", data)
    patient = db.session.execute(db.select(Patient).where(Patient.email == data['email'])).scalar_one_or_none()
    if patient:
        if patient.password == data['password']:
            return {"patient": patient_schema.dump(patient), "code": 200}
        else:
            return {"patient": None, "code": 401}
    else:
        return {"patient": None, "code": 401}


def exist_patient_by_email(email):
    patient = db.session.execute(db.select(Patient).where(Patient.email == email)).scalar_one_or_none()
    if patient:
        return True
    else:
        return False


@app.route("/messages", methods=["GET"])
def message_list():
    messages = db.session.execute(db.select(Message).order_by(Message.id)).scalars().all()
    return {"messages": ([message_schema.dump(item) for item in messages]), "code": 200}


@app.route("/messages/<patient_email>", methods=["GET"])
def message_list_by_patient(patient_email):
    messages = db.session.execute(db.select(Message).where(Message.patient_email == patient_email).order_by(Message.id)).scalars().all()
    return {"messages": ([message_schema.dump(item) for item in messages]), "code": 200}


@app.route("/messages/clear/<patient_email>", methods=["GET"])
def message_clear_by_patient(patient_email):
    messages = db.session.execute(db.select(Message).where(Message.patient_email == patient_email).order_by(Message.id)).scalars().all()
    for message in messages:
        db.session.delete(message)
    db.session.commit()
    return {"message": "clear succeffully !", "code": 200}



if __name__ == '__main__':
    with app.app_context():
        db.drop_all()
        db.create_all()
        for patient in constants.defaultPatients:
            if not exist_patient_by_email(patient['email']):
                db.session.add(
                    Patient(firstname=patient['firstname'], lastname=patient['lastname'], email=patient['email'],
                              phone=patient['phone'], bloodgroup=patient['bloodgroup'],
                              password=patient['password'], address=patient['address'],
                              age=patient['age'], bio=patient['bio']) )
        db.session.commit()
    app.run(host="localhost", port=9999)
