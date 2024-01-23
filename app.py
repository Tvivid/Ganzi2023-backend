from datetime import timedelta, datetime


from fastapi import FastAPI

from fastapi import APIRouter, HTTPException
from fastapi import Depends

from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager
from pydantic import BaseModel
from ganzi import MyModel

from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from sqlalchemy.orm import Session
from starlette import status


from database import get_db
from user import user_crud, user_schema
from user.user_crud import pwd_context

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
SECRET_KEY = "4ab2fce7a6bd79e1c014396315ed322dd6edb1c5d975c6b74a2904135172c03c"
ALGORITHM = "HS256"

# app config
app = FastAPI(title="Ganzi API",
              description="API description",
              version="0.0.0",
              docs_url="/api",
              redoc_url="/redoc")

socket_manager = SocketManager(app=app)

origins = [
	"http://localhost",
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

ganzi = MyModel()

# basemodel

class SignUpForm(BaseModel):
	username: str
	email: str
	password: str


class LoginForm(BaseModel):
	email: str
	password: str


class ModelChangeForm(BaseModel):
	room: str
	id: str
	model: str

email_to_socket_mapping = dict()
socket_to_email_mapping = dict()
socket_to_model_mapping = dict()

# method
@app.get("/")
def root():
	return {"message": "Hello World"}


@app.post("/signup", status_code=status.HTTP_204_NO_CONTENT)
def user_create(_user_create: user_schema.UserCreate, db: Session = Depends(get_db)):
    user_crud.create_user(db=db, user_create=_user_create)



@app.post("/login", response_model=user_schema.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(),
                           db: Session = Depends(get_db)):

    # check user and password
    user = user_crud.get_user(db, form_data.username)
    if not user or not pwd_context.verify(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # make access token
    data = {
        "sub": user.username,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username
    }


@app.post("/model", status_code=200)
async def changemodel(form: ModelChangeForm):
	return {'message': "모델 변환 성공", 'model': form.model}

# socket_manager

@socket_manager.on('room:join')
async def join_room(sid, *args, **kwargs):
	data_dict = args[0]  # 튜플 안의 딕셔너리를 추출
	email = data_dict['email']
	room = data_dict['room']
	print(f'sid = {sid} emailId = {email}, roomId = {room}')
	email_to_socket_mapping[email] = sid
	socket_to_model_mapping[email] = "disney"
	socket_to_email_mapping[sid] = email
	await socket_manager.emit("user:joined", {'email': email, 'id': sid}, room)
	await socket_manager.enter_room(sid, room)
	await socket_manager.emit('room:join', args[0], sid)

@socket_manager.on('user:call')
async def call_user(sid, *args, **kwargs):
	data_dict = args[0]
	# print(f"call_user event data =  {data_dict}")
	to = data_dict['to']
	offer = data_dict['offer']
	print(f"from {sid} socket id = {to}")
	await socket_manager.emit('incomming:call', {'from': sid, 'offer': offer}, to)


@socket_manager.on('call:accepted')
async def call_accepted(sid, *args, **kwargs):
	data_dict = args[0]
	# print(f"call_accepted event data =  {data_dict}")
	to = data_dict['to']
	ans = data_dict['ans']
	print(f"emailId =  {to} socket id = {sid} ans = {ans}")
	await socket_manager.emit('call:accepted', {'from': sid, 'ans': ans}, to)


@socket_manager.on('peer:nego:needed')
async def peer_nego_needed(sid, *args, **kwargs):
	data_dict = args[0]
	# print(f"peer:nego:needed  {data_dict}")
	to = data_dict['to']
	offer = data_dict['offer']
	print(f"peer:nego:needed offer = {offer}")
	await socket_manager.emit('peer:nego:needed', {'from': sid, 'offer': offer}, to)

@socket_manager.on('peer:nego:done')
async def peer_nego_done(sid, *args, **kwargs):
	data_dict = args[0]
	to = data_dict['to']
	ans = data_dict['ans']
	print(f"peer:nego:done , ans = {ans}")
	# print(f"emailId =  {to} socket id = {sid} ans = {ans}")
	await socket_manager.emit('peer:nego:final', {'from': sid, 'ans': ans}, to)


@socket_manager.on('image')
async def handle_join(sid, *args, **kwargs):
	print(args[0])
	raw_data = args[0]['rawData']
	character = socket_to_model_mapping.get(sid)
	if character is None:
		character = "pixar"
	processed_img_data = args
	if character == "original": # 변환 안함
		await app.sio.emit("processed_image", processed_img_data)
		return;
	else:
		processed_img_data = await ganzi.image_handler(raw_data.split(',')[1], character) # (base64,ewlkjflwwljkfj9992
	await app.sio.emit("processed_image", processed_img_data)

@socket_manager.on('new-message')
async def handle_chat_message(sid, *args, **kwargs):
	email = socket_to_email_mapping.get(sid)
	msg = args[0]
	await app.sio.emit("new-message", {'email': email, 'msg': msg})