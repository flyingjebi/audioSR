import runpod
import io
import time
import concurrent
import uuid
import json
from audiosr import build_model, super_resolution
import soundfile as sf
import numpy as np

#initialize firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore

#load credential and authenticate firebase
#cred=credentials.Certificate("./brainstorm-2dcc5-firebase-adminsdk-7hskt-73e0cf2f00.json")
cred = credentials.Certificate("./brainstorm-2dcc5-firebase-adminsdk-7hskt-73e0cf2f00.json")
bucket_name="brainstorm-2dcc5.appspot.com"
if not firebase_admin._apps:
  firebase_admin.initialize_app(cred, {
      'storageBucket': bucket_name
  })
bucket = storage.bucket()
database = firestore.client()
print("firebase authorized")

def upload_audio_to_firebase(audio_data):
    #create audio_id
    audio_id=str(uuid.uuid4())
    #audio_data: torchTensor
    out_wav = (audio_data[0] * 32767).astype(np.int16).T
    buffer=io.BytesIO()
    # Save the audio_data to the file object
    sf.write(buffer, out_wav, 48000, format='WAV')  
    blob = bucket.blob('generated_from_video/{:s}'.format(audio_id))
    buffer.seek(0)  # Reset the file pointer to the beginning
    blob.upload_from_file(buffer, content_type='audio/wav')
    print("uploaded")
    return audio_id

audiosr = build_model(model_name="basic", device="auto")

#runpod handler
def handler(event):
    total_it=time.time()
    input_data=event['input']
    #handle input data -> input: text_list, duration
    path=input_data['path']
    ddim_step=input_data['ddim_step']

    outputs = super_resolution(
    audiosr,
    path,
    seed=42,
    guidance_scale=3.5,
    ddim_steps=ddim_step,
    latent_t_per_second=12.8
    )
    
    #upload audio in parallel to firebase
    audio_id = upload_audio_to_firebase(outputs)

    #return: list of audio ids
    response={
        'audio_id': audio_id,
    }
    return json.dumps(response)


#runpod config.
runpod.serverless.start({
    "handler": handler
})