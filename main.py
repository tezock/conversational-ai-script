import threading
import speech_recognition as sr
import time
import os
from gtts import gTTS
from playsound import playsound
import openai
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa
import base64
import requests
from dotenv import load_dotenv

# load environment variables from the .env file
load_dotenv()

# Set your OpenAI API key
api_key = os.getenv('OPENAI_GPT_KEY')
client = OpenAI(api_key=api_key)

# Shared variables to indicate if the user or robot has spoken
user_spoke = threading.Event()
robot_spoke = threading.Event()

# Shared variable to stop audio playback thread
stop_audio = threading.Event()

# lock to synchronize access to the 'mood' data
mood_lock = threading.Lock()
current_mood = ""

# stores the playing audio as global
playing_audio_obj = None

# Conversation history for context
conversation_history = [
    {"role": "system", "content": "You are a chatbot designed for interaction. "}
]

webcam_image_file = "webcam_image.jpg"
tts_speech_file = "response.mp3"
frames_to_ignore = 15

def listen():
    """Continuously listens for user input."""
    global conversation_history
    with sr.Microphone() as source:
        print("Listening...")

        # listen to the user
        while True:
            try:
                # Listen to the user

                # initialize a new recognizer every time, as
                # the recognizer dies after a few calls to listen() otherwise
                recognizer = sr.Recognizer()
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=None)
                
                user_input = recognizer.recognize_google(audio)

                with mood_lock:
                    conversation_history.append({"role": "system", "content": "user's current mood: " + current_mood})

                # stop the current audio playback if there is any
                kill_audio_thread()

                # Add user input to conversation history
                print(f"User: {user_input}")
                conversation_history.append({"role": "user", "content": user_input})

                # Generate a response from the chatbot
                conversation()

            # Handle cases where no speech was detected
            except sr.WaitTimeoutError:
                continue

            # Handle cases where the user's speech was unclear
            except sr.UnknownValueError:
                print("System: Sorry, I didn't catch that.")
                continue

            # Handle other errors
            except Exception as e:
                print(f"Listening error: {e}")
                continue

def kill_audio_thread():

    global playing_audio_obj

    print("killing audio...")

    if playing_audio_obj != None and playing_audio_obj.is_playing():
        playing_audio_obj.stop()
        print("Audio stopped!")

    if os.path.exists(tts_speech_file):
        os.remove(tts_speech_file)
        print("Audio file deleted.")



def conversation():
    """Handles conversation with the AI model."""
    global conversation_history

    # Generate completion from OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history
    )

    robot_response = completion.choices[0].message.content

    # Add robot response to conversation history
    conversation_history.append({"role": "assistant", "content": robot_response})
    print(f"Robot: {robot_response}")

    # Speak the response
    speak(robot_response)

def speak(text):
    """Convert text to speech and play audio using a thread."""
    global stop_audio
    stop_audio.clear()  # Reset the stop signal

    # Generate speech using gTTS
    tts = gTTS(text=text, lang='en')
    tts.save(tts_speech_file)

    # Start a new thread to play the audio
    audio_thread = threading.Thread(target=play_audio)
    audio_thread.start()

def play_audio():
    """Play audio in a separate thread, stopping if stop signal is received."""

    global playing_audio_obj

    try:

        audio = AudioSegment.from_file(tts_speech_file)

        # Play the audio in the foreground
        playing_audio_obj = sa.play_buffer(
            audio.raw_data,
            num_channels=audio.channels,
            bytes_per_sample=audio.sample_width,
            sample_rate=audio.frame_rate
        )

        time.sleep(len(audio) / 1000)

        if os.path.exists(tts_speech_file):
            os.remove(tts_speech_file)


    
    except Exception as e:
        print(f"Error playing sound: {e}")


def see_user():

    global api_key, mood_lock, current_mood

    import cv2
    import numpy as np

    while True:

        # Initialize the webcam (0 is the default ID for the main camera)
        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened successfully
        if not cap.isOpened():
            print("Error: Could not open the webcam.")
        else:

            ret = None
            frame = None
            # Capture the 15th frame (to discard the opening dark frames)
            for _ in range(frames_to_ignore):
                ret, frame = cap.read()

            # Convert the image to float32 for precision
            frame = np.float32(frame)

            # Increase the brightness by adding a constant value to all pixels
            brightness_value = 150  # Adjust this value as needed
            brightened_frame = frame + brightness_value

            # Ensure that pixel values remain within [0, 255]
            brightened_frame = np.clip(brightened_frame, 0, 255)

            # Convert back to uint8
            brightened_frame = np.uint8(brightened_frame)

            if ret:
                # Save the captured image to a file
                cv2.imwrite(webcam_image_file, frame)

                # Function to encode the image
                def encode_image(image_path):
                    with open(image_path, "rb") as image_file:
                        return base64.b64encode(image_file.read()).decode('utf-8')

                # Getting the base64 string
                base64_image = encode_image(webcam_image_file)

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }

                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe the person's mood in the image in ONE word that is simple and easy to understand, such as Happy, Sad, Mad, Etc..' "},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 300
                }

                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                
                with mood_lock:
                    current_mood = response.json()["choices"][0]["message"]["content"]
                    print(current_mood)

            else:
                print("Error: Failed to capture an image.")

        # Release the webcam
        cap.release()
        time.sleep(10)

# start listening and watching
listening_thread = threading.Thread(target=listen, daemon=True)
watching_thread = threading.Thread(target=see_user, daemon=True)

# start the watching and listening threads
watching_thread.start()
listening_thread.start()

# starts the program
try:
    while True:
        time.sleep(0.1)

# handles keyboard interrupt
except KeyboardInterrupt:

    # clean up files
    if os.path.exists(webcam_image_file):
        os.remove(webcam_image_file)

    if os.path.exists(tts_speech_file):
        os.remove(tts_speech_file)

    # print out details from the conversation
    print("---- Conversation ended ----")
    print("Printing out conversation")
    print()
    for row in conversation_history:
        
        print(row["role"].title() + ": ", row["content"])
