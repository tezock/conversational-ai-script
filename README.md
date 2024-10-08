# Realistic Conversational AI

## What is it?

This is a conversational AI script created to handle both sentiment and gesture
analysis, mostly due to my interest in making AI chatbots more human-like
by taking advantage of multithreading.

The main issue with most voice-to-text and text-to-speech methods of communicating with
computers, is largely that whenever they start talking... you can't stop them.

Thus, I sought to implement a more realistic conversation by utilizing multithreading!


## How does it work?:


I utilized multithreading such that users are able to interrupt the chatbot
when it's speaking, and not be delayed by the long i/o of a process trying
to complete whatever nonsense it may be responding back with.

The main loop begins with a user speaking a complete phrase, and when recognized
by the machine, creates a new thread to begin playing the audio. Then, when the
user tries to say a new phrase, if the audio is still playing (or the computer
is still talking), then it is pre-empted, allowing for a more realistic flow of
dialogue similar to real life!

Additionally, given how modular (though computationally heavy) threading makes
such a system, I added gesture/emotion analysis (which can be altered again
through prompting), which takes a picture of the user to determine what their
mood is. This long-running i/o, when combined with multithreading, makes a great
way to enhance a chatbot in a practical setting. Additional threads could also
be created to support tone analysis as well!

### Built With

* [Python](https://www.python.org/)

## Setup/Usage:
- Add a OPENAI_GPT_KEY field with an OpenAI GPT Key to your .env
- Install any necessary dependencies not yet installed on your machine
- Run!
