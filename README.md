# AI-powered Personal VoiceBot for Language Learning
This is voicebot powered by ChatGPT tuned for language learning. The user gives voice inputs just like in a casual chat. Text transcription is done via OpenAI's Whisper, which is later fed into ChatGPT. Response from GPT is converted into a speech by gTTS.

Detailed explanation of the repo is given on my [Medium article](https://medium.com/towards-data-science/ai-powered-personal-voicebot-for-language-learning-5ada0dfb1f9b).

For installing requirements, simply run:
```console
pip install -r requirements.txt
```

For starting the chat, run:
```console
sudo python chat.py
```