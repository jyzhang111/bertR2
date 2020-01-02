TO RUN:

Make sure you install the pytorch bert library:

pip install pytorch-pretrained-bert

Then, move all the files in the bertR2 folder over to the sphero-project/src directory, and run the R2D2 setup instructions. If you decide to use audio, you need to run:

export GOOGLE_APPLICATION_CREDENTIALS="/[Path to sphero-project/src]/credentials.json"

before running the audio_io.py script. You can change the R2D2 id in robot_com and audio_io, but with the connection parsing functionality, you shouldn't need to :).