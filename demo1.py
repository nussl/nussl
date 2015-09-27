import AudioSignal, REPET_sim

def main():
	myFile = 'Input/mix.wav'
	myAudioFile = AudioSignal(myFile)
	repet = RepetSim(audioSignal=myAudioFile)
	

if __name__ == '__main__':
	main()