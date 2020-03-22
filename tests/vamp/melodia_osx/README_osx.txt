MELODIA - Melody Extraction Vamp plug-in 1.0 (OSX universal 32/64-bit)
======================================================================

Created By
----------
Justin Salamon
Music Technology Group
Universitat Pompeu Fabra
Barcelona, Spain
http://www.justinsalamon.com
http://mtg.upf.edu


Description
-----------
The MELODIA plug-in automatically estimates the pitch of a song's main melody. More specifically, it implements an algorithm that automatically estimates the fundamental frequency corresponding to the pitch of the predominant melodic line of a piece of polyphonic (or homophonic or monophonic) music. 

Given a song, the algorithm estimates:
1) When the melody is present and when it is not (a.k.a. voicing detection)
2) The pitch of the melody when it is present

Full details of the algorithm can be found in the following paper:

J. Salamon and E. Gomez, "Melody Extraction from Polyphonic Music Signals using Pitch Contour Characteristics", IEEE Transactions on Audio, Speech and Language Processing, 20(6):1759-1770, Aug. 2012.

We would highly appreciate if scientific publications of works partly based on the MELODIA plug-in cite the above publication.

A non-scientist friendly introduction to Melody Extraction as well as the algorithm, including graphs and sound examples, can be found on: www.justinsalamon.com/melody-extraction

For computational reasons, MELODIA is composed of two vamp plug-ins: "MELODIA - Melody Extraction" and "MELODIA - Melody Extraction (intermediate steps)". The former provides the main output of MELODIA (the pitch of the predominant melody), whilst the latter provides visualisations of the intermediate steps calculated by the algorithm (see Input/Output below for further details). Both plug-ins are included in a single MELODIA library file.


Conditions of Use
-----------------
The MELODIA 1.0 Vamp plug-in is offered free of charge for non-commercial use only. MELODIA is not free software, meaning you can not redistribute it nor modify it. If you are interested in using MELODIA for commercial purposes please contact: mtg-techs@llista.upf.edu
Plug-in by Justin Salamon. Copyright © 2012 Music Technology Group, Universitat Pompeu Fabra. All Rights Reserved.


Please Acknowledge MELODIA in Academic Research
-----------------------------------------------
When MELODIA is used for academic research, we would highly appreciate if scientific publications of works partly based on the MELODIA plug-in cite the following publication:

J. Salamon and E. Gomez, "Melody Extraction from Polyphonic Music Signals using Pitch Contour Characteristics", IEEE Transactions on Audio, Speech and Language Processing, 20(6):1759-1770, Aug. 2012.


Input/Output
------------
Input: audio file in a format supported by your Vamp host (e.g. wav, mp3, ogg). 
The supported sampling rates are: 48kHz, 44.1kHz, 32kHz, 24kHz, 22050Hz, 16kHz, 12kHz, 11025Hz, 8kHz. 
The recommended sampling rate is 44.1kHz.

Output: MELODIA offers 4 different types of output. The first (Melody) is computed by the "MELODIA - Melody Extraction" plug-in and the rest by the "MELODIA - Melody Extraction (intermediate steps)" plug-in:

- Melody 
The pitch of the main melody. Each row of the output contains a timestamp and the corresponding frequency of the melody in Hertz (please see IMPORTANT below regarding timestamps). Non-voiced segments are indicated by zero or negative frequency values. Negative values represent the algorithm's pitch estimate for segments estimated as non-voiced, in case the melody is in fact present there. 

- Salience Function
A 2D time-frequency representation of pitch salience over time on a cent scale. The salience function covers five octaves, from 55Hz to 1760Hz, divided into 600 bins with a resolution of 10 cents per bin (= 120 bins per octave).

- Pitch Contours: All
In order to estimate the melody, the algorithm first tracks all salient pitch contours present in the signal. This output is a 2D representation of pitch contours vs. time, using the same scale as the salience function.

- Pitch Contours: Melody
This output is the same as output "Pitch Contours: All", except that only contours which were identified by the algorithm as part of the melody are displayed. By comparing "Pitch Contours: All" and "Pitch Contours: Melody" you can observe how the algorithm filters out non-melody pitch contours.

IMPORTANT:
As explained above, the Melody output includes a timestamp column. The timestamp of each analysis frame is provided to the plug-in by the Vamp host. In the case of both Sonic Visualiser and Sonic Annotator, the timestamp provided corresponds to the time at the *beginning* of the analysis frame (i.e. the time of the first sample of the frame). Since in melody extraction it is customary to report the time at the *middle* of the analysis frame, the MELODIA plug-in adjusts the timestamps passed to it by the host by adding to each timestamp the duration of half the analysis frame. Always make sure you know what timestamps are provided by your Vamp host. Melody extraction evaluation is highly sensitive to annotation offsets - for a MIREX style evaluation you must ensure that your annotations use the same timestamp convention as MELODIA (or make the relevant adjustments). 


Parameters
----------
- Min Frequency: the minimum frequency allowed for the melody
- Max Frequency: the maximum frequency allowed for the melody
- Voicing Tolerance: determine the tolerance of the voicing filter. Higher values mean more tolerance (i.e. more pitch contours will be included in the melody even if they are less salient), lower values mean less tolerance (i.e. only the most salient pitch contours will be included in the melody).
- Monophonic Noise Filter: for monophonic recordings only (i.e. solo melody with no accompaniment). Increase this value to filter out background noise (e.g. noise introduced by a laptop microphone). Always set to 0 for polyphonic recordings.

IMPORTANT: 
(a) the default parameter values (55, 1760, 0.2 and 0.0 respectively) are the ones used by the algorithm in the MIREX 2011 evaluation campaign and in the IEEE TASLP paper, and should be left unchanged if you wish to reproduce the performance of the algorithm as reported in MIREX and in the paper.
(b) The advanced parameters in Sonic Visualiser ("Audio frames per block" and "Window increment") are computed automatically as a function of the file's sampling rate and should not be changed.


System Requirements
--------------------
- Intel Mac running OSX 10.5 (Leopard) or newer
- A Vamp host such as Sonic Visualiser (www.sonicvisualiser.org), Sonic Annotator (www.omras2.org/SonicAnnotator) or Audacity (audacity.sourceforge.net)


Installation
------------
Copy all files in "MTG-MELODIA 1.0 (OSX universal).zip" to:
/Library/Audio/Plug-Ins/Vamp

NOTE: you can copy the files to any other folder of your choice as long as you add it to your vamp path. General instructions for installing vamp plug-ins can be found here: http://vamp-plugins.org/download.html#install


How to use MELODIA
------------------
For computational reasons, MELODIA is composed of two vamp plug-ins: "MELODIA - Melody Extraction" and "MELODIA - Melody Extraction (intermediate steps)". The former provides the Melody output (see Input/Output), whilst the latter provides the other three - Salience Function, Pitch Contours: All and Pitch Contours: Melody.

In Sonic Visualiser:
1. Load an audio file
2. From the menu select: 
Transform -> Analysis by Plugin Name -> MELODIA - Melody Extraction...
Or
Transform -> Analysis by Plugin Name -> MELODIA - Melody Extraction (intermediate steps) -> desired_output
3. If needed, adjust the parameters values (not recommended unless you know what you are doing)
4. Click OK

With Sonic Annotator:
The plug-in identifier for "MELODIA - Melody Extraction" is "melodia", and the identifier for "MELODIA - Melody Extraction (intermediate steps)" is "melodiaviz".

1. Open the command prompt (Start -> Run -> cmd)
2. Make sure the MELODIA plug-in is recognised by Sonic Annotator (requires adding the MELODIA library location to your vamp path or launching the annotator from the directory containing the .dylib file). You can check the list of recognised plug-ins by running:
$ ./sonic-annotator -l
3. For a single file using the default parameters, run:
$ ./sonic-annotator -d vamp:mtg-melodia:melodia:melody AUDIOFILE_PATH -w csv
Or
$ ./sonic-annotator -d vamp:mtg-melodia:melodiaviz:OUTPUT_TYPE AUDIOFILE_PATH -w csv
where OUTPUT_TYPE can be "saliencefunction", "contoursall" or "contoursmelody".
Examples:
$ ./sonic-annotator -d vamp:mtg-melodia:melodia:melody ~/audio/song.wav -w csv
$ ./sonic-annotator -d vamp:mtg-melodia:melodiaviz:saliencefunction ~/audio/song.wav -w csv

For processing all files in a directory (recursively) using the default parameters:
$ ./sonic-annotator -d vamp:mtg-melodia:melodia:melody -r ~/audio -w csv


Known Limitations
-----------------
- MELODIA uses a fair amount of RAM. Basically, the longer the song you analyse, the more RAM you will need.
- In the vamp architecture, all outputs are computed every time a plug-in is run (even though only one output is returned). For this reason, the Melody output is computed in a separate plug-in (MELODIA - Melody Extraction), allowing the analysis of long songs without running into memory issues. The outputs for visualising intermediate steps (Salience Function, Pitch Contours: All and Pitch Contours: Melody) are computed by the same plug-in (MELODIA - Melody Extraction (intermediate steps)), meaning memory may become an issue for long songs, depending on the system on which the plug-in is run.


Bug Reports and Feedback
------------------------
Problems, crashes, bugs, positive feedback, negative feedback... it is all welcome! Please help me improve MELODIA by sending your feedback to:
mtg-techs@llista.upf.edu

In case of a bug report please include as many details (operating system, plug-in version, file analysed, etc.) as possible.