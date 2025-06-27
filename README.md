**SPEED: A GUI Software for Processing Pupil Labs Neon Eye Tracking Data**

SPEED (LabSCoC Processing and Extraction of Eye-tracking Data) is a Python-based GUI application designed to streamline the processing and feature extraction of data acquired with Pupil Labs Neon eye trackers. Developed by the members and collaborators of the LabSCoC - Laboratorio di Scienze Cognitive e del Comportamento (Cognitive and Behavioural Science Lab) at the University of L'Aquila, SPEED simplifies the initial steps of eye-tracking data analysis.

*Authors*
Daniele Lozzi,
Ilaria Di Pompeo,
Martina Marcaccio,
Matias Ademaj,
Simone Migliore,
Giuseppe Curcio

*Getting Started*
To use SPEED, follow these steps to prepare your data:

1. Data Download from Pupil Cloud
You'll need to download three sets of data from Pupil Cloud:

Download data after QR code (using "Marker Mapper" enrichment function): This will provide enriched gaze data based on the QR code calibration.

Download time series + video: This includes various event files and the external camera video.

Download native record data: This is where you'll find the internal eye camera video.

2. Prepare Your eyetracking_file Folder
Create a new folder named eyetracking_file in the same directory as the SPEED_0_1_gui.py script. All your downloaded data files will go into this folder.

3. Organize Your Files
After downloading, move and rename the necessary files into your eyetracking_file folder as follows:

From "Download data after QR code":

-fixations.csv

-gaze.csv

From "Download time series + video":

-3d_eye_states.csv

-blink.csv

-events.csv

-saccades.csv

-The external camera video (usually named with a series of digits and characthers .mp4 or similar) should be renamed to external.mp4.

-Another gaze.csv file from this download should be renamed to gaze_not_enr.csv.

From "Download native record data":

-The internal eye camera video (usually named eye_0.mp4 or eye_1.mp4) should be renamed to internal.mp4.

Folder Content Example
Your eyetracking_file folder should look something like this:

eyetracking_file/

├── 3d_eye_states.csv

├── blink.csv

├── events.csv

├── external.mp4

├── fixations.csv

├── gaze.csv

├── gaze_not_enr.csv

├── internal.mp4

└── saccades.csv


4. Run SPEED
Ensure you have Python 3 installed. Once all your files are correctly placed and named within the eyetracking_file folder, simply run the main script:

python SPEED_0_1_gui.py

SPEED will then process your data.
