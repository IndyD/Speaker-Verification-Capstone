# Speaker-Verification-Capstone
Text-independent, one-shot speaker verification. Capstone project for SMU MSDS graduate program.

This project uses the VoxCeleb dataset, which is a open dataset of over 7000 speakers and 1 million utterances (https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) to create a speaker verification system that can be trained on this corpus and find the distances betwreen 2 arbitary voices.

Three different neural network architechutes are tested: one using contrastive pairwise loss, one using triplet loss, and one using quaruplet loss. 

To run the project: 
1. Download the data (https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
2. Specify the directory (relative to the git repo) in the config.json file in th PATHS.AUDIO_DIR field
3. Convert the audio wav using the m4a_to_wav.py as follows: python m4a_to_wav.py <AUDIO_DIR>
4. Run the experiment: python run_experiment.py config.json

This outputs the accuracy and EER along with the trained model in the specified output directory