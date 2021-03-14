#!/bin/bash

## download files sequentially
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4_partaa &&
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4_partab &&
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4_partac &&
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4_partad &&
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4_partae &&
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4_partaf &&
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4_partag &&
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4_partah &&
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4_partai &&
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test_mp4.zip 

## gather data
cat vox2_dev_aac* > vox2_aac.zip
unzip -d data vox2_aac.zip
rm vox2_dev_aac*
