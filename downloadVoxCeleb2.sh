#!/bin/bash

## download files sequentially
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaa
sleep 60s
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partab
sleep 60s
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partac
sleep 60s
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partad
sleep 60s
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partae
sleep 60s
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaf
sleep 60s
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partag
sleep 60s
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partah
sleep 60s
wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test_aac.zip 

## gather data
cat vox2_dev_aac* > vox2_aac.zip
unzip -d data vox2_aac.zip
unzip -d test_data vox2_test_aac.zip 
rm -f vox2_aac.zip
rm -f vox2_test_aac.zip
rm -f vox2_dev_aac* 
