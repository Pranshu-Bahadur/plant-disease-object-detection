# Plant disease Object-detection

libraries installation:
- pytorch 1.8

pip3 install timm
pip3 install randaugment

example of commandline args:

Following will give you list of all commandline args
python main.py --help 

sudo python3 main.py -m resnet18 --resolution 224 --batch_size 8 --learning_rate 0.01 --num_classes 38 --epochs 10 --optimizer SGD --scheduler CosineAnnealing --loss CCE --library timm --train --dataset_directory /path/to/dataset/on/local --pretrained --save_directory path/on/local/to/save/checkpoints