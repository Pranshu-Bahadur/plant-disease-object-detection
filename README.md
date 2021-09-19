# Pytorch Image classification pipeline:

Dependencies:

```sh
pip install timm torch randaugment torchvision
```
example of commandline args:

Following will give you list of all commandline args:

```
python main.py --help 
```

Example Cli Command:

```sh
sudo python3 main.py -m resnet18 --resolution 224 --batch_size 8 --learning_rate 0.01 --num_classes 38 --epochs 10 --optimizer SGD --scheduler CosineAnnealing --loss CCE --library timm --train --dataset_directory /path/to/dataset/on/local --pretrained --save_directory path/on/local/to/save/checkpoints
```
