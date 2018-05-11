# datmo iris example

Flow: 
1. Run datmo task run or python train.py

2. Run datmo snapshot create -m "model created with perceptron"

3. Run `datmo snapshot checkout --id <snapshot-id>`

4. Run datmo task run or python train.py
