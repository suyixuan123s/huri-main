Powershell
1. Set the python environemnt variable (Should at the root directory)
```shell
$Env:PYTHONPATH=$(pwd)
```

2. Execute training program
```shell
python .\huri\learning\method\DQN\main.py --step-num 100000000 --reset-num 100 --start-step 500 --eval-interval 70 --batch-size 32 --lr 0.0003 --gamma 0.95 --update-freq 2000
```