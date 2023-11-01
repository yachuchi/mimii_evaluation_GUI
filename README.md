# mimii_evaluation_GUI
Evaluate whether a wave file is normal or abnormal 

## File description
```
evaluateGUI.py: main code
evaluate.yaml : parameter setting
```
## Create folder
```
dataset: for test file(type: .wav)
model  : for trained model(type: .pth)
pickle_unsupervised : for unsupervised model test(type: .pickle) 
```
## Usage
Environment : mobaxtern
```
 python3 evaluateGUI.py
```
Result:
![image](https://github.com/yachuchi/mimii_evaluation_GUI/blob/master/demo%20image.png)
More Detail
```
Button description
Open wave file : choose test wavefile
Open model file: choosed trained model
start testing  : test wavefile you choose by trained model you choose
reset          : clean the result
Visualization  : visualize the wavefile and the test error
```
