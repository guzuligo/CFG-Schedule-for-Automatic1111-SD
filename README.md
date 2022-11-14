# CFG-Schedule-for-Automatic1111-SD
A script for scheduling CFG scale and ETA to change during the denoising steps
<pre>
Modified code of https://github.com/tkalayci71/attenuate-cfg-scale
To use: Place the .py file in AUTOMATIC1111\scripts folder
Update: 11/14/2022
  -Now works with all samplers
  -new function x. Example use: x(t,60,70,2) which will map t to be 0 at 60 and 1 and 70, then multipy by 2
  -edit: Indecies can now have more than one index. Example: 0,5:=1 ; 1:10
Update: 11/13/2022

You can now use functions like: 
0:=math.sin(t) , 10:=math.cos(t)
or simply just:
=math.sin(t)

Available variables:
t: current step
T: total steps
e: (1-t/T) which is current ETA
d: denoising strength
pi:pi
cfg: initial CFG
math: The python math module

Available functions:
-From python: min, max, pow, abs
-x(value,min,max,multiply)
  This will take value and make it 0 when <=min, and make it 1 when >=max. Then, multiply result by last argumant.
  If multiply is less than 0, the range becomes 1 to 0 instead



I've tried the following function on 58 steps and it is giving good results:
CFG:
0:=math.fabs( (e*1.4-2)*15);30:5
ETA:
0:=0.01; 20:=e*1.5;10:=e
</pre>
![00031-336559885-car](https://user-images.githubusercontent.com/4189008/201653592-f719533e-573a-4a59-807a-085fb7e320d0.jpg)
