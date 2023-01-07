# CFG-Schedule-for-Automatic1111-SD
## CFG Auto.py
A simplified version. Contains tools to make the small denoising effective.

Following was set on denising: 0.1
Batch size is used to loop. Batch size:4
##
Input:![index](https://user-images.githubusercontent.com/4189008/211174581-c115bfde-970a-4a41-a9f7-306138e71462.jpg) output ![00150-2661151044-k-pop star  Red Necktie  Black suit and a white shirt under it](https://user-images.githubusercontent.com/4189008/211174606-1d540377-3e5e-48de-82a4-890a4b4287be.jpg)


## CFG Schedule.py
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
