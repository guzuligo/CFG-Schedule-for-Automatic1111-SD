# CFG-Schedule-for-Automatic1111-SD
A script for scheduling CFG scale and ETA to change during the denoising steps
<pre>
Modified code of https://github.com/tkalayci71/attenuate-cfg-scale
To use: Place the .py file in AUTOMATIC1111\scripts folder

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
cfg: initial CFG
math: The python math module


I've tried the following function on 58 steps and it is giving good results:
CFG:
0:=math.fabs( (e*1.4-2)*15),30:5
ETA:
0:=0.01, 20:=e*1.5,10:=e
