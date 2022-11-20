#CFG Scheduler for Automatic1111 Stable Diffusion web-ui
#Author: https://github.com/guzuligo/
#Based on: https://github.com/tkalayci71/attenuate-cfg-scale
#Version: 1.4

from logging import PlaceHolder
import math
import os
import sys
import traceback
import numpy as np
import modules.scripts as scripts
import gradio as gr

#from modules.processing import Processed, process_images
from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
class Script(scripts.Script):
    def title(self):
        return "CFG Scheduling"

    def ui(self, is_img2img):
        placeholder="The steps on which to modify, in format step:value - example: 0:10 ; 10:15"
        n0 = gr.Textbox(label="CFG",placeholder=placeholder)
        placeholder="You can also use functions like: 0:=math.fabs(-t) ; 1:=(1-t/T) ; 2:=e ;3:t*d"
        n1 = gr.Textbox(label="ETA",placeholder=placeholder)
        #n2 =gr.Textbox(label="lalala",placeholder="")
        return [n0,n1]

    def run(self, p, cfg,eta):

        #if p.sampler_index in (0,1,2,7,8,10,14):
        if p.sampler_name in ('Euler a','Euler','LMS','DPM++ 2M','DPM fast','LMS Karras','DPM++ 2M Karras'):
            max_mul_count = p.steps * p.batch_size
            steps_per_mul = p.batch_size
        #elif p.sampler_index in (3,4,5,6,11,12,13):
        elif p.sampler_name in ('Heun','DPM2','DPM2 a','DPM++ 2S a','DPM2 Karras','DPM2 a Karras','DPM++ 2S a Karras'):
            max_mul_count = ((p.steps*2)-1) * p.batch_size
            steps_per_mul = 2 * p.batch_size
        #elif p.sampler_index==15: # ddim
        elif p.sampler_name=='DDIM': # ddim
            max_mul_count = fix_ddim_step_count(p.steps)
            steps_per_mul = 1
        #elif p.sampler_index==16: # plms
        elif p.sampler_name=='PLMS': # plms
            max_mul_count = fix_ddim_step_count(p.steps)+1
            steps_per_mul = 1
        else:
            return # 9=dpm adaptive


        #print("it is:",n0t)
        #for x in range(int(n)):
        self.p=p
        cfg=cfg.strip()
        eta=eta.strip()
        if cfg:
            p.cfg_scale=Fake_float(p.cfg_scale,self.split(cfg,str(p.cfg_scale))  , max_mul_count, steps_per_mul)
            #p.cfg_scale.p=p
        if eta:
            p.eta=Fake_float(p.eta or 1,self.split(eta,str(p.eta)), max_mul_count, steps_per_mul)
            #p.cfg_scale.p=p
        
        proc = process_images(p)
        return proc #Processed(p, image, p.seed, proc.info)


    def split(self,src,default='0'):
        p=self.p
        self.P={
            'cfg':p.cfg_scale,
            'd':p.denoising_strength or 1,

            'min':min,
            'max':max,
            'abs':abs,
            'pow':pow,
            'pi':math.pi,
            'x':self._interpolate
        }

        if src[0:4]=="eval":
            return self.evaluate(src[4:])
        if src[0]=="=":
            return self.evaluate(src[1:])

        
        
        arr0 = src.split(';')##2

        #resort array accounting for commas in indecies
        arr=[]
        for j in arr0:
            #print(j)
            v=j.split(":")
            q=v[0].split(",")
                
            for i in q:
                arr.append(i+":"+v[1])

                


        arr.sort(key=self._sort)
        s=[]
        val=default
        for j in range(p.steps+1):
          i=0
          while i<len(arr) and i<=j:
              v=arr[i].split(":")
              #s=proc[j].n_iter
              if int(v[0])==j:
                 val=v[1].strip()
                 break
              i=i+1
              
          #lets just evaluate all        
          if val[0]=="=":
            val=val[1:]
          
          _eta=1-j/p.steps
          params={'t':j,'T':p.steps,'math':math,'p':p,'e':_eta}
          params.update(self.P)
          s.append(float(eval(val,params)))
          #end while loop
          #else:    
            #s.append(float(val))
        print(np.round(s,1),"\n")
        return s
    #limits a range of a value
    def _interpolate(self,v,start=0,end=None,m=1):
        end=end or self.p.steps
        v=min(max(v,start),end)-start
        return v*m/(end-start)+(1 if m<0 else 0)

    def _sort(self,a):
        return int(a.split(":")[0])

    def evaluate (self,src):
        s=[]
        p=self.p
        T=self.p.steps
        for j in range(T+1):
            _eta=1-j/p.steps
            params={'t':j,'T':p.steps,'math':math,'p':p,'e':_eta}
            params.update(self.P)
            s.append(float(eval(src,params)))
        return s

class Fake_float(float):
    def __new__(self, value, arr, max_mul_count, steps_per_mul):
        return float.__new__(self, value)

    def __init__(self, value, arr, max_mul_count, steps_per_mul):
        float.__init__(value)
        self.arr = arr
        self.curstep = 0
        #self.p=p

        #self.orig_value = orig_value
        #self.target_value = target_value
        self.max_mul_count = max_mul_count
        self.current_mul = 0
        self.steps_per_mul = steps_per_mul
        self.current_step = 0 #fake
        self.max_step_count = (max_mul_count // steps_per_mul) + (max_mul_count % steps_per_mul > 0)

    def __mul__(self,other):
        return self.fake_mul(other)

    def __rmul__(self,other):
        return self.fake_mul(other)

    def fake_mul(self,other):
        #print("steps",self.p.steps)
        #print("\n",self.p.n_iter,"\n")
        if (self.max_step_count==1):
            fake_value = self.arr[0]#self.curstep]
        else:
            #fake_value = self.arr[self.curstep+ (self.target_value - self.orig_value)*(self.current_step/(self.max_step_count-1))]
            fake_value = self.arr[self.curstep]
        #print(self.curstep,fake_value)
        #self.curstep += 1
        
        
        #print("---\nstep:",self.curstep,"\nmax steps:",self.max_step_count,"\nfake step:",self.current_step,"\n--")
        self.current_mul = (self.current_mul+1) % self.max_mul_count
        self.curstep = (self.current_mul) // self.steps_per_mul
        self.current_step+=1#FAKE STEP
        return fake_value * other


def fix_ddim_step_count(steps):
    valid_step = 999 / (1000 // steps)
    if valid_step == int(valid_step): steps=int(valid_step)+1
    if ((1000 % steps)!=0): steps +=1
    return steps