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
from modules import images,processing
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
class Script(scripts.Script):
    def show(self, is_img2img):
        return True
    def title(self):
        return "CFG Scheduling"

    def ui(self, is_img2img):
        placeholder="The steps on which to modify, in format step:value - example: 0:10 ; 10:15"
        n0 = gr.Textbox(label="CFG",placeholder=placeholder)
        placeholder="You can also use functions like: 0:=math.fabs(-t) ; 1:=(1-t/T) ; 2:=e ;3:t*d"
        n1 = gr.Textbox(label="ETA",placeholder=placeholder)
        #loops
        n2 = gr.Slider(minimum=1, maximum=32, step=1, label='Loops', value=1)
        n3 = gr.Slider(minimum=0.5, maximum=1.5, step=0.01, label='Denoising strength change factor', value=1)
        return [n0,n1,n2,n3]

    def prepare(self,p,cfg,eta):
        sampler_name=p.sampler_name
        if not sampler_name:
            print("Warning: sampler not specified. Using Euler a")
            sampler_name="Euler a"
        #if p.sampler_index in (0,1,2,7,8,10,14):
        if sampler_name in ('Euler a','Euler','LMS','DPM++ 2M','DPM fast','LMS Karras','DPM++ 2M Karras'):
            max_mul_count = p.steps * p.batch_size
            steps_per_mul = p.batch_size
        #elif p.sampler_index in (3,4,5,6,11,12,13):
        elif sampler_name in ('Heun','DPM2','DPM2 a','DPM++ 2S a','DPM2 Karras','DPM2 a Karras','DPM++ 2S a Karras'):
            max_mul_count = ((p.steps*2)-1) * p.batch_size
            steps_per_mul = 2 * p.batch_size
        #elif p.sampler_index==15: # ddim
        elif sampler_name=='DDIM': # ddim
            max_mul_count = fix_ddim_step_count(p.steps)
            steps_per_mul = 1
        #elif p.sampler_index==16: # plms
        elif sampler_name=='PLMS': # plms
            max_mul_count = fix_ddim_step_count(p.steps)+1
            steps_per_mul = 1
        else:
            print("Not supported sampler", p.sampler_name, p.sampler_index)
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
            if (eta.find("@")==-1):
                p.s_churn=p.eta    =Fake_float(p.eta     or 1,self.split(eta,str(p.eta))    , max_mul_count, steps_per_mul)
                #print(p.s_noise)
            
                #Fake_float(p.s_churn or 1,self.split(eta,str(p.s_churn)), max_mul_count, steps_per_mul)
            else:
                eta=eta.split("@")
                if eta[0].strip()!="":
                    p.s_churn=Fake_float(p.s_churn or 1,self.split(eta[0],str(p.s_churn)), max_mul_count, steps_per_mul)
                if len(eta)>1 and eta[1].strip()!="":
                    p.s_noise=Fake_float(p.s_noise or 1,self.split(eta[1],str(p.s_noise)), max_mul_count, steps_per_mul)
                if len(eta)>2 and eta[2].strip()!="":
                    p.s_tmin=Fake_float(p.s_tmin or 1,self.split(eta[2],str(p.s_tmin)), max_mul_count, steps_per_mul)
                if len(eta)>3 and eta[3].strip()!="":
                    p.s_tmax=Fake_float(p.s_tmax or 1,self.split(eta[2],str(p.s_tmax)), max_mul_count, steps_per_mul)


            #p.cfg_scale.p=p
        #




    def run(self, p, cfg,eta,loops,dns):
        p.denoising_strength=p.denoising_strength or 1
        if loops>1:
            processing.fix_seed(p)
            #self.initDenoise=p.denoising_strength
            p.extra_generation_params = {
                "Denoising strength change factor": str(dns)+"\n",
                "CFG Scheduler Info":"\nCFG: "+cfg+"\nETA: "+eta+"\nloops:"+str(loops),
            }  
        history=[]
        for loop in range(loops):
            self.loop=loop
            self.prepare(p, cfg,eta)
            proc = process_images(p)
            if loop==0:
                self.initInfo=proc.info
                self.initSeed=proc.seed
            history.append(proc.images[0])
            p.seed+=1
            p.init_images=[proc.images[0]]
            p.denoising_strength=min(max(p.denoising_strength * dns, 0.1), 1)
            #print("New denoising:"+str(p.denoising_strength)+"\n" )
        if loops>0:
            p.seed=self.initSeed
        return proc if (loops==1) else Processed(p, history, self.initSeed, self.initInfo)








    def split(self,src,default='0'):
        p=self.p
        self.P={
            'cfg':p.cfg_scale,
            'd':p.denoising_strength or 1,
            'l':self.loop,
            'min':min,
            'max':max,
            'abs':abs,
            'pow':pow,
            'pi':math.pi,
            'x':self._interpolate,
            'int':int,
            'floor':math.floor,
            
        }

        if src[0:4]=="eval":
            src="0:"+src[4:]
        if src[0]=="=":
            src="0:"+src[1:]

        
        
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

    def __add__(self,other):
        #print("ADD!")
        return self.get_fake_value(other)+other
    def __sub__(self,other):
        #print("SUB!")
        return self.get_fake_value(other)-other



    def fake_mul(self,other):
        #print("MUL!")
        return self.get_fake_value(other) * other


    def get_fake_value(self,other):
        if (self.max_step_count==1):
            fake_value = self.arr[0]
        else:
           
            fake_value = self.arr[self.curstep]
        self.current_mul = (self.current_mul+1) % self.max_mul_count
        self.curstep = (self.current_mul) // self.steps_per_mul
        self.current_step+=1#FAKE STEP
        return fake_value

    


def fix_ddim_step_count(steps):
    valid_step = 999 / (1000 // steps)
    if valid_step == int(valid_step): steps=int(valid_step)+1
    if ((1000 % steps)!=0): steps +=1
    return steps
