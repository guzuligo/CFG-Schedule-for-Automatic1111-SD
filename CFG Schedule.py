from logging import PlaceHolder
import math
import os
import sys
import traceback

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
        placeholder="You can also use functions like: 0:=math.fabs(-t) ; 1:=(1-t/T) ; 2:=e ;3:=t*d"
        n1 = gr.Textbox(label="ETA",placeholder=placeholder)
        #n2 =gr.Textbox(label="lalala",placeholder="")
        return [n0,n1]

    def run(self, p, cfg,eta):
        #print("it is:",n0t)
        #for x in range(int(n)):
        self.p=p
        cfg=cfg.strip()
        eta=eta.strip()
        #n2=n2.strip()
        if cfg:
            p.cfg_scale=Fake_float(p.cfg_scale,self.split(cfg))
        #for i in p.__dict__:
        #    print(i)
        #    #print(p[i])
        if eta:
            p.eta=Fake_float(p.eta or 1,self.split(eta)) 
        
        proc = process_images(p)
        return proc #Processed(p, image, p.seed, proc.info)
    def split(self,src,default='0'):
        p=self.p
        self.P={
            'cfg':p.cfg_scale,
            'd':p.denoising_strength,

            'min':min,
            'max':max,
            'abs':abs,
            'pow':pow,
            'pi':math.pi,

        }

        if src[0:4]=="eval":
            return self.evaluate(src[4:])
        if src[0]=="=":
            return self.evaluate(src[1:])

        
        
        arr = src.split(';')##2
        s=[]
        val=default
        for j in range(p.steps):
          i=0
          while i<len(arr):
              v=arr[i].split(":")
              #s=proc[j].n_iter
              if int(v[0])>=j:
               
                 break
              i=i+1
              val=v[1].strip()
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
        #print(s)
        return s

    def evaluate (self,src):
        s=[]
        p=self.p
        T=self.p.steps
        for j in range(T):
            _eta=1-j/p.steps
            params={'t':j,'T':p.steps,'math':math,'p':p,'e':_eta}
            params.update(self.P)
            s.append(float(eval(src,params)))
        return s

class Fake_float(float):
    def __new__(self, value, arr):
        return float.__new__(self, value)

    def __init__(self, value, arr):
        float.__init__(value)
        self.arr = arr
        self.curstep = 0
        #self.p=p

    def __mul__(self,other):
        return self.fake_mul(other)

    def __rmul__(self,other):
        return self.fake_mul(other)

    def fake_mul(self,other):
        #print("\n",self.p.n_iter,"\n")
        fake_value = self.arr[self.curstep]
        #print(self.curstep,fake_value)
        self.curstep += 1
        return fake_value * other