#CFG Scheduler for Automatic1111 Stable Diffusion web-ui
#Author: https://github.com/guzuligo/
#Based on: https://github.com/tkalayci71/attenuate-cfg-scale
#Version: 1.81

from logging import PlaceHolder
import math
import os
import sys
import traceback
import copy
import numpy as np
import modules.scripts as scripts
import gradio as gr

#from modules.processing import Processed, process_images
from modules import images,processing
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
class Script(scripts.Script):
    def run(self,p,n0,dns,ns1,ns2,nr1,nr2   ,loops,nSingle):
        return self.runBasic(p,n0,dns,ns1,ns2,nr1,nr2   ,loops,nSingle)

    #def run(self,p,cfg,eta,dns ,loops,nSingle):
    #    return self.runAdvanced(p,cfg,eta,dns ,loops,nSingle)

    def show(self, is_img2img):
        self.isAdvanced=False
        return True
    def title(self):
        return "CFG Scheduling" if (self.isAdvanced) else "CFG Auto"

    def uiAdvanced(self, is_img2img):

        placeholder="The steps on which to modify, in format step:value - example: 0:10 ; 10:15"
        n0 = gr.Textbox(label="CFG",placeholder=placeholder)
        placeholder="You can also use functions like: 0: math.fabs(-t) ; 1: (1-t/T) ; 2:=e ;3:t*d"
        n1 = gr.Textbox(label="ETA",placeholder=placeholder)
        #loops
        #n2 = gr.Slider(minimum=1, maximum=32, step=1, label='Loops', value=1)
        n2 = gr.Slider(minimum=0, maximum=1, step=0.01, label='Target Denoising : Decay per Batch', value=0.5)
        with gr.Row():
            loops=gr.Number(value=1,precision=0,label="loops")
            nSingle= gr.Checkbox(label="Loop returns one")
            
        return [n0,n1,n2    ,loops,nSingle]
    #uiBasic
    def uiAuto(self, is_img2img):
        self.autoOptions={"b1":"Blur First V1","b2":"Blur Last","f1":"Force at Start V1","f2":"Force Allover"}
        with gr.Row():
            dns = gr.Slider(minimum=0, maximum=1, step=0.01, label='Target Denoising : Decay per Batch', value=0.25)
            n0=gr.Dropdown(list(self.autoOptions.values()),value=self.autoOptions["b1"],label="Scheduler")
        with gr.Row():
            n1 = gr.Slider(minimum=0, maximum=100, step=1, label='Main Strength', value=10)
            n2 = gr.Slider(minimum=0, maximum=100, step=1, label='Sub- Strength', value=10)
        with gr.Row():
            n3 = gr.Slider(minimum=0, maximum=100, step=1, label='Main Range', value=10)
            n4 = gr.Slider(minimum=0, maximum=100, step=1, label='Sub- Range', value=10)
        with gr.Row():
            loops=gr.Number(value=1,precision=0,label="loops")
            nSingle= gr.Checkbox(label="Loop returns one")
        return [n0,dns,   n1,n2,n3,n4   ,loops,nSingle]

    def ui(self, is_img2img):
        return self.uiAdvanced(is_img2img) if (self.isAdvanced) else self.uiAuto(is_img2img)

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



    def runBasic(self,p,n0,dns,ns1,ns2,nr1,nr2  ,loops,nSingle):
        if(n0==self.autoOptions["b1"]):
            cfg=f"""0:{ns2}/2 if (t<T* (({nr1}/100)**2)) else cfg"""
            eta=f"""0:{ns1}+1 if (t<T*(({nr1}/100)**2)  ) else e*({nr2}/50)"""
        elif(n0==self.autoOptions["f1"]):
            cfg=f"""0:({ns1}*4)*((1-d**0.5)**1.5)/(t*(30-cfg)/30+1)/(l*2+1) 	if (t<T*{nr1}/100) else 0.1 if (t<T*({nr1}+{nr2}-{nr1}*{nr2})/100) else 7-d*7"""
            eta=f"""0:0.8+{ns2}/25-min(t*0.1, 0.8+{ns2}/25 -0.01)			if (t<T*{nr1}/100) else {ns2}/(10*(1+l*0.5)) if (t<T*({nr1}+{nr2}-{nr1}*{nr2})/100) else 1+e"""
        elif(n0==self.autoOptions["b2"]):
            cfg=f"""0:cfg if (e>{nr1}/100 or e<(1-({nr1}+{nr2}*(100-{nr1})/100)/100)) else {ns2}/10"""
            eta=f"""0:e   if (e>{nr1}/100 or e<(1-({nr1}+{nr2}*(100-{nr1})/100)/100)) else {ns1}/10"""
        elif(n0==self.autoOptions["f2"]):
            cfg=f"""= min(40,max(0,cfg+x(t)*({ns2}-50)/2 )) """
            eta=f"""0:(1-(t%(2+  10-.1*{nr1}  ))/ (2+10-.1*{nr1}) )*{ns1}*.1  * (e*(100-{nr2})+{nr2})*.01 """
        self.cfgsib={"Scheduler":n0,'Main Strength':ns1,'Sub- Strength':ns2,'Main Range':nr1,'Sub- Range':nr2}
        return self.runAdvanced(p,cfg,eta,dns   ,loops,nSingle)


    def runAdvanced(self, p, cfg,eta,dns    ,loops,nSingle):
        self.initSeed=p.seed
        #loops=p.batch_size
        loops = loops if (loops>0) else 1

        batch_count=p.n_iter
        state.job_count = loops*p.n_iter
        p.denoising_strength=p.denoising_strength or (1 if (self.isAdvanced) else 0.2)
        initial_denoising_strength=p.denoising_strength
        p.do_not_save_grid = True
        if hasattr(p,"init_images"):
            original_init_image = p.init_images
            initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]
        else:
            original_init_image=None
        
        all_images = []
        cfgsi=" loops:"+str(loops)+" terget denoising: "+str(dns)+"\nCFG: "+cfg+"\nETA: "+eta+"\n"
        
        p.extra_generation_params = {
                "CFG Scheduler Info":cfgsi,
            } 
        

        #if basic, add basic info as well
        if (self.isAdvanced==False):
            self.cfgsib.update(p.extra_generation_params)
            p.extra_generation_params=self.cfgsib

        if loops>1:
            processing.fix_seed(p)
            #self.initDenoise=p.denoising_strength
            
        for n in range(batch_count):
            proc=None
            history = []
            p.denoising_strength=initial_denoising_strength
            if (original_init_image!=None):
                p.init_images=original_init_image
            for loop in range(loops):
                if opts.img2img_color_correction and original_init_image!=None:
                        p.color_corrections = initial_color_corrections

                p.batch_size = 1
                p.n_iter = 1
                self.loop=loop
                self.prepare(p, cfg,eta)
                proc = process_images(p)
                if loop==0:
                    self.initInfo=proc.info
                    self.initSeed=proc.seed
                if len(proc.images)>0:
                    history.append(proc.images[0])
                    p.seed+=1
                    p.init_images=[proc.images[0]]
                    #p.denoising_strength=min(max(p.denoising_strength * dns, 0.05), 1)
                    p.denoising_strength=initial_denoising_strength+(dns-initial_denoising_strength)*(loop+1)/(loops)
                else:#interrupted
                    break
                #print("New denoising:"+str(p.denoising_strength)+"\n" )
            all_images += history
        if loops>0:#TODO:maybe this is not needed
            p.seed=self.initSeed
        #return proc if (loops==1 and p.batch_size==1) else Processed(p, all_images, self.initSeed, self.initInfo)
        return proc if(nSingle) else Processed(p, all_images, self.initSeed, self.initInfo)







    def peek(self,val):
        print(val)
        return val

    def split(self,src,default='0'):
        p=self.p
        self.P=copy.copy({
            'cfg':float(str(p.cfg_scale)),
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
            'peek':self.peek,
        })

        if src[0:4]=="eval":
            src="0:"+src[4:]
        if src[0]=="=":
            src="0:"+src[1:]

        #clean up
        while src[len(src)-1] in [";"," "]:
            src=src[0:len(src)-1]
        while src[0] in [";"," "]:
            src=src[1:]

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
              if math.floor(int(v[0]) if v[0].isnumeric() else float(v[0])*p.steps)==j:
                 val=v[1].strip()
                 break
              i=i+1
              
          #lets just evaluate all        
          if val[0]=="=":
            val=val[1:]
          
          _eta=1-j/p.steps
          params={'t':j,'T':p.steps,'math':math,'p':p,'e':float(str(_eta))}
          
          params.update(copy.copy(self.P))
          #print(params)
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
        _=a.split(":")[0]#splitter tester
        return math.floor(int(_) if (_.isnumeric()) else float(_)*self.p.steps)

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

    #def __add__(self,other):
        #print("ADD!")
    #    return self.get_fake_value(other)+other
    #def __sub__(self,other):
        #print("SUB!")
    #    return self.get_fake_value(other)-other



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
