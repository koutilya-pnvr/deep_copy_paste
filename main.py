from train import train_ic, train_disc, train_total

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import random,torch
import numpy as np
import os
import scipy.misc
import os.path
import argparse
import time
import torchvision.models as models

parser = argparse.ArgumentParser()
parser.add_argument('--use_pretrained', type=int, required=True, help='1 | 0')
parser.add_argument('--size', type=int, default=512, required=True, help='512 | 1024')
parser.add_argument('--bs', type=int, default=4, required=True, help='1 | 2 | 4 ...')
parser.add_argument('--pm','--pretraining_mode', dest='pretraining_mode', action='store_true', 
						help='Indicate if you are training for a pretrained version of the model')
parser.add_argument('--ft','--finetune', dest='finetuning_mode', action='store_true', 
						help='Indicate if you are finetuning the model on guide images')
parser.add_argument('--e','--evaluate', dest='evaluation_mode', action='store_true', 
						help='Indicate if you want inference mode')
parser.add_argument('--FE', dest='feature_extractor', help='Indicate which Imagenet pretrained network you want to use as Feature_Extractor')
parser.add_argument('--RC', dest='random_cropping', action='store_true', help='Indicate if you are using random cropping as DA')
parser.add_argument('--v', dest='vulcan', action='store_true', help='Use this if you are using vulcan')

def Feature_Extractor(feature_extractor):
	fe = models.vgg16(pretrained=True)
	if(feature_extractor == 'VGG16'):
		fe = models.vgg16(pretrained=True)
		fe = nn.Sequential(*list(fe.features.children())[:-1])
	for param in fe.parameters():
		param.requires_grad = False
	return fe


def network_factory(image_size,pretraining_mode,finetuning_mode, random_cropping, feature_extractor):
	l2loss=nn.L1Loss()
	if(pretraining_mode):
		if(image_size==256):
			from car_network_pretraining import image_completion_network as ic, global_discriminator_256 as gd
		elif(image_size==512):
			if(random_cropping):
				from car_network_pretraining import image_completion_network as ic, global_discriminator_256 as gd
			else:
				from car_network_pretraining import image_completion_network as ic, global_discriminator_512 as gd
		elif(image_size==1024):
			from car_network_pretraining import image_completion_network as ic, global_discriminator_1024 as gd
		
		def loss_fun(output_fake,mask,new_chip,gen):
			fe_fake=feature_extractor(output_fake)
			fe_gt=feature_extractor(new_chip)
			return l2loss(fe_fake,fe_gt)

	elif(not pretraining_mode and finetuning_mode):
		if(image_size==256):
			from car_network import image_completion_network as ic, global_discriminator_256 as gd
		elif(image_size==512):
			from car_network import image_completion_network as ic, global_discriminator_512 as gd
		elif(image_size==1024):
			from car_network import image_completion_network as ic, global_discriminator_1024 as gd
		
		def loss_fun(output_fake,mask,new_chip,gen):
			FM_fake_input=torch.cat((mask,output_fake),dim=1)
			features_fake=gen.FE2(FM_fake_input)
			features_real=gen.FE2(new_chip)
			FM_loss=l2loss(features_fake,features_real.detach())
			return FM_loss
	else:
		assert 0, "Invalid Image size : " + image_size + "or Invalid mode"

	return ic, gd, loss_fun

def dataset_factory(pretraining_mode):
	if(pretraining_mode):
		from data_loader_pretraining import NYC3dcars
		return NYC3dcars
	else:
		from data_loader import NYC3dcars
		return NYC3dcars

def save_checkpoint(state, filename):
    torch.save(state, filename)

def main():
	global opt, cuda_use
	opt = parser.parse_args()
	cuda_use=1
	seed=250
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	if(cuda_use==1):
		torch.cuda.manual_seed_all(seed)
		# torch.backends.cudnn.enabled = False
		# print("torch.backends.cudnn.enabled is: ", torch.backends.cudnn.enabled)

	batchsize=int(opt.bs)
	image_size=int(opt.size)

	fe=Feature_Extractor(opt.feature_extractor)
	if(cuda_use):
		fe.cuda()
	print(fe)
	inpainter,global_discriminator,loss_fun=network_factory(image_size, opt.pretraining_mode, opt.finetuning_mode, opt.random_cropping, fe)
	gen=inpainter()
	disc=global_discriminator()
	if(cuda_use):
		gen.cuda()
		disc.cuda()
	print(gen)
	print(disc)

	if(opt.vulcan):
		project_path='/fs/vulcan-scratch/koutilya/projects/deep_copy_paste/'
		data_path='/vulcan/scratch/koutilya/NYC3dcars/'
	else:
		project_path='/scratch0/projects/deep_copy_paste/'
		data_path='/scratch0/datasets/NYC3dcars/'

	dataset_class=dataset_factory(opt.pretraining_mode)
	train_dataset=dataset_class(root_dir=data_path, training=1, size=image_size, DA=opt.random_cropping)
	train_dataloader=DataLoader(train_dataset,batch_size=batchsize,shuffle=False)
	if(opt.random_cropping):
		val_dataset=dataset_class(root_dir=data_path, size=image_size/2)
	else:
		val_dataset=dataset_class(root_dir=data_path, size=image_size)
	val_dataloader=DataLoader(val_dataset,batch_size=batchsize,shuffle=False)

	optim_g=optim.Adam(gen.parameters(),lr=0.0001,betas=(0.5,0.999))
	optim_d=optim.Adam(disc.parameters(),lr=0.0001,betas=(0.5,0.999))

	criterion=nn.BCELoss()

	display_step=10

	if(opt.pretraining_mode):
		saved_model=os.path.join(project_path,'model_pretrained')
		# gen_saved_model='/scratch0/projects/deep_copy_paste/gen_model_pretrained'
		# disc_saved_model='/scratch0/projects/deep_copy_paste/disc_model_pretrained'
	else:
		saved_model=os.path.join(project_path,'model')
		# gen_saved_model='/scratch0/projects/deep_copy_paste/gen_model'
		# disc_saved_model='/scratch0/projects/deep_copy_paste/disc_model'

	if(os.path.isfile(saved_model) and opt.use_pretrained):
		model_state = torch.load(saved_model)
		gen.load_state_dict(model_state['gen_state_dict'])
		optim_g.load_state_dict(model_state['optimizer_g'])
		disc.load_state_dict(model_state['disc_state_dict'])
		optim_d.load_state_dict(model_state['optimizer_d'])
		start_epoch=model_state['epoch']

	elif(os.path.isfile(os.path.join(project_path,'model_pretrained')) and opt.finetuning_mode and not opt.use_pretrained):
		## Need to change this
		gen_state = torch.load(gen_saved_model)
		gen.load_state_dict(gen_state['state_dict'])
		
		disc_state = torch.load(disc_saved_model)
		disc.load_state_dict(disc_state['state_dict'])
	
	model={'gen':gen,'disc':disc}

	if(not opt.evaluation_mode):
		if(opt.pretraining_mode):
			epochs={'ic':3000,'disc':1000,'total':4000}
			# epochs={'ic':1,'disc':1,'total':1}
			run_pretraining(train_dataloader, val_dataloader, model, criterion, loss_fun, optim_g, optim_d, epochs, display_step, saved_model)
		elif(opt.finetuning_mode):
			epochs=3000
			run_training(train_dataloader, val_dataloader, model, criterion, loss_fun, optim_g, optim_d, epochs, display_step, saved_model)
	else:
		epoch=0
		validate(val_dataloader, model, epoch, cuda_use, os.path.join(project_path,'evaluation_mode_inpainter/'))

def run_pretraining(train_loader, val_loader, model, criterion, loss_fun, optim_g, optim_d, epochs, display_step, saved_model):
	epochs_ic,epochs_disc,epochs_total = epochs['ic'], epochs['disc'], epochs['total']

	for epoch in range(epochs_ic):
		train_ic(train_loader, model, loss_fun, optim_g, epoch, cuda_use)

		if(epoch%display_step==0):
			validate(val_loader, model, epoch, cuda_use, os.path.join(project_path,'evaluation_mode_inpainter/'))
			# save checkpoint
			save_checkpoint({
			    'epoch': epoch + 1,
			    'gen_state_dict': model['gen'].state_dict(),
			    'disc_state_dict': model['disc'].state_dict(),
			    'optimizer_g' : optim_g.state_dict(),
			    'optimizer_d' : optim_d.state_dict(),
			}, saved_model)

	for epoch in range(epochs_disc):
		train_disc(train_loader, model, criterion, optim_d, epoch, cuda_use)

	for epoch in range(epochs_total):
		train_total(train_loader, model, criterion, loss_fun, optim_g, optim_d, epoch, cuda_use)

		if(epoch%display_step==0):
			validate(val_loader, model, epochs_ic+epochs_disc+epoch, cuda_use, os.path.join(project_path,'evaluation_mode_inpainter/'))
			# save checkpoint
			save_checkpoint({
			    'epoch': epoch + 1,
			    'gen_state_dict': model['gen'].state_dict(),
			    'disc_state_dict': model['disc'].state_dict(),
			    'optimizer_g' : optim_g.state_dict(),
			    'optimizer_d' : optim_d.state_dict(),
			}, saved_model)

	return 

def run_training(train_loader, val_loader, model, criterion, loss_fun, optim_g, optim_d, epochs, display_step, saved_model):
	for epoch in range(epochs):
		train_total(train_loader, model, criterion, loss_fun, optim_g, optim_d, epoch)

		if(epoch%display_step==0):
			validate(val_loader, model, criterion)
			save_checkpoint({
	            'epoch': epoch + 1,
	            'gen_state_dict': model['gen'].state_dict(),
	            'disc_state_dict': model['disc'].state_dict(),
	            'optimizer_g' : optim_g.state_dict(),
	            'optimizer_d' : optim_d.state_dict(),
	        }, saved_model)
	return 

def validate(val_loader, model, epoch, cuda_use, folder='/scratch0/projects/deep_copy_paste/validation_inpainter/'):
	if(not os.path.exists(folder)):
		os.system('mkdir -p '+os.path.join(folder,'gt'))
		os.system('mkdir -p '+os.path.join(folder,'output'))
	
	os.system('mkdir -p '+os.path.join(folder,'output',str(epoch)))
	gen=model['gen']
	for i,data in enumerate(val_loader):
		input,new_chip,gt,pid,vid,path,filename=data
		if(cuda_use):
			input,new_chip=Variable(input.cuda(),volatile=True),Variable(new_chip.cuda(),volatile=True)
		else:
			input,new_chip=Variable(input,volatile=True),Variable(new_chip,volatile=True)
		
		for k in range(input.shape[0]):
			os.system('cp '+path[k]+' '+os.path.join(folder,'gt'))

		mask=input[:,0,:,:].unsqueeze(dim=1)
		val_output=gen(input,new_chip,mask)
		val_complete=(val_output+input[:,1:,:,:]).permute(0,2,3,1).cpu().data.numpy()
		for k in range(input.shape[0]):
			temp=val_complete[k,:,:,:]
			temp1=(temp-np.min(temp))/(np.max(temp)-np.min(temp))
			scipy.misc.imsave(os.path.join(folder,'output',str(epoch),filename[k].split('/')[-1]), temp1)

if __name__ == '__main__':
	main()