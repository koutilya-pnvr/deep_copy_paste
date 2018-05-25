from car_network import image_completion_network,global_discriminator
from data_loader import cityscapes

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

parser = argparse.ArgumentParser()
parser.add_argument('--use_pretrained', type=int, required=True, help='1 | 0')
opt = parser.parse_args()

cuda_use=1
seed=250

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if(cuda_use==1):
	torch.cuda.manual_seed_all(seed)
	# torch.backends.cudnn.enabled = True
	# print("torch.backends.cudnn.enabled is: ", torch.backends.cudnn.enabled)

training_batchsize=4

gen=image_completion_network()
disc=global_discriminator()
if(cuda_use):
	gen.cuda()
	disc.cuda()
print(gen)
print(disc)
train_dataset=cityscapes()
train_dataloader=DataLoader(train_dataset,batch_size=training_batchsize,shuffle=False)
val_dataset=cityscapes(root_dir='/vulcan/scratch/koutilya/cityscapes/leftImg8bit/val')
val_dataloader=DataLoader(val_dataset,batch_size=4,shuffle=False)

optimizer=optim.Adam(gen.parameters(),lr=0.0001,betas=(0.5,0.999))
optim_d=optim.Adam(disc.parameters(),lr=0.0001,betas=(0.5,0.999))

epochs=10000
start_epoch=0
l2loss=nn.MSELoss()
criterion=nn.BCELoss()

display_step=10
training_purpose_np=np.ones((512,1024),dtype=np.float32)
training_purpose=torch.from_numpy(training_purpose_np)
training_purpose=training_purpose.type(torch.FloatTensor)
if(cuda_use):
	training_purpose=Variable(training_purpose.cuda())
else:
	training_purpose=Variable(training_purpose)

label=torch.FloatTensor(training_batchsize)
real_label=1
fake_label=0

gen_saved_model='/fs/vulcan-scratch/koutilya/projects/core3d/gen_model'
disc_saved_model='/fs/vulcan-scratch/koutilya/projects/core3d/disc_model'

if(os.path.isfile(gen_saved_model) and int(opt.use_pretrained)):
	gen_state = torch.load(gen_saved_model)
	gen.load_state_dict(gen_state['state_dict'])
	optimizer.load_state_dict(gen_state['optimizer'])

	disc_state = torch.load(disc_saved_model)
	disc.load_state_dict(disc_state['state_dict'])
	optim_d.load_state_dict(disc_state['optimizer'])

	start_epoch=disc.load_state_dict(disc_state['epoch'])

for epoch in range(start_epoch+1,epochs):
	for i,data in enumerate(train_dataloader):
		input,chip_only,chip_with_mask,gt,filenames=data
		label.resize_(training_batchsize).fill_(real_label)

		if(cuda_use):
			input,chip_only,chip_with_mask,gt,labelv=Variable(input.cuda()),Variable(chip_only.cuda()),Variable(chip_with_mask.cuda()),Variable(gt.cuda()),Variable(label.cuda())
		else:
			input,chip_only,chip_with_mask,gt,labelv=Variable(input),Variable(chip_only),Variable(chip_with_mask),Variable(gt),Variable(label)

		# update discriminator
		# train with real
		disc.zero_grad()
		disc_out=disc(gt)
		errD_real=criterion(disc_out,labelv)
		errD_real.backward()

		#train with fake
		mask=input[:,0,:,:].unsqueeze(dim=1)
		inv_mask=training_purpose-mask
		output_fake=gen(input,chip_with_mask,mask)
		if(cuda_use):
			labelv=Variable(label.fill_(fake_label).cuda())
		else:
			labelv=Variable(label.fill_(fake_label))
		fake_complete=(output_fake+input[:,1:,:,:])

		output=disc(fake_complete.detach())
		errD_fake=criterion(output,labelv)
		errD_fake.backward(retain_graph=True)
		errD=errD_fake+errD_real
		optim_d.step()

		# update generator
		gen.zero_grad()
		if(cuda_use):
			labelv=Variable(label.fill_(real_label).cuda())
		else:
			labelv=Variable(label.fill_(real_label))

		output=disc(fake_complete)
		errG=criterion(output,labelv)
		errG.backward(retain_graph=True)

		# L2 and FM (Feature Matching) losses
		reconstructed_image=fake_complete
		reconstructed_image=torch.cat((inv_mask,reconstructed_image),1)
		gt_with_invmask=torch.cat((inv_mask,gt),1)
		features_fake=gen.FE(reconstructed_image)
		features_real=gen.FE(gt_with_invmask)
		FM_loss=l2loss(features_fake,features_real.detach())

		# cost=l2loss(output_fake,chip_only)

		error=100*FM_loss
		error.backward()
		optimizer.step()
		print('Epoch: '+str(epoch)+' Iteration: '+str(i)+
			' FM_loss: '+ str(FM_loss.cpu().data.numpy()[0])+
			' GAN_Gen_loss: '+ str(errG.cpu().data.numpy()[0])+
			' GAN_Disc_loss: '+ str(errD.cpu().data.numpy()[0])+
			' Total_Gen_Training_error: '+ str(error.cpu().data.numpy()[0]))

	if((epoch)%display_step==0):
		os.system('mkdir -p '+os.path.join('/fs/vulcan-scratch/koutilya/projects/deep_copy_paste/val_testing/L2',str(epoch)))
		for j,data in enumerate(val_dataloader):
			input,chip_only,chip_with_mask,gt,filenames=data
			if(cuda_use):
				input,chip_only,chip_with_mask,gt,labelv=Variable(input.cuda()),Variable(chip_only.cuda()),Variable(chip_with_mask.cuda()),Variable(gt.cuda()),Variable(label.cuda())
			else:
				input,chip_only,chip_with_mask,gt,labelv=Variable(input),Variable(chip_only),Variable(chip_with_mask),Variable(gt),Variable(label)
			mask=input[:,0,:,:].unsqueeze(dim=1)
			output=gen(input,chip_with_mask,mask)
			reconstructed_image=output+input[:,1:,:,:]
			output_cpu=reconstructed_image.permute(0,2,3,1).cpu().data.numpy()
			for k in range(output_cpu.shape[0]):
				temp=output_cpu[k,:,:,:]
				temp1=255*(temp-np.min(temp))/(np.max(temp)-np.min(temp))
				scipy.misc.imsave(os.path.join('/fs/vulcan-scratch/koutilya/projects/deep_copy_paste/val_testing/L2',str(epoch),filenames[k]), temp1)

		gen_state={
        'epoch': epoch ,
        'state_dict': gen.state_dict(),
        'optimizer' : optimizer.state_dict()}
		torch.save(gen_state, gen_saved_model)
		disc_state={
        'epoch': epoch ,
        'state_dict': disc.state_dict(),
        'optimizer' : optim_d.state_dict()}
		torch.save(disc_state, disc_saved_model)