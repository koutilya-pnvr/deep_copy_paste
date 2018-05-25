from car_network import image_completion_network
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

cuda_use=1
seed=250

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if(cuda_use==1):
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.enabled = False
	print("torch.backends.cudnn.enabled is: ", torch.backends.cudnn.enabled)

training_batchsize=1

gen=image_completion_network()
if(cuda_use):
	gen.cuda()
print(gen)
train_dataset=cityscapes()
train_dataloader=DataLoader(train_dataset,batch_size=training_batchsize,shuffle=False)

optimizer=optim.Adam(gen.parameters(),lr=0.0001,betas=(0.5,0.999))
epochs=1000
l2loss=nn.MSELoss()
display_step=10
training_purpose_np=np.ones((512,1024),dtype=np.float32)
training_purpose=torch.from_numpy(training_purpose_np)
training_purpose=training_purpose.type(torch.FloatTensor)
if(cuda_use):
	training_purpose=Variable(training_purpose.cuda())
else:
	training_purpose=Variable(training_purpose)

for epoch in range(epochs):
	for i,data in enumerate(train_dataloader):
		if(i==1):
			break
		input,chip_only,chip_with_mask,gt,filenames=data
		
		if(cuda_use):
			input,chip_only,chip_with_mask,gt=Variable(input.cuda()),Variable(chip_only.cuda()),Variable(chip_with_mask.cuda()),Variable(gt.cuda())
		else:
			input,chip_only,chip_with_mask,gt=Variable(input),Variable(chip_only),Variable(chip_with_mask),Variable(gt)

		mask=input[:,0,:,:].unsqueeze(dim=1)
		inv_mask=training_purpose-mask

		optimizer.zero_grad()
		output=gen(input,chip_with_mask,mask)

		reconstructed_image=torch.max(output,input[:,1:,:,:])
		reconstructed_image=torch.cat((inv_mask,reconstructed_image),1)
		gt_with_invmask=torch.cat((inv_mask,gt),1)
		features_fake=gen.FE(reconstructed_image)
		features_real=gen.FE(gt_with_invmask)
		FM_loss=l2loss(features_fake,features_real.detach())

		cost=l2loss(output,chip_only)

		error=FM_loss+cost
		error.backward()
		optimizer.step()
		print('Epoch: '+str(epoch)+' Iteration: '+str(i)+' Training error: '+ str(cost.cpu().data.numpy()[0]))

	if((epoch)%display_step==0):
		os.system('mkdir -p '+os.path.join('/fs/vulcan-scratch/koutilya/projects/deep_copy_paste/val_testing/L2',str(epoch)))
		# reconstructed_image=torch.max(output,input[:,1:,:,:])
		output_cpu=(output.permute(0,2,3,1).cpu().data.numpy())
		input_cpu=(input[:,1:,:,:].permute(0,2,3,1).cpu().data.numpy())
		#print(input_cpu.shape)
		val_out_new=input_cpu+output_cpu
		for k in range(val_out_new.shape[0]):
			temp=val_out_new[k,:,:,:]
			val_out_un=255*(temp-np.min(temp))/(np.max(temp)-np.min(temp))
			scipy.misc.imsave(os.path.join('/fs/vulcan-scratch/koutilya/projects/deep_copy_paste/val_testing/L2',str(epoch),filenames[0]), val_out_un)

