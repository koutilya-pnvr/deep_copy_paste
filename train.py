import time
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch

def train_ic(train_loader, model, loss_fun, optim_g, epoch, cuda_use):
	gen, disc = model['gen'], model['disc']
	for i,data in enumerate(train_loader):
		start=time.time()
		input,new_chip,gt=data
		
		if(cuda_use):
			input,new_chip=Variable(input.cuda()),Variable(new_chip.cuda())
		else:
			input,new_chip=Variable(input),Variable(new_chip)

		#train with fake
		mask=input[:,0,:,:].unsqueeze(dim=1)
		# inv_mask=training_purpose-mask
		output_fake=gen(input,new_chip,mask)
		# fake_complete=(output_fake+input[:,1:,:,:])

		# update generator
		gen.zero_grad()

		error=loss_fun(output_fake,mask,new_chip,gen)
		error.backward()
		optim_g.step()
	print('Epoch: '+str(epoch)+' Iteration: '+str(i)+
			' time: '+str(time.time()-start)+
			' Error: '+ str(error.cpu().data.numpy()[0]))

def train_disc(train_loader, model, criterion, optim_d, epoch, cuda_use):
	gen,disc=model['gen'],model['disc']
	real_label=1
	fake_label=0

	for i,data in enumerate(train_loader):
		start=time.time()
		input,new_chip,gt=data
		bs=input.shape[0]
		label=torch.FloatTensor(bs)
		label.resize_(bs).fill_(real_label)

		if(cuda_use):
			input,new_chip,gt,labelv=Variable(input.cuda()),Variable(new_chip.cuda()),Variable(gt.cuda()),Variable(label.cuda())
		else:
			input,new_chip,gt,labelv=Variable(input),Variable(new_chip),Variable(gt),Variable(label)

		# update discriminator
		# train with real
		disc.zero_grad()
		disc_out=disc(gt)
		errD_real=criterion(disc_out,labelv) # labelv holds real labels here
		errD_real.backward()

		#train with fake
		mask=input[:,0,:,:].unsqueeze(dim=1)
		# inv_mask=training_purpose-mask
		output_fake=gen(input,new_chip,mask)
		if(cuda_use):
			labelv=Variable(label.fill_(fake_label).cuda())
		else:
			labelv=Variable(label.fill_(fake_label))
		fake_complete=(output_fake+input[:,1:,:,:])

		output=disc(fake_complete.detach())
		errD_fake=criterion(output,labelv) # labelv holds fake labels here
		# errD_fake.backward(retain_graph=True)
		errD_fake.backward()
		errD=errD_fake+errD_real
		optim_d.step()

	print('Epoch: '+str(epoch)+' Iteration: '+str(i)+
		' time: '+str(time.time()-start)+
		' Error_Fake: '+ str(errD_fake.cpu().data.numpy()[0])+
		' Error_Real: '+ str(errD_real.cpu().data.numpy()[0])+
		' Total Discriminator Error: '+ str(errD.cpu().data.numpy()[0]))


def train_total(train_loader, model, criterion, loss_fun, optim_g, optim_d, epoch, cuda_use):
	gen,disc=model['gen'],model['disc']
	real_label=1
	fake_label=0

	for i,data in enumerate(train_loader):
		start=time.time()
		input,new_chip,gt=data
		bs=input.shape[0]
		label=torch.FloatTensor(bs)
		label.resize_(bs).fill_(real_label)
		
		if(cuda_use):
			input,new_chip,gt,labelv=Variable(input.cuda()),Variable(new_chip.cuda()),Variable(gt.cuda()),Variable(label.cuda())
		else:
			input,new_chip,gt,labelv=Variable(input),Variable(new_chip),Variable(gt),Variable(label)

		# update discriminator
		# train with real
		disc.zero_grad()
		disc_out=disc(gt)
		errD_real=criterion(disc_out,labelv)
		errD_real.backward()

		#train with fake
		mask=input[:,0,:,:].unsqueeze(dim=1)
		# inv_mask=training_purpose-mask
		output_fake=gen(input,new_chip,mask)
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

		error=loss_fun(output_fake,mask,new_chip,gen)
		error.backward()
		optim_g.step()
	print('Epoch: '+str(epoch)+' Iteration: '+str(i)+
		' time: '+str(time.time()-start)+
		' Error: '+ str(error.cpu().data.numpy()[0])+
		' GAN_Gen_loss: '+ str(errG.cpu().data.numpy()[0])+
		' GAN_Disc_loss: '+ str(errD.cpu().data.numpy()[0]))