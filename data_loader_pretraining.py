from torch.utils.data import Dataset,DataLoader
import numpy as np
import os,glob,random
import pandas as pd
import torch
from scipy.misc import imread,imsave,imresize
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors
from math import ceil,floor

class NYC3dcars(Dataset):
	def __init__(self,root_dir='/scratch0/datasets/NYC3dcars/',training=0,size=512,DA=False):
		super(NYC3dcars,self).__init__()
		self.size=size
		self.DA=DA
		self.root_dir=root_dir
		self.training=training
		self.complete_images_path=os.path.join(self.root_dir,'times-square-images/')
		self.datasets=pd.read_csv(os.path.join(self.root_dir,'nyc3dcars-csv/datasets.csv'))
		self.photos=pd.read_pickle(os.path.join(self.root_dir,'nyc3dcars-csv/preprocessed_photos'))
		self.vehicles=pd.read_csv(os.path.join(self.root_dir,'nyc3dcars-csv/vehicles.csv'))
		self.vehicle_types=pd.read_csv(os.path.join(self.root_dir,'nyc3dcars-csv/vehicle_types.csv'))
		self.vehicles.drop(self.vehicles.columns[[13,14]], axis=1, inplace=True)
		self.photos1=self.photos[['id', 'filename','width', 'height','roll','sees_ground', 'camera_height','test', 'dataset_id','daytime']]
		self.photos1 = self.photos1.rename(columns={'id': 'pid'})
		self.right_result=pd.merge(self.photos1, self.vehicles, how='inner', on='pid',
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
		self.right_result['area_bb']=self.right_result.apply(lambda row: (min(row.x2,1)-max(row.x1,0))*(min(row.y2,1)-max(row.y1,0)), axis=1)
		k=0.2
		threshold=k*self.right_result['area_bb'].mean()
		self.right_result=self.right_result.loc[self.right_result['area_bb']>=threshold]
		self.right_result=self.right_result.loc[self.right_result['occlusion']<=1]
		# By this point we would have selected all the images in times_square_images with atleast a vehicle that has decent proportion of area in the image it belongs
		# Also we are only considering cars with <=1 occlusion

		self.right_result_ts=self.right_result.sample(frac=1,random_state=250).reset_index(drop=True) # shuffle all the rows in a consistent fashion
		# self.right_result_ts=self.right_result_ts.loc[self.right_result_ts['occlusion']<=1] # consider only little occluded cars during training or testing
		self.right_result_ts.index = np.arange(0, len(self.right_result_ts))
		self.right_result1=self.right_result_ts.copy()

		if(self.training):
			self.right_result_ts=self.right_result_ts.iloc[:len(self.right_result_ts)-100,:]
			self.right_result_ts.index = np.arange(0, len(self.right_result_ts))
		else:
			self.right_result_ts=self.right_result_ts.iloc[-100:,:] ## 100 samples for testing
			self.right_result_ts.index = np.arange(0, len(self.right_result_ts))
		
	def __len__(self):
		return len(self.right_result_ts)

	def __getitem__(self,idx):
		filename=self.right_result_ts.loc[idx,'filename']
		pid=self.right_result_ts.loc[idx,'pid']
		vid=self.right_result_ts.loc[idx,'id']
		day_time=self.right_result_ts.loc[idx,'daytime']

		input=imread(os.path.join(self.complete_images_path,filename))
		input_resized=resize(input,(self.size,self.size))
		x1,y1,x2,y2=self.right_result_ts.loc[idx,'x1'],self.right_result_ts.loc[idx,'y1'],self.right_result_ts.loc[idx,'x2'],self.right_result_ts.loc[idx,'y2']
		x1,y1,x2,y2=max(0,x1),max(0,y1),min(1,x2),min(1,y2)
		w,h=self.right_result_ts.loc[idx,'width'],self.right_result_ts.loc[idx,'height']
		# bb_x1,bb_y1,bb_w,bb_h=int(x1*w),int(y1*h),int((x2-x1)*w),int((y2-y1)*h)
		# bb_x1n,bb_y1n,bb_wn,bb_hn=int(bb_x1*self.size.0/w),int(bb_y1*self.size.0/h),int(bb_w*self.size.0/w),int(bb_h*self.size.0/h)
		bb_x1,bb_y1,bb_x2,bb_y2 = int(floor(x1*self.size)),int(floor(y1*self.size)),int(ceil(x2*self.size)),int(ceil(y2*self.size)) 
		bb_x1,bb_y1,bb_x2,bb_y2=max(bb_x1,0),max(bb_y1,0),min(bb_x2,self.size-1),min(bb_y2,self.size-1) # since we are limiting our input image to self.sizexself.size
		bb_w,bb_h=(bb_x2-bb_x1)+1,(bb_y2-bb_y1)+1

		chip=np.zeros((self.size,self.size,3),dtype=np.float32)
		chip[bb_y1:(bb_y2+1),bb_x1:(1+bb_x2),:]=input_resized[bb_y1:(bb_y2+1),bb_x1:(1+bb_x2),:]

		condition=not(bb_w>256 or bb_h>256) # check if the car mask itself is more than 256

		if(self.DA and self.training):
			if(condition):
				p,q=random.randint(max(0,bb_x2-256),min(bb_x1,255)),random.randint(max(0,bb_y2-256),min(bb_y1,255))
				# print((p,q),(bb_x1,bb_y1),(bb_x2,bb_y2))
				chip_da=np.zeros((256,256,3),dtype=np.float32)
				chip_da=chip[q:q+256,p:p+256,:]
				chip=chip_da.copy()
			else:
				chip=resize(chip,(256,256))

		horizontal_flip_parameter=random.randint(0,1)
		if(horizontal_flip_parameter and self.training):
			chip=np.flip(chip,axis=1).copy()

		# imsave('chip_'+str(idx)+'.jpg',chip)
		chip=torch.from_numpy(chip)
		chip=chip.type(torch.FloatTensor)
		chip_only=chip.permute(2,0,1)

		box=np.ones((bb_h,bb_w,1),dtype=np.float32)
		mask=np.zeros((self.size,self.size,1),dtype=np.float32)
		mask[bb_y1:(bb_y2+1),bb_x1:(1+bb_x2),:]=box
		inv_mask=np.ones((self.size,self.size,1),dtype=np.float32)-mask
		input_new=input_resized*inv_mask
		# input_new/=255.0

		if(self.DA and self.training):
			if(condition):
				input_da=np.zeros((256,256,3),dtype=np.float32)
				input_da=input_new[q:q+256,p:p+256,:]
				input_new=input_da.copy()
				mask_da=np.zeros((256,256,1),dtype=np.float32)
				mask_da=mask[q:q+256,p:p+256,:]
				mask=mask_da.copy()
			else:
				input_new=resize(input_new,(256,256))
				mask=resize(mask,(256,256))

		if(horizontal_flip_parameter and self.training):
			input_new=np.flip(input_new,axis=1).copy()
		
		# imsave('input_'+str(idx)+'.jpg',input_new)
		input=torch.from_numpy(input_new)
		input=input.type(torch.FloatTensor)
		input=torch.cat((torch.from_numpy(mask).type(torch.FloatTensor),input),dim=2)
		input=input.permute(2,0,1)

		gt=input_resized
		if(self.DA and self.training):
			if(condition):
				gt_da=np.zeros((256,256,3),dtype=np.float32)
				gt_da=gt[q:q+256,p:p+256,:]
				gt=gt_da.copy()
			else:
				gt=resize(gt,(256,256))

		if(horizontal_flip_parameter and self.training):
			gt=np.flip(gt,axis=1).copy()
		# gt/=255.0

		# imsave('gt_'+str(idx)+'.png',gt)
		gt=torch.from_numpy(gt)
		gt=gt.type(torch.FloatTensor)
		gt=gt.permute(2,0,1)
		
		if(self.training):
			return input,chip_only,gt
		else:
			return input,chip_only,gt,pid,vid,os.path.join(self.complete_images_path,filename),filename

# cnt=0
train_dataset=NYC3dcars(training=1,DA=True)
# print(train_dataset.__len__())
# # val_dataset=NYC3dcars()
# # print(val_dataset.__len__())
train_dataloader=DataLoader(train_dataset,batch_size=100,shuffle=False)
# # val_dataloader=DataLoader(val_dataset,batch_size=10,shuffle=False)
for i,data in enumerate(train_dataloader):
# 	if(i==1):
# 		break
	input,new_chip,gt,count=data #,pid,vid,path,filename
	# cnt+=torch.sum(count)
	# print('Iteration: '+str(i)+' cnt: '+ str(cnt))
	# print(input.shape,new_chip.shape,gt.shape)
# 	print(h)
	# print(len(set(list(path))))
# # 	# print(filename[0])
# # 	# k=input[0,1:,:,:].permute(1,2,0).cpu().numpy()
# # 	# print(k.shape)
# # 	# imsave('test_input.png',k)
# print(i)
# train_dataset[61]