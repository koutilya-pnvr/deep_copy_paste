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
	def __init__(self,root_dir='/scratch0/datasets/NYC3dcars/',training=0,size=512):
		super(NYC3dcars,self).__init__()
		self.size=size
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
		
		num_rows=self.right_result1.shape[0]
		theta=[self.right_result1.loc[i,'view_theta'] for i in range(0,num_rows)]
		phi=[self.right_result1.loc[i,'view_phi'] for i in range(0,num_rows)]
		x=[self.right_result1.loc[i,'x'] for i in range(0,num_rows)]
		z=[self.right_result1.loc[i,'z'] for i in range(0,num_rows)]
		th=[self.right_result1.loc[i,'theta'] for i in range(0,num_rows)]
		occ=[self.right_result1.loc[i,'occlusion'] for i in range(0,num_rows)]
		cam_height=[self.right_result1.loc[i,'camera_height'] for i in range(0,num_rows)]
		daytime=[self.right_result1.loc[i,'daytime'] for i in range(0,num_rows)]

		self.filenames=[self.right_result1.loc[i,'filename'] for i in range(0,num_rows)]
		self.pids=[self.right_result1.loc[i,'pid'] for i in range(0,num_rows)]
		self.vids=[self.right_result1.loc[i,'id'] for i in range(0,num_rows)]

		self.theta_max,self.theta_min=self.stats(theta)
		self.phi_max,self.phi_min=self.stats(phi)
		self.x_max,self.x_min=self.stats(x)
		self.z_max,self.z_min=self.stats(z)
		self.th_max,self.th_min=self.stats(th)
		self.cam_height_max,self.cam_height_min=self.stats(cam_height)

		X=np.zeros((num_rows,7),dtype=np.float32)
		self.normalized_param=1
		for i in range(0,self.right_result1.shape[0]):
			X[i,0]=self.normalized(theta[i],self.theta_max,self.theta_min,self.normalized_param)
			X[i,1]=self.normalized(phi[i],self.phi_max,self.phi_min,self.normalized_param)
			X[i,2]=self.normalized(x[i],self.x_max,self.x_min,self.normalized_param)
			X[i,3]=self.normalized(z[i],self.z_max,self.z_min,self.normalized_param)
			X[i,4]=self.normalized(th[i],self.th_max,self.th_min,self.normalized_param)
			X[i,5]=self.normalized(cam_height[i],self.cam_height_max,self.cam_height_min,self.normalized_param)
			X[i,6]=int(daytime[i]=='t')

		ind0=self.find_indices(X,dim=6,value=0.0)
		ind1=self.find_indices(X,dim=6,value=1.0)
		X0=X[ind0]
		X1=X[ind1]
		self.pids0=[self.pids[i] for i in ind0]
		self.pids1=[self.pids[i] for i in ind1]
		self.vids0=[self.vids[i] for i in ind0]
		self.vids1=[self.vids[i] for i in ind1]

		self.nbrs0 = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X0)
		self.nbrs1 = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X1)
		# distances, indices = nbrs.kneighbors(X)
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

		chip=input_resized[bb_y1:(bb_y2+1),bb_x1:(1+bb_x2),:]
		# chip_resized=resize(chip,(256,256))
		# chip_resized/=255.0

		box=np.ones((chip.shape[0],chip.shape[1],1),dtype=np.float32)
		mask=np.zeros((self.size,self.size,1),dtype=np.float32)
		mask[bb_y1:(bb_y2+1),bb_x1:(1+bb_x2),:]=box
		inv_mask=np.ones((self.size,self.size,1),dtype=np.float32)-mask
		input_new=input_resized*inv_mask
		# input_new/=255.0

		horizontal_flip_parameter=random.randint(0,1)
		if(horizontal_flip_parameter and self.training):
			input_new=np.flip(input_new,axis=1).copy()
		# imsave('input_'+str(idx)+'.jpg',input_new)

		input=torch.from_numpy(input_new)
		input=input.type(torch.FloatTensor)
		input=torch.cat((torch.from_numpy(mask).type(torch.FloatTensor),input),dim=2)
		input=input.permute(2,0,1)
		

		th,x,z=self.right_result_ts.loc[idx,'theta'],self.right_result_ts.loc[idx,'x'],self.right_result_ts.loc[idx,'z']
		cam_height,occ,daytime=self.right_result_ts.loc[idx,'camera_height'],self.right_result_ts.loc[idx,'occlusion'],self.right_result_ts.loc[idx,'daytime']
		theta,phi=self.right_result_ts.loc[idx,'view_theta'],self.right_result_ts.loc[idx,'view_phi']
		X=np.zeros((1,7),dtype=np.float32)

		X[0,0]=self.normalized(theta,self.theta_max,self.theta_min,self.normalized_param)
		X[0,1]=self.normalized(phi,self.phi_max,self.phi_min,self.normalized_param)
		X[0,2]=self.normalized(x,self.x_max,self.x_min,self.normalized_param)
		X[0,3]=self.normalized(z,self.z_max,self.z_min,self.normalized_param)
		X[0,4]=self.normalized(th,self.th_max,self.th_min,self.normalized_param)
		X[0,5]=self.normalized(cam_height,self.cam_height_max,self.cam_height_min,self.normalized_param)
		X[0,6]=int(daytime=='t')

		if(X[0,6]==1):
			_,indices=self.nbrs1.kneighbors(X)
		else:
			_,indices=self.nbrs0.kneighbors(X)

		if(self.training):
			pick=random.randint(0,5)
		else:
			pick=1
		idx_new=indices[0,pick]

		pid_new=self.pids1[idx_new] if X[0,6]==1 else self.pids0[idx_new]
		vid_new=self.vids1[idx_new] if X[0,6]==1 else self.vids0[idx_new]
		# filename_new_resized=str(pid_new)+'_'+str(vid_new)+'_resized.jpg'
		filename_new=str(pid_new)+'_'+str(vid_new)+'.jpg'
		# new_chip=imread(os.path.join(self.root_dir,'all_cars/resized',filename_new_resized))
		new_chip_original=imread(os.path.join(self.root_dir,'all_cars/original/',filename_new))
		new_chip=resize(new_chip_original,(bb_h,bb_w))
		second_image=np.zeros((self.size,self.size,3),dtype=np.float32)
		second_image[bb_y1:(bb_y2+1),bb_x1:(1+bb_x2),:]=new_chip
		# second_image/=255.0
		if(horizontal_flip_parameter and self.training):
			second_image=np.flip(second_image,axis=1).copy()
		# imsave('second_image_'+str(idx)+'.jpg',second_image)
		
		new_chip=torch.from_numpy(second_image)
		new_chip=new_chip.type(torch.FloatTensor)
		new_chip=torch.cat((torch.from_numpy(mask).type(torch.FloatTensor),new_chip),dim=2)
		new_chip=new_chip.permute(2,0,1)

		# chip=torch.from_numpy(chip_resized)
		# chip=chip.type(torch.FloatTensor)
		# chip=chip.permute(2,0,1)

		gt=input_resized
		if(horizontal_flip_parameter and self.training):
			gt=np.flip(gt,axis=1).copy()
		# gt/=255.0
		gt=torch.from_numpy(gt)
		gt=gt.type(torch.FloatTensor)
		gt=gt.permute(2,0,1)
		
		# print(torch.max(input),torch.max(new_chip),torch.max(gt))
		if(self.training):
			return input,new_chip,gt
		else:
			return input,new_chip,gt,pid,vid,os.path.join(self.complete_images_path,filename),filename

	def stats(self,a):
		return max(a),min(a)
	def normalized(self,a,ma,mi,normalized_param=1):
		if(normalized_param):
			return (a)/(ma)
		else:
			return a
	def find_indices(self,X,dim=6,value=1.0):
		indices=[]
		for i in range(len(X)):
			if(X[i,dim]==value):
				indices.append(i)
		return indices

# train_dataset=NYC3dcars(training=1)
# print(train_dataset.__len__())
# val_dataset=NYC3dcars()
# print(val_dataset.__len__())
# train_dataloader=DataLoader(train_dataset,batch_size=5,shuffle=False)
# # # # # val_dataloader=DataLoader(val_dataset,batch_size=10,shuffle=False)
# for i,data in enumerate(train_dataloader):
# 	if(i==1):
# 		break
# 	input,new_chip,gt=data
	# print(input.shape,new_chip.shape,gt.shape)	
# # 	# print(filename[0])
# # 	# k=input[0,1:,:,:].permute(1,2,0).cpu().numpy()
# # 	# print(k.shape)
# # 	# imsave('test_input.png',k)
# print(i)
