import numpy as np
import cv2,os,glob
from scipy.misc import imread,imsave

root_dir='/vulcan/scratch/koutilya'
modes=['val']

for mode in modes:
	cities=os.listdir(os.path.join(root_dir,'cityscapes/gtFine/',mode))
	print(cities)
	for city in cities:
		print(city)
		color_files=glob.glob(os.path.join(root_dir,'cityscapes/gtFine/',mode,city,'*color.png'))
		inst_files=[w.replace('color', 'instanceIds') for w in color_files]
		label_files=[w.replace('color', 'labelIds') for w in color_files]
		files=[w.replace('/gtFine/', '/leftImg8bit/') for w in color_files]
		files=[w.replace('gtFine_color', 'leftImg8bit') for w in files]
		print(color_files[1],inst_files[1],label_files[1],files[1])
		for i in range(len(color_files)):
			color=imread(color_files[i])
			inst=imread(inst_files[i])%1000
			label=imread(label_files[i])
			img=imread(files[i])
			filename=files[i].split('/')[-1][:-15]

			temp=color[:,:,2]
			new=np.zeros((256,256),dtype=np.uint8)
			mask=(np.logical_and(temp>140,temp<150).astype(np.uint8))
			if(np.max(mask)<1):
				print('No Cars in '+files[i])
				continue
			temp1=mask*inst

			bin_counts, bin_edges=np.histogram(temp1, range(np.max(temp1)+2))
			bin_counts=list(bin_counts)
			# _, contours1, hierarchy = cv2.findContours(temp1.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

			if(len(bin_counts)==1):
				# Car is there (otherwise wouldnt have come down) but the instanceid is 0
				temp1=mask
				bin_counts, bin_edges=np.histogram(temp1, range(np.max(temp1)+2))
				bin_counts=list(bin_counts)

			car_index=bin_counts.index(max(bin_counts[1:]))
			car_mask=np.zeros((1024,2048),dtype=np.uint8)
			car_mask[np.where(temp1==car_index)]=1

			_, contours, hierarchy = cv2.findContours(car_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			contours_max=sorted(contours,key=cv2.contourArea)
			contours_max.reverse()
			c=contours_max[0]
			mask1=np.copy(car_mask)
			x, y, w, h = cv2.boundingRect(c)
			cv2.rectangle(mask1, (x, y), (x+w, y+h), 5, 5)
			car_mask_rect=np.zeros_like(car_mask)
			car_mask_rect[y:y+h,x:x+w]=1

			img=imread(files[i])
			k=img[y:y+h,x:x+w,:]
			p=cv2.resize(k,(256,256))
			if(len(os.listdir(os.path.join(root_dir,'cityscapes_modified',mode,city)))==0):
				# os.system('mkdir '+os.path.join(root_dir,'cityscapes_modified',mode,city,'chip_original'))
				os.system('mkdir '+os.path.join(root_dir,'cityscapes_modified',mode,city,'chip'))
				os.system('mkdir '+os.path.join(root_dir,'cityscapes_modified',mode,city,'mask'))
			# imsave(os.path.join(root_dir,'cityscapes_modified',mode,city,'chip_original',filename+'chip.png'),k)
			imsave(os.path.join(root_dir,'cityscapes_modified',mode,city,'chip',filename+'chip.png'),p)
			imsave(os.path.join(root_dir,'cityscapes_modified',mode,city,'mask',filename+'mask.png'),car_mask_rect)


