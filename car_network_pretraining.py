import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

class image_completion_network(nn.Module):
	def __init__(self):
		super(image_completion_network,self).__init__()
		self.hidden1=32
		self.hidden2=32
		self.hidden3=32
		self.hidden4=32
		self.hidden5=32

		self.hidden6=128
		self.hidden7=256
		self.hidden8=256
		self.hidden9=256

		self.hidden10=256
		self.hidden11=256
		self.hidden12=256
		self.hidden13=128
		self.hidden14=128
		self.hidden15=64
		self.hidden16=32
		self.hidden17=3

		self.elu=nn.ELU()
		#self.elu=F.tanh
		self.conv1 = nn.Conv2d(in_channels=4, out_channels=self.hidden1,
                               kernel_size=3,
                               stride=1,padding=1)
		self.conv1_bn=nn.BatchNorm2d(self.hidden1)

		self.conv2 = nn.Conv2d(in_channels=self.hidden1, out_channels=self.hidden2,
                               kernel_size=3,
                               stride=2,padding=1)
		self.conv2_bn=nn.BatchNorm2d(self.hidden2)

		self.conv3 = nn.Conv2d(in_channels=self.hidden2, out_channels=self.hidden3,
                               kernel_size=3,
                               stride=1,padding=1)
		self.conv3_bn=nn.BatchNorm2d(self.hidden3)

		self.conv4 = nn.Conv2d(in_channels=self.hidden3, out_channels=self.hidden4,
                               kernel_size=3,
                               stride=2,padding=1)
		self.conv4_bn=nn.BatchNorm2d(self.hidden4)

		self.conv5 = nn.Conv2d(in_channels=self.hidden4, out_channels=self.hidden5,
                               kernel_size=3,
                               stride=1,padding=1)
		self.conv5_bn=nn.BatchNorm2d(self.hidden5)

		self.dil_conv1 = nn.Conv2d(in_channels=1*(self.hidden5), out_channels=self.hidden6,
                               kernel_size=3,
                               stride=1,dilation=2,padding=2)
		self.dil_conv1_bn=nn.BatchNorm2d(self.hidden6)

		self.dil_conv2 = nn.Conv2d(in_channels=self.hidden6, out_channels=self.hidden7,
                               kernel_size=3,
                               stride=1,dilation=4,padding=4)
		self.dil_conv2_bn=nn.BatchNorm2d(self.hidden7)

		self.dil_conv3 = nn.Conv2d(in_channels=self.hidden7, out_channels=self.hidden8,
                               kernel_size=3,
                               stride=1,dilation=8,padding=8)
		self.dil_conv3_bn=nn.BatchNorm2d(self.hidden8)

		self.dil_conv4 = nn.Conv2d(in_channels=self.hidden8, out_channels=self.hidden9,
                               kernel_size=3,
                               stride=1,dilation=16,padding=16)
		self.dil_conv4_bn=nn.BatchNorm2d(self.hidden9)

		self.conv6 = nn.Conv2d(in_channels=self.hidden9, out_channels=self.hidden10,
                               kernel_size=3,
                               stride=1,padding=1)
		self.conv6_bn=nn.BatchNorm2d(self.hidden10)

		self.conv7 = nn.Conv2d(in_channels=self.hidden10, out_channels=self.hidden11,
                               kernel_size=3,
                               stride=1,padding=1)
		self.conv7_bn=nn.BatchNorm2d(self.hidden11)

		self.deconv1 = nn.ConvTranspose2d(in_channels=self.hidden11, out_channels=self.hidden12,
                               kernel_size=4,
                               stride=2,padding=1)
		self.deconv1_bn=nn.BatchNorm2d(self.hidden12)

		self.conv8 = nn.Conv2d(in_channels=self.hidden12, out_channels=self.hidden13,
                               kernel_size=3,
                               stride=1,padding=1)
		self.conv8_bn=nn.BatchNorm2d(self.hidden13)

		self.deconv2 = nn.ConvTranspose2d(in_channels=self.hidden13, out_channels=self.hidden14,
                               kernel_size=4,
                               stride=2,padding=1)
		self.deconv2_bn=nn.BatchNorm2d(self.hidden14)

		self.conv9 = nn.Conv2d(in_channels=self.hidden14, out_channels=self.hidden15,
                               kernel_size=3,
                               stride=1,padding=1)
		self.conv9_bn=nn.BatchNorm2d(self.hidden15)

		self.conv10 = nn.Conv2d(in_channels=self.hidden15, out_channels=self.hidden16,
                               kernel_size=3,
                               stride=1,padding=1)
		self.conv10_bn=nn.BatchNorm2d(self.hidden16)

		self.conv11 = nn.Conv2d(in_channels=self.hidden16, out_channels=self.hidden17,
                               kernel_size=3,
                               stride=1,padding=1)
		#self.conv11_bn=nn.BatchNorm2d(self.hidden17)
		#self.linear = nn.Linear(self.hidden17) # confirm there is no extra layer

		# self.conv1_2=nn.Conv2d(in_channels=4, out_channels=self.hidden1,
  #                              kernel_size=3,
  #                              stride=1,padding=1)
		# self.conv1_2_bn=nn.BatchNorm2d(self.hidden1)
		# self.conv2_2=nn.Conv2d(in_channels=self.hidden1, out_channels=self.hidden2,
  #                              kernel_size=3,
  #                              stride=2,padding=1)
		# self.conv2_2_bn=nn.BatchNorm2d(self.hidden2)
		# self.conv3_2=nn.Conv2d(in_channels=self.hidden2, out_channels=self.hidden3,
  #                              kernel_size=3,
  #                              stride=1,padding=1)
		# self.conv3_2_bn=nn.BatchNorm2d(self.hidden3)
		# self.conv4_2=nn.Conv2d(in_channels=self.hidden3, out_channels=self.hidden4,
  #                              kernel_size=3,
  #                              stride=2,padding=1)
		# self.conv4_2_bn=nn.BatchNorm2d(self.hidden4)
		# self.conv5_2=nn.Conv2d(in_channels=self.hidden4, out_channels=self.hidden5,
  #                              kernel_size=3,
  #                              stride=1,padding=1)
		# self.conv5_2_bn=nn.BatchNorm2d(self.hidden5)
	def FE(self,x):
		x=self.elu(self.conv2_bn(self.conv2(self.elu(self.conv1_bn(self.conv1(x))))))
		x=self.elu(self.conv4_bn(self.conv4(self.elu(self.conv3_bn(self.conv3(x))))))
		x=self.elu(self.conv5_bn(self.conv5(x)))
		return x
	# def FE2(self,x):
	# 	x=self.elu(self.conv2_2_bn(self.conv2_2(self.elu(self.conv1_2_bn(self.conv1_2(x))))))
	# 	x=self.elu(self.conv4_2_bn(self.conv4_2(self.elu(self.conv3_2_bn(self.conv3_2(x))))))
	# 	x=self.elu(self.conv5_2_bn(self.conv5_2(x)))
	# 	return x

	def forward(self,x,new_chip,mask):
		x1=self.FE(x)
		# x2=self.FE2(new_chip)
		# x=torch.cat((x1,x2),1)
		x=x1
		x=self.elu(self.dil_conv2_bn(self.dil_conv2(self.elu(self.dil_conv1_bn(self.dil_conv1(x))))))
		x=self.elu(self.dil_conv4_bn(self.dil_conv4(self.elu(self.dil_conv3_bn(self.dil_conv3(x))))))
		x=self.elu(self.conv7_bn(self.conv7(self.elu(self.conv6_bn(self.conv6(x))))))
		x=self.elu(self.conv8_bn(self.conv8(self.elu(self.deconv1_bn(self.deconv1(x))))))
		x=self.elu(self.conv9_bn(self.conv9(self.elu(self.deconv2_bn(self.deconv2(x))))))
		x=self.conv11(self.elu(self.conv10_bn(self.conv10(x))))
		return (F.sigmoid(x)*mask)

class global_discriminator_1024(nn.Module):
	def __init__(self):
		super(global_discriminator_1024,self).__init__()

		self.hidden1=32
		self.hidden2=32
		self.hidden3=32
		self.hidden4=32
		self.hidden5=32
		self.hidden6=32
		self.hidden7=32
		self.hidden8=32
		self.kernel_size=5
		self.elu=nn.ELU()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.hidden1,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv1_bn=nn.BatchNorm2d(self.hidden1)
		self.conv2 = nn.Conv2d(in_channels=self.hidden1, out_channels=self.hidden2,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv2_bn=nn.BatchNorm2d(self.hidden2)
		self.conv3 = nn.Conv2d(in_channels=self.hidden2, out_channels=self.hidden3,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv3_bn=nn.BatchNorm2d(self.hidden3)
		self.conv4 = nn.Conv2d(in_channels=self.hidden3, out_channels=self.hidden4,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv4_bn=nn.BatchNorm2d(self.hidden4)
		self.conv5 = nn.Conv2d(in_channels=self.hidden4, out_channels=self.hidden5,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv5_bn=nn.BatchNorm2d(self.hidden5)
		self.conv6 = nn.Conv2d(in_channels=self.hidden5, out_channels=self.hidden6,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv6_bn=nn.BatchNorm2d(self.hidden6)
		self.conv7 = nn.Conv2d(in_channels=self.hidden6, out_channels=self.hidden7,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv7_bn=nn.BatchNorm2d(self.hidden7)
		self.conv8 = nn.Conv2d(in_channels=self.hidden7, out_channels=self.hidden8,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv8_bn=nn.BatchNorm2d(self.hidden8)
		self.main=nn.Sequential(nn.Linear(288,10),nn.Tanh(),nn.Linear(10,1),nn.Sigmoid())
	def forward(self,x):
		x=self.elu(self.conv2_bn(self.conv2(self.elu(self.conv1_bn(self.conv1(x))))))
		x=self.elu(self.conv4_bn(self.conv4(self.elu(self.conv3_bn(self.conv3(x))))))
		x=self.elu(self.conv6_bn(self.conv6(self.elu(self.conv5_bn(self.conv5(x))))))
		x=self.elu(self.conv8_bn(self.conv8(self.elu(self.conv7_bn(self.conv7(x))))))
		x=x.view(x.shape[0],-1)
		x=self.main(x)
		return x.view(-1,1).squeeze(1)
		
class global_discriminator_512(nn.Module):
	def __init__(self):
		super(global_discriminator_512,self).__init__()

		self.hidden1=32
		self.hidden2=32
		self.hidden3=32
		self.hidden4=32
		self.hidden5=32
		self.hidden6=32
		self.hidden7=32
		self.hidden8=32
		self.kernel_size=5
		self.elu=nn.ELU()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.hidden1,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv1_bn=nn.BatchNorm2d(self.hidden1)
		self.conv2 = nn.Conv2d(in_channels=self.hidden1, out_channels=self.hidden2,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv2_bn=nn.BatchNorm2d(self.hidden2)
		self.conv3 = nn.Conv2d(in_channels=self.hidden2, out_channels=self.hidden3,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv3_bn=nn.BatchNorm2d(self.hidden3)
		self.conv4 = nn.Conv2d(in_channels=self.hidden3, out_channels=self.hidden4,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv4_bn=nn.BatchNorm2d(self.hidden4)
		self.conv5 = nn.Conv2d(in_channels=self.hidden4, out_channels=self.hidden5,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv5_bn=nn.BatchNorm2d(self.hidden5)
		self.conv6 = nn.Conv2d(in_channels=self.hidden5, out_channels=self.hidden6,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv6_bn=nn.BatchNorm2d(self.hidden6)
		self.conv7 = nn.Conv2d(in_channels=self.hidden6, out_channels=self.hidden7,
                               kernel_size=self.kernel_size,
                               stride=2,padding=1)
		self.conv7_bn=nn.BatchNorm2d(self.hidden7)
		self.main=nn.Sequential(nn.Linear(288,10),nn.Tanh(),nn.Linear(10,1),nn.Sigmoid())
	def forward(self,x):
		x=self.elu(self.conv2_bn(self.conv2(self.elu(self.conv1_bn(self.conv1(x))))))
		x=self.elu(self.conv4_bn(self.conv4(self.elu(self.conv3_bn(self.conv3(x))))))
		x=self.elu(self.conv6_bn(self.conv6(self.elu(self.conv5_bn(self.conv5(x))))))
		x=self.elu(self.conv7_bn(self.conv7(x)))
		x=x.view(x.shape[0],-1)
		x=self.main(x)
		return x.view(-1,1).squeeze(1)

# disc=global_discriminator_512()
# print([name for name,params in disc.named_parameters()])
# print(list(disc.parameters())[0])
# x=Variable(torch.FloatTensor(1,4,512,512).zero_())
# mask=Variable(torch.FloatTensor(1,3,512,512).zero_())
# disc(mask)