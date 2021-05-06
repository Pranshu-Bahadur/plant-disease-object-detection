import torch.utils.data as data
from PIL import Image
import os
import os.path

def default_loader(path):
	return Image.open(path).convert('RGB')

def default_flist_reader(flist):
	"""
	flist format: impath label\nimpath label\n ...(same to caffe's filelist)
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			impath, imlabel = line.split("\n")
			imlist.append((impath, int(0 if imlabel=="Healthy\n" or imlabel=="normal\n" else 1)))#, int(1 if imlabel=="TB" else 0)) )
					
	return imlist

class ImageFilelist(data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		self.root   = root
		self.imlist = flist_reader(flist)		
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root,impath))
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target

	def __len__(self):
		return len(self.imlist)



#def default_loader(path):
#	return Image.open(path).convert('RGB')

def default_flist_reader_1(flist):
	"""
	flist format: impath label\nimpath label\n ...(same to caffe's filelist)
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			impath = line.split("\n")
			imlist.append((impath[0], 0 if "health" in impath[0] or ("extra" in impath[0] and (("da+db" and "/n" in impath[0]) or "0" in impath[0][-5])) else (1 if "sick" in impath[0] else 2)))
	#print(imlist)
	return imlist

class ImageFilelistWithLabels(data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader_1, loader=default_loader):
		self.root   = root
		self.imlist = flist_reader(flist)		
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root,impath))
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target

	def __len__(self):
		return len(self.imlist)