import numpy as np
from utils import face_detect
import cv2
import os
from imutils import paths
import pickle
import sys
import time
import vptree
import threading
from progress import Progress
def dhash(image, hashSize=8):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	diff = resized[:, 1:] > resized[:, :-1]
	result = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
	return int(np.array(result, dtype="float64"))
def hamming(a, b):
	return bin(int(a) ^ int(b)).count("1")
def convert_hash(h):
	return int(np.array(h, dtype="float64"))
def init():
	imagePaths = list(paths.list_images('faces'))
	hashes = {}
	for (i, imagePath) in enumerate(imagePaths):
		image = cv2.imread(imagePath)
		h = dhash(image)
		h = convert_hash(h)
		l = hashes.get(h, [])
		l.append(imagePath)
		hashes[h] = l
	points = list(hashes.keys())
	tree = vptree.VPTree(points, hamming)
	f = open('tree.pickle', "wb")
	f.write(pickle.dumps(tree))
	f.close()
	f = open('hashes.pickle', "wb")
	f.write(pickle.dumps(hashes))
	f.close()
def load_tree():
	tree = pickle.loads(open('tree.pickle', "rb").read())
	return tree
def load_hashes():
	hashes = pickle.loads(open('hashes.pickle', "rb").read())
	return hashes
def recognize(face, tree, hashes):
	queryHash = dhash(face)
	queryHash = convert_hash(queryHash)
	results = tree.get_all_in_range(queryHash, 50)
	results = sorted(results)
	resultPaths = None
	for (d, h) in results:
		resultPaths = hashes.get(h, [])
	results = []
	for path in resultPaths:
		results.append(path.replace('faces/', '').replace('.png', ''))
	return results
progress = Progress()
progress.start('Loading Recognizer...')
init()
tree = load_tree()
hashes = load_hashes()
progress.stop()
