import os
import subprocess

def download_checkpoint(path, name, url):
	if os.path.isfile(path+name) == False:
		print("checkpoint(model) file does not exist, now download ...")
		subprocess.run(["wget", "-P", path, url])

def main():
	print("test finish")

if __name__ == "__main__":
	main()
