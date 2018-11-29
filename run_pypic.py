from __future__ import print_function
import pypic as p
import convert as c
import time
import os

def main():
	os.system('rm plots/*')

	start = 0
	stop  = 500
	skip  = 100
	start_time = time.time()
	p.implicit_pic(stop, skip)
	stop_time = time.time()
	time_out = open('plots/time.out','w+')
	print(-start_time+stop_time,file=time_out)
	#c.convert('plots','ps_i',start,stop,skip,'mov_i.gif')
	c.convert('plots','ps',start,stop,skip,'plots/mov.gif')
	c.convert('plots','e',start,stop,skip,'plots/mov_f.gif')
#end def main

if __name__ == '__main__':
	main()
