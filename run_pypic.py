from __future__ import print_function
import pypic as p
import convert as c
import time
import os

def main():
	os.system('rm plots/*')

	start = 0
	stop  = 10000
	skip  = 10
	start_time = time.time()
	p.main(stop,skip)
	stop_time = time.time()
	time_out = open('plots/time.out','w+')
	print(-start_time+stop_time,file=time_out)
	#c.convert('plots','ps_i',start,stop,skip,'mov_i.gif')
	c.convert('plots','ps',start,stop,skip,'plots/mov.gif')
	c.convert('plots','e',start,stop,skip,'plots/mov_f.gif')
	c.convert('plots','summary',start,stop,skip,'plots/mov_all.gif')
#end def main

if __name__ == '__main__':
	main()
