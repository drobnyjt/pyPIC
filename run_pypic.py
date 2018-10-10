import PIC_L as p
import convert as c



def main():
	start = 0
	stop  = 1000
	skip  = 10
	p.main_i(stop, skip)
	#c.convert('plots','ps_i',start,stop,skip,'mov_i.gif')
	c.convert('plots','ps',start,stop,skip,'mov.gif')
	c.convert('plots','e',start,stop,skip,'mov_f.gif')
#end def main

if __name__ == '__main__':
	main()
