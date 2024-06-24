from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid.axislines import SubplotZero
 
def makeT(lim=1):	# Make the X-axis
	result = linspace(-lim,lim,500)
	return result

def sinc(x):		# define normlized sinc function
	return sin(pi*x)/(pi*x)
 
def raisedCos(x,B=0,T=1):	# define raised cosine function
	return sinc(x/T) * cos(pi*B*x/T) / (1 - (4*B*B*x*x/(T*T)) )

fig = figure(figsize=(8,4))
ax = SubplotZero(fig,111)
fig.add_subplot(ax)
ax.grid(True)
ax.set_xticks([-3,-2,-1,0,1,2,3])
ax.set_xticklabels(["-3T","-2T","-T","0","T","2T","3T"])
ax.set_ylim((-.2,1.))
ax.set_yticklabels([])	
for direction in ["xzero","yzero"]:
	ax.axis[direction].set_axisline_style("->")
	ax.axis[direction].set_visible(True)
for direction in ["left","right","bottom","top"]:
	ax.axis[direction].set_visible(False)

t = makeT(4)

ax.plot(t,raisedCos(t+2),'b')
ax.plot(t,raisedCos(t+1),'b')
ax.plot(t,raisedCos(t),'b')
ax.plot(t,raisedCos(t-2),'b')
ax.plot(t,raisedCos(t-1),'b')

ax.text(4.,-.1,r"$t$")
ax.text(.2,1.01,r"$h(t)$")

#fig.show()
fig.savefig("Raised-cosine-ISI.svg",bbox_inches="tight",\
	pad_inches=.15)