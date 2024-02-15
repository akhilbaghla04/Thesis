/*==============================================================================
 
	Replication Code for "Banking Crises without Panics":
	 Results Using Monthly Data
	
	Matthew Baron, Emil Verner, and Wei Xiong
	
	September 2020

	
================================================================================*/


clear all

* Set path here
* example:
global root "/Users/mbaron/Dropbox/BVX/QJE Final Files/replication kit"


set scheme s1color
pause on 




* timing results and event studies around BVX crises
do "$root/additional do files/create Figure 7 and Tables 3 4 and A10"
do "$root/additional do files/create Figure 6"

* timing results and event studies around Narrative Crises
do "$root/additional do files/create Tables A9 and A10"
do "$root/additional do files/create Figure A13"

* timing results around the U.S. 2008 crisis
do "$root/additional do files/create Figure 5"




