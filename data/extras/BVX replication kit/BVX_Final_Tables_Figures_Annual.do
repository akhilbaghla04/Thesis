/*==============================================================================
 
	Replication Code for "Banking Crises without Panics":
	 Results Using Annual Data
	
	Matthew Baron, Emil Verner, and Wei Xiong
	
	September 2020

	
================================================================================*/



/*------------------------------------------------------------------------

 Set your path here
 
-------------------------------------------------------------------------*/

*example:
global root "/Users/mbaron/Dropbox/BVX/QJE Final Files/replication kit"

global datapath "$root/data"
global output   "$root/output"

set more off
set scheme sol // scheme for producing nicer graphs (included in replication kit folder), can be commented out
graph set window fontface "Times New Roman" 

global L=6 // Driscoll Kraay lag length





/*------------------------------------------------------------------------

 Figure I: 
 Dynamics of Output and Credit Around Bank Equity Crashes
 
-------------------------------------------------------------------------*/

use "$datapath/BVX_annual_regdata.dta", clear

* for text
tab C_B30
tab C_B30 if rgdp_gr~=. & credit_to_gdp~=. & Rtot_nonfin_real_w~=.

keep ccode year Fd*credit_to_gdp Fd*y C_B30

local W=5
foreach X in credit_to_gdp y  {
	gen Fd0`X'=0
}
foreach X of varlist Fd*credit_to_gdp Fd*y  {
	gen L_`X' = L`W'.`X'
}

* keep only the first of successive 30% crashes
gen cum_C_B30 = 0
replace cum_C_B30 = L.cum_C_B30 + C_B30 if _n>1 & C_B30==1
drop if cum_C_B30 > 1 
tab C_B30


collapse (mean) L_Fd*credit_to_gdp L_Fd*y , by(C_B30)
drop if C_B30==.
ren L_Fd* Fd*
foreach X in credit_to_gdp y {
	ren Fd*`X' `X'*
}
reshape long y credit_to_gdp  , i(C_B30) j(time)
label val time
replace time=time-`W'
sum time
reshape wide y credit_to_gdp  , i(time) j(C_B30)

#delimit ;
twoway connected credit_to_gdp1 credit_to_gdp0 time if time<=5, 
	yline(0, lcolor(black) lpattern(solid) lwidth(thin)) xline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
	xtitle("Years after bank equity crash") ytitle("Credit-to-GDP change relative to t=-5")
	mcolor(blue red) lcolor(blue red) xlabel(#15) ylabel(#5, format(%03.2f)) 
	legend(order(1 "Bank equity crash in t=0" 2 "No crash") size(medium)) name(credit,replace);
#delimit cr
graph export "$output/Figure 1B.pdf", as(pdf) replace

#delimit ;
twoway connected y1 y0 time if time<=5, 
	yline(0, lcolor(black) lpattern(solid) lwidth(thin)) xline(0, lcolor(black) lpattern(solid) lwidth(thin))  
	xtitle("Years after bank equity crash") ytitle("Log real GDP change relative to t=-5")
	mcolor(blue red) lcolor(blue red) xlabel(#10) ylabel(#5, format(%03.2f)) 
	legend(order(1 "Bank equity crash in t=0" 2 "No crash") size(medium)) name(y,replace);
#delimit cr
graph export "$output/Figure 1A.pdf", as(pdf) replace








/*------------------------------------------------------------------------

 Figure II: 
 Bank Equity Crashes Predict Output Gaps and Credit Contractions
 
 Figure A4: 
 Bank Equity Crashes Predict Output Gaps and Credit Contractions: 
 Robustness Including Year Fixed Effects
 
 Table A3:
 Bank Equity Return Bins, Real GDP, and Credit-to-GDP
 
-------------------------------------------------------------------------*/


use "$datapath/BVX_annual_regdata.dta", clear

*** Prepare Variables
gen  R_B = Rtot_real_w
gen  R_N = Rtot_nonfin_real_w

* bins
foreach X in R_B R_N {
	gen B1_`X' = ( `X' <= -.45)              if `X'~=.
	gen B2_`X' = ( `X' >  -.45 & `X' <=-.30) if `X'~=.
	gen B3_`X' = ( `X' >  -.30 & `X' <=-.15) if `X'~=.
	gen B4_`X' = ( `X' >  -.15 & `X' <=0.0)  if `X'~=.	
	gen B5_`X' = ( `X' >  0.0 & `X' <=.15)   if `X'~=.
	gen B6_`X' = ( `X' >  0.15 & `X' <=.30)  if `X'~=.
	gen B7_`X' = ( `X' >  0.30 & `X' <=.45)  if `X'~=.
	gen B8_`X' = ( `X' >  0.45 ) if `X'~=.
}		

* lags
foreach X of varlist  R_B R_N  B?_R_B B?_R_N {
	forv i=1/3{
		gen L`i'`X' = L`i'.`X'
	}
}

* controls
foreach X of varlist g0y g0credit_to_gdp {
	forv i=0/3{
		gen L`i'`X' = L`i'.`X'
	}
}

global lags3 "L?B?_R_B L?B?_R_N"
local X0 ""
local X1 "$lags3	L0g0y L1g0y L2g0y L3g0y L0g0credit_to_gdp L1g0credit_to_gdp L2g0credit_to_gdp L3g0credit_to_gdp"

global RHS        "B1_R_B B2_R_B B3_R_B B4_R_B 	      B6_R_B B7_R_B B8_R_B  B1_R_N B2_R_N B3_R_N B4_R_N        B6_R_N B7_R_N B8_R_N"
global RHS_noomit "B1_R_B B2_R_B B3_R_B B4_R_B B5_R_B B6_R_B B7_R_B B8_R_B  B1_R_N B2_R_N B3_R_N B4_R_N B5_R_N B6_R_N B7_R_N B8_R_N"

foreach LHS in "y" "credit_to_gdp" {
	qui reghdfe Fd3`LHS'   $RHS `X1' , absorb(ccode year) cluster(ccode year)
	gen smp_`LHS' = e(sample)
}


tempfile dta_crash
save    `dta_crash'


*** Figure II: Local Projection without year FE

global H = 6
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in $RHS_noomit {
			gen b`k'`LHS'   = 0 
			gen b`k'`LHS'se = 0 
			gen b`k'`LHS'u  = 0 
			gen b`k'`LHS'l  = 0 	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash',clear
			
			reghdfe Fd`i'`LHS'   $RHS `X`j'' if smp_`LHS'==1  , absorb( ccode ) cluster(ccode year)
			
			use `EST_`LHS'_`j'', clear
			foreach k in $RHS {
				replace b`k'`LHS'   = _b[`k']  if h ==`i'
				replace b`k'`LHS'se = _se[`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h


* Credit
#delimit ;
twoway connected bB1_R_Bcredit_to_gdp bB2_R_Bcredit_to_gdp bB3_R_Bcredit_to_gdp bB4_R_Bcredit_to_gdp
				 bB6_R_Bcredit_to_gdp bB7_R_Bcredit_to_gdp bB8_R_Bcredit_to_gdp
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Bank equity returns") name(credit_R_B_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.12(0.02)0.04))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%" ) size(small) rows(2));
#delimit cr

#delimit ;
twoway connected bB1_R_Ncredit_to_gdp bB2_R_Ncredit_to_gdp bB3_R_Ncredit_to_gdp	bB4_R_Ncredit_to_gdp 
                 bB6_R_Ncredit_to_gdp bB7_R_Ncredit_to_gdp bB8_R_Ncredit_to_gdp	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Nonfin. equity returns") name(credit_R_N_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.12(0.02)0.04))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%" ) size(small) rows(2));
#delimit cr
grc1leg credit_R_B_bins credit_R_N_bins, name(cred_bins,replace) altshrink  
graph display cred_bins, ysize(5.5) xsize(9.0)
graph export "$output/Figure 2B.pdf", replace as(pdf) 


* GDP
#delimit ;
twoway connected bB1_R_By bB2_R_By bB3_R_By bB4_R_By
				 bB6_R_By bB7_R_By bB8_R_By	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Bank equity returns") name(y_R_B_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.06(0.02)0.02))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%" ) size(small) rows(2));
#delimit cr
#delimit ;
twoway connected bB1_R_Ny bB2_R_Ny bB3_R_Ny bB4_R_Ny
				 bB6_R_Ny bB7_R_Ny bB8_R_Ny	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Nonfin. equity returns") name(y_R_N_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.06(0.02)0.02))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%" ) size(small) rows(2));
#delimit cr

*net install grc1leg.pkg 
grc1leg y_R_B_bins y_R_N_bins, name(y_bins,replace) altshrink xsize(4) ysize(2)
graph display y_bins, ysize(5.5) xsize(9.0)
graph export "$output/Figure 2A.pdf", replace as(pdf) 





*** Figure A4: Local Projection ~with~ year FE

global H = 10
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" "2" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in $RHS_noomit {
			gen b`k'`LHS'   = 0 
			gen b`k'`LHS'se = 0 
			gen b`k'`LHS'u  = 0 
			gen b`k'`LHS'l  = 0 	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash',clear
			
			reghdfe Fd`i'`LHS'   $RHS `X`j'' if smp_`LHS'==1  , absorb(ccode year) cluster(ccode year)
			
			use `EST_`LHS'_`j'', clear
			foreach k in $RHS {
				replace b`k'`LHS'   = _b[`k']  if h ==`i'
				replace b`k'`LHS'se = _se[`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	append using `EST_`LHS'_2'	
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h

graph drop _all

* Credit
#delimit ;
twoway connected bB1_R_Bcredit_to_gdp bB2_R_Bcredit_to_gdp bB3_R_Bcredit_to_gdp bB4_R_Bcredit_to_gdp
				 bB6_R_Bcredit_to_gdp bB7_R_Bcredit_to_gdp	bB8_R_Bcredit_to_gdp
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Bank equity returns") name(credit_R_B_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.12(0.02)0.04))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%" ) size(small) rows(2));
#delimit cr

#delimit ;
twoway connected bB1_R_Ncredit_to_gdp bB2_R_Ncredit_to_gdp bB3_R_Ncredit_to_gdp	bB4_R_Ncredit_to_gdp 
                 bB6_R_Ncredit_to_gdp bB7_R_Ncredit_to_gdp bB8_R_Ncredit_to_gdp	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Nonfin. equity returns") name(credit_R_N_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.12(0.02)0.04))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%" ) size(small) rows(2));
#delimit cr
grc1leg credit_R_B_bins credit_R_N_bins, name(cred_bins,replace) altshrink  
graph display cred_bins, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 4B.pdf", replace as(pdf) 


* GDP
#delimit ;
twoway connected bB1_R_By bB2_R_By bB3_R_By bB4_R_By
				 bB6_R_By bB7_R_By bB8_R_By	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Bank equity returns") name(y_R_B_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.06(0.02)0.02))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%" ) size(small) rows(2));
#delimit cr
#delimit ;
twoway connected bB1_R_Ny bB2_R_Ny bB3_R_Ny bB4_R_Ny
				 bB6_R_Ny bB7_R_Ny bB8_R_Ny	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Nonfin. equity returns") name(y_R_N_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.06(0.02)0.02))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%" ) size(small) rows(2));
#delimit cr
grc1leg y_R_B_bins y_R_N_bins, name(y_bins,replace) altshrink xsize(5) ysize(2) 
graph display y_bins, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 4A.pdf", replace as(pdf) 






*** Table A3: return bins at 3-year horizon

use    `dta_crash', clear

foreach k in B N {
	label var B1_R_`k' "\(r^`k'_{it} \leq -45\% \)"
	label var B2_R_`k' "\(-45\% < r^`k'_{it} \leq -30\% \)"
	label var B3_R_`k' "\(-30\% < r^`k'_{it} \leq -15\% \)"
	label var B4_R_`k' "\(-15\% < r^`k'_{it} \leq 0\% \)"
	label var B6_R_`k' "\(15\% < r^`k'_{it} \leq 30\% \)"
	label var B7_R_`k' "\(30\% < r^`k'_{it} \leq 45\% \)"
	label var B8_R_`k' "\( r^`k'_{it} > 45\% \)"
}
	
cap est drop *

foreach LHS in y credit_to_gdp {
	
	* no controls
	eststo: reghdfe Fd3`LHS'   $RHS   if smp_`LHS'==1  , absorb( ccode ) cluster(ccode year)
	local aR2: display %5.2f e(r2_a_within)
	estadd local CFE  "\checkmark"
	estadd local RNcont "\checkmark"
	estadd local aR2 = "`aR2'" 

	* with controls
	eststo:reghdfe Fd3`LHS'   $RHS `X1' if smp_`LHS'==1  , absorb( ccode ) cluster(ccode year)
	local aR2: display %5.2f e(r2_a_within)
	estadd local CFE  "\checkmark"
	estadd local RNcont "\checkmark"
	estadd local LDV  "\checkmark"	
	estadd local aR2 = "`aR2'" 
	
	eststo: reghdfe Fd3`LHS'   $RHS `X1' if smp_`LHS'==1  , absorb( ccode year) cluster(ccode year)
	local aR2: display %5.2f e(r2_a_within)
	estadd local CFE  "\checkmark"
	estadd local RNcont "\checkmark"
	estadd local LDV  "\checkmark"	
	estadd local YFE  "\checkmark"	
	estadd local aR2 = "`aR2'" 
}

#delimit ;
esttab using "$output/Appendix Table 3.tex", 
	    replace compress b(a2) t(a2)  star(* 0.10 ** 0.05 *** 0.01 ) noconstant  brackets		
	    noobs booktabs  nonotes  nomtitles
	    mgroups("Real GDP  \(\text{growth}_{t,t+3}\)" "Credit-GDP \(\text{change}_{t,t+3}\)",
	          pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )	
	    scalars("CFE Country fixed effects" "RNcont Nonfin. eq. bins" "LDV Other controls" "YFE Year fixed effects" "aR2 Adj. \(R^2 \) (within)" "N N")
		keep(B1_R_B B2_R_B B3_R_B B4_R_B B6_R_B B7_R_B B8_R_B ) order(B1_R_B B2_R_B B3_R_B B4_R_B B6_R_B B7_R_B B8_R_B  )
		label substitute(\_ _);
#delimit cr







/*------------------------------------------------------------------------

 Table I:
 Bank Equity Crashes Predict Output Gaps and Credit Contraction
 
 Table A5: 
 Bank Equity Crashes and Subsequent GDP and Credit Growth: Subsample Analysis

 Table A4:
 Appendix Table: Bank equity continuous returns and nonlinearity
 
-------------------------------------------------------------------------*/

*** Prepare data
use "$datapath/BVX_annual_regdata.dta", clear
set matsize 5000

* bank equity 30% crash variable
gen R_B = C_B30 
gen R_N = C_N30 
gen D1d_y = g0credit_to_gdp
gen D1y   = rgdp_gr

* lags
foreach X of varlist R_B R_N D1d_y D1y {
	forv i=1/3{
		gen L`i'`X'    = L`i'.`X' 
	}
}

label var R_B "Bank equity crash"
label var R_N "Nonfinancial equity crash"

keep if R_B~=. & R_N~=.
tsfill

cap est drop *




*** Table I
* Panel A: GDP

reghdfe Fd3y R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y , absorb(ccode year) cluster(ccode year)
gen smp=e(sample)

* 1
eststo: reghdfe Fd1y R_B R_N if smp==1, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 2
eststo: reghdfe Fd1y R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y if smp==1, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 3
eststo: reghdfe Fd1y R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y if smp==1, absorb(ccode year) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local YFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 4
eststo: reghdfe Fd3y R_B R_N  if smp==1, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 5
eststo: reghdfe Fd3y R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y if smp==1, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 

* 6
eststo: reghdfe Fd3y R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y if smp==1, absorb(ccode year) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local YFE "\checkmark"
estadd local aR2 = "`aR2'" 

#delimit ;
esttab using "$output/Table 1A.tex", replace label compress
	   keep(R_B R_N) substitute(\__)
	   b(a2) t(a2) star(* 0.1 ** 0.05 *** 0.01) 
	   mgroups("Real GDP \(\text{growth}_{t,t+1}\)" "Real GDP \(\text{growth}_{t,t+3}\)",
	          pattern(1 0 0 1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
	    noobs booktabs  nonotes  nomtitles noconstant brackets
	   scalars("CFE Country fixed effects" "LDV Controls" "YFE Year fixed effects" "aR2 Adj. \( R^2 \) (within)" "N N");
#delimit cr

* Panel B: Credit-to-GDP
cap est drop *
reghdfe Fd3credit_to_gdp R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y  , absorb(ccode year) cluster(ccode year)
gen smp2= (e(sample) & Fd1credit_to_gdp!=.)

* 1 
eststo: reghdfe Fd1credit_to_gdp R_B R_N if smp2==1, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 2
eststo: reghdfe Fd1credit_to_gdp R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y  if smp2==1, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 3
eststo: reghdfe Fd1credit_to_gdp R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y  if smp2==1, absorb(ccode year) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local YFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 4
eststo: reghdfe Fd3credit_to_gdp R_B R_N  if smp2==1, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 5
eststo: reghdfe Fd3credit_to_gdp R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y  if smp2==1, absorb(ccode ) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 6
eststo: reghdfe Fd3credit_to_gdp R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y  if smp2==1, absorb(ccode year) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local YFE "\checkmark"
estadd local aR2 = "`aR2'" 


#delimit ;
esttab using "$output/Table 1B.tex", replace label compress
	   keep(R_B R_N) substitute(\__)
	   b(a2) t(a2) star(* 0.1 ** 0.05 *** 0.01) 
	   mgroups("Credit-to-GDP \(\text{change}_{t,t+1}\)" "Credit-to-GDP \(\text{change}_{t,t+3}\)",
	          pattern(1 0 0 1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
	    noobs booktabs  nonotes  nomtitles noconstant brackets
	   scalars("CFE Country fixed effects" "LDV Controls" "YFE Year fixed effects" "aR2 Adj. \( R^2 \) (within)" "N N");
#delimit cr





*** Table A5: Robustness of bank equity crash - Subsamples

* GDP growth from t to t+3
cap est drop *
* 1
eststo: reghdfe Fd3y R_B R_N  if smp==1 & year<1939, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 2
eststo: reghdfe Fd3y R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y if smp==1 & year<1939, absorb(ccode ) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 3
eststo: reghdfe Fd3y R_B R_N  if smp==1 & inrange(year,1946,1970), absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 4
eststo: reghdfe Fd3y R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y if smp==1 & inrange(year,1946,1970), absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 5
eststo: reghdfe Fd3y R_B R_N  if smp==1 & year>=1971, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 6
eststo: reghdfe Fd3y R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y if smp==1 & year>=1971, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 

#delimit ;
esttab using "$output/Appendix Table 5A.tex", replace label compress
	   keep(R_B R_N) substitute(\__)
	   b(a2) t(a2) star(* 0.1 ** 0.05 *** 0.01) 
	   mgroups("Pre-1939" "1946-1970" "1971-2016",
	          pattern(1 0 1 0 1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
	    noobs booktabs  nonotes  nomtitles noconstant brackets
	   scalars("CFE Country fixed effects" "LDV Controls" "aR2 Adj. \( R^2 \) (within)" "N N");
#delimit cr

* Credit-GDP change from t to t+3
cap est drop *
* 1
eststo: reghdfe Fd3credit_to_gdp R_B R_N  if smp2==1 & year<1939, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 2
eststo: reghdfe Fd3credit_to_gdp R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y  if smp2==1 & year<1939, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local YFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 3
eststo: reghdfe Fd3credit_to_gdp R_B R_N  if smp2==1 & inrange(year,1946,1970), absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 4
eststo: reghdfe Fd3credit_to_gdp R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y  if smp2==1 & inrange(year,1946,1970), absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local YFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 5
eststo: reghdfe Fd3credit_to_gdp R_B R_N  if smp2==1 & year>=1971, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local CFE "\checkmark"
estadd local aR2 = "`aR2'" 
* 6
eststo: reghdfe Fd3credit_to_gdp R_B R_N L?R_B L?R_N D1y D1d_y L?D1y L?D1d_y  if smp2==1 & year>=1971, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local LDV "\checkmark"
estadd local CFE "\checkmark"
estadd local YFE "\checkmark"
estadd local aR2 = "`aR2'" 


#delimit ;
esttab using "$output/Appendix Table 5B.tex", replace label compress
	   keep(R_B R_N) substitute(\__)
	   b(a2) t(a2) star(* 0.1 ** 0.05 *** 0.01) 
	   mgroups("Pre-1939" "1946-1970" "1971-2016",
	          pattern(1 0 1 0 1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
	    noobs booktabs  nonotes  nomtitles noconstant brackets
	   scalars("CFE Country fixed effects" "LDV Controls" "aR2 Adj. \( R^2 \) (within)" "N N");
#delimit cr





*** Table A4: Bank Equity Returns, Output, and Credit: Alternative Specifications

drop R_B R_N L?R_B L?R_N

gen  R_B = Rtot_real_w         // use continuous returns
gen  R_N = Rtot_nonfin_real_w

foreach X in R_B R_N {
	gen `X'pos = `X'*(`X'>=0) // positive vs negative returns
	gen `X'neg = `X'*(`X'<0) 
	forv i=2/2{
		gen `X'`i' = (`X')^(`i') // quadratic
	}	
}


foreach X of varlist R_B R_N R_B2 R_N2 R_Bpos R_Bneg R_Npos R_Nneg  {
	forv i=1/3{
		gen L`i'`X'=L`i'.`X'
	}
}

label var R_B "Bank eq. ret."
label var R_B2 "\( (\text{Bank eq. ret.})^2 \)"
label var R_N "Nonfin. eq. ret."
label var R_N2 "\( (\text{Nonfin. eq. ret.})^2 \)"
label var R_Bpos "Positive bank eq. ret."
label var R_Bneg "Negative bank eq. ret."
label var R_Npos "Positive nonfin. eq. ret."
label var R_Nneg "Negative nonfin. eq. ret."

global XX      "L1R_B L2R_B L3R_B L1R_N L2R_N L3R_N D1y L1D1y L2D1y L3D1y D1d_y L1D1d_y L2D1d_y L3D1d_y"
global Xquad   "L1R_B L2R_B L3R_B L1R_B2 L2R_B2 L3R_B2 L1R_N L2R_N L3R_N L1R_N2 L2R_N2 L3R_N2 D1y L1D1y L2D1y L3D1y D1d_y L1D1d_y L2D1d_y L3D1d_y"
global Xposneg "L1R_Bpos L2R_Bpos L3R_Bpos L1R_Bneg L2R_Bneg L3R_Bneg L1R_Npos L2R_Npos L3R_Npos L1R_Nneg L2R_Nneg L3R_Nneg D1y L1D1y L2D1y L3D1y D1d_y L1D1d_y L2D1d_y L3D1d_y"

cap est drop *

* 1 Linear
eststo:     reghdfe Fd3y R_B  R_N  $XX if smp==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local CFE "\checkmark"
estadd local LDV "\checkmark"
estadd local aR2 = "`aR2'" 
* 2 Quadratic
eststo:     reghdfe Fd3y R_B R_B2 R_N R_N2 $Xquad if smp==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local CFE "\checkmark"
estadd local LDV "\checkmark"
estadd local aR2 = "`aR2'" 
* 3 Positive vs Negative
eststo:     reghdfe Fd3y R_Bpos R_Bneg R_Npos R_Nneg $Xposneg if smp==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local CFE "\checkmark"
estadd local LDV "\checkmark"
estadd local aR2 = "`aR2'" 


* Credit-GDP change from t to t+3
* 1 Linear
eststo:     reghdfe Fd3credit_to_gdp R_B  R_N  $XX if smp==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local CFE "\checkmark"
estadd local LDV "\checkmark"
estadd local aR2 = "`aR2'" 
* 2 Quadratic
eststo:     reghdfe Fd3credit_to_gdp R_B R_B2 R_N R_N2 $Xquad if smp==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local CFE "\checkmark"
estadd local LDV "\checkmark"
estadd local aR2 = "`aR2'" 
* 3 Positive vs Negative
eststo:     reghdfe Fd3credit_to_gdp R_Bpos R_Bneg R_Npos R_Nneg $Xposneg if smp==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within) 
estadd local CFE "\checkmark"
estadd local LDV "\checkmark"
estadd local aR2 = "`aR2'" 


#delimit ;
esttab using "$output/Appendix Table 4.tex", replace label compress
	   keep(R_B R_B2 R_N R_N2 R_Bpos R_Bneg R_Npos R_Nneg) substitute(\__)
	   order(R_B R_B2 R_N R_N2 R_Bpos R_Bneg R_Npos R_Nneg)
	   b(a2) t(a2) star(* 0.1 ** 0.05 *** 0.01) 
	   mgroups("Real GDP \(\text{growth}_{t,t+3}\) "
			   "Credit-to-GDP \(\text{change}_{t,t+3}\) " ,
	          pattern(1 0 0 1 0 0 ) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
	    noobs booktabs  nonotes  nomtitles noconstant brackets
	   scalars("CFE Country fixed effects" "LDV Controls" "aR2 Adj. \( R^2 \) (within)" "N N");
#delimit cr











/*------------------------------------------------------------------------

 Figure III:
 Banking Distress With and Without Banking Panics
	
 Table II: 
 Impact of Banking Distress With and Without Panics

-------------------------------------------------------------------------*/



***** Panel (A): Baseline *****

use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill

*** Prepare Data
gen R_B   = C_B30             // bank equity crash
gen R_B_P = C_B30*PANIC_f     // use "filled down" panic indicator (for consecutive BE crashes associated with a panic)
gen R_P   = PANIC_f           // panic
gen R_N   = C_N30
gen g0d_y = g0credit_to_gdp
foreach X of varlist R_B R_N R_B_P g0y g0d_y R_P {
	forv i=0/3{
		gen L`i'`X'=L`i'.`X'
	}
}

*** controls
local X0 ""
local X1 "L(1/3).R_B_P   L(1/3).R_B   L(1/3).R_P  L(1/3).R_N  L(0/3).g0y   L(0/3).g0d_y  "
foreach LHS in "y" "credit_to_gdp" {
	reghdfe Fd3`LHS'   R_B R_P R_B_P R_N   `X1' , absorb(ccode) cluster(ccode year)   
	gen smp_`LHS'=e(sample)
}

* for text
tab R_B if smp_y==1
tab R_B if smp_y==1 & R_B_P==0
tab R_B if smp_y==1 & PANIC_f==0
tab R_B if smp_y==1 & PANIC_f==1
tab R_B_P if smp_y==1
tab R_P if smp_y==1 & R_B==0 
tab R_P if R_B==0 
tab R_P if bankeqdecline~=1 
tab R_B_P
tab R_P

tempfile  dta_PANIC
save     `dta_PANIC'

global H = 6
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N" "P" "B_P" {
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
			gen bBOTH`LHS' = 0 if h==0
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_PANIC',clear
			
			reghdfe Fd`i'`LHS'   R_B R_P R_B_P R_N `X`j''  if smp_`LHS'==1, absorb(ccode) cluster(ccode year)  

			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" "P" "B_P"{
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
				replace bBOTH`LHS' = bB`LHS' + bP`LHS' + bB_P`LHS' 
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h


#delimit ;
* Output Response;
twoway connected bBy bByu bByl bPy bBOTHy 
				 h if j==1 & h<=6, color(blue blue blue red dkgreen) lpattern(solid dot dot solid solid) m(s i i c t) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP Response") name(gdp,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 legend(order(1 "Bank equity crash, no panic" 4 "Panic, no bank equity crash" 5 "Bank equity crash and panic"));
* Credit to GDP Response;				 
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bPcredit_to_gdp bBOTHcredit_to_gdp 
				 h if j==1 & h<=6, color(blue blue blue red dkgreen) lpattern(solid dot dot solid solid) m(s i i c t) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP Response") name(credit,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 legend(order(1 "Bank equity crash, no panic" 4 "Panic, no bank equity crash" 5 "Bank equity crash and panic"));
#delimit cr

grc1leg gdp credit, name(baseline,replace) altshrink xsize(5) ysize(2) 
graph display baseline, ysize(5.5) xsize(9.0)
graph export "$output/Figure 3A.pdf", replace as(pdf) 



***** Panel (B): Conditioning on bank failures *****

use `dta_PANIC', clear
ren R_B R_B_old
ren R_P R_P_old
ren R_B_P R_B_P_old
gen R_B   = 0 if Rtot_real!=.
gen R_B_P = 0 if Rtot_real!=.
gen R_P   = 0 if Rtot_real!=.
replace R_B   = 1 if C_B30==1 & bankfailure_narrative_f==1
replace R_B_P = 1 if C_B30==1 & bankfailure_narrative_f==1 & PANIC_f==1
replace R_P   = 1 if PANIC_f==1 

* for text
tab R_B   if smp_y==1 & R_P==0
tab R_B_P if smp_y==1
tab R_P   if smp_y==1 & R_B==0 


local X1 "L(1/3).R_B_P   L(1/3).R_B   L(1/3).R_P  L(1/3).R_N  L(0/3).g0y   L(0/3).g0d_y  "

tempfile  dta_PANIC_bankfailures
save     `dta_PANIC_bankfailures'

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N" "P" "B_P" {
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
			gen bBOTH`LHS' = 0 if h==0
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_PANIC_bankfailures',clear
			
			  reghdfe Fd`i'`LHS'   R_B R_P R_B_P R_N `X`j''  if smp_`LHS'==1, absorb(ccode) cluster(ccode year)  			
			  
			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" "P" "B_P"{
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
				replace bBOTH`LHS' = bB`LHS' + bP`LHS' + bB_P`LHS' 
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h

#delimit ;
* Output Response;
twoway connected bBy bByu bByl bPy bBOTHy 
				 h if j==1 & h<=6, color(blue blue blue red dkgreen) lpattern(solid dot dot solid solid) m(s i i c t) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP Response") name(gdp_bankfailures,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 legend(order(1 "Bank eq. crash and failures, no panic" 4 "Panic, no bank failures" 5 "Bank eq. crash and failures, panic"));
* Credit to GDP Response;				 
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bPcredit_to_gdp bBOTHcredit_to_gdp 
				 h if j==1 & h<=6, color(blue blue blue red dkgreen) lpattern(solid dot dot solid solid) m(s i i c t) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP Response") name(credit_bankfailures,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 legend(order(1 "Bank eq. crash and failures, no panic" 4 "Panic, no bank failures" 5 "Bank eq. crash and failures, panic"));
#delimit cr

grc1leg gdp_bankfailures credit_bankfailures, name(bankfailures,replace) altshrink xsize(5) ysize(2) 
graph display bankfailures, ysize(5.5) xsize(9.0)
graph export "$output/Figure 3B.pdf", replace as(pdf)







***** Table II: Impact of Banking Distress With and Without Panics *****

use `dta_PANIC', clear

gen     R_B_2  = 0 if Rtot_real!=.  // conditioning on bank failures variables
gen     R_P_2  = 0 if Rtot_real!=.
gen    R_B_P_2 = 0 if Rtot_real!=.
replace R_B_2   = 1 if C_B30==1 & bankfailure_narrative_f==1
replace R_P_2   = 1 if PANIC_f==1 
replace R_B_P_2 = 1 if C_B30==1 & bankfailure_narrative_f==1 & PANIC_f==1

label var R_B   "Bank equity crash"
label var R_P   "Panic"
label var R_B_P "Bank equity crash \( \times \) Panic"

label var R_B_2   "Bank eq. crash and failures"
label var R_P_2   "Panic"
label var R_B_P_2 "Bank eq. crash and failures \( \times \) Panic"
label var R_N     "Nonfinancial equity crash"

tab year, gen(year_)
drop L0R_B L0R_P L0R_B_P  L0R_N

foreach X in R_B_2 R_P_2 R_B_P_2 {
	forv i=1/3{ 
		gen L`i'`X'=L`i'.`X'
	}
}

cap est drop *
global X1   "L?R_B   L?R_P   L?R_B_P   L?R_N L?g0y L?g0d_y "
global X1_2 "L?R_B_2 L?R_P_2 L?R_B_P_2 L?R_N L?g0y L?g0d_y "


** (A) Baseline
* 1 
eststo: reghdfe Fd3y R_B R_P R_B_P R_N if smp_y==1 , cluster(ccode year) absorb(ccode )
local aR2: display %5.2f e(r2_a_within)
estadd local CFE  "\checkmark"
estadd local aR2 = "`aR2'" 
* 2
eststo: reghdfe Fd3y R_B R_P R_B_P R_N $X1 if smp_y==1 , cluster(ccode year) absorb(ccode)
local aR2: display %5.2f e(r2_a_within)
estadd local CFE  "\checkmark"
estadd local cont "\checkmark"
estadd local aR2 = "`aR2'" 
	lincom R_B+R_P+R_B_P
* 3
eststo: reghdfe Fd3y R_B R_P R_B_P R_N $X1 if smp_y==1 , cluster(ccode year) absorb(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local CFE  "\checkmark"
estadd local cont "\checkmark"
estadd local YFE  "\checkmark"
estadd local aR2 = "`aR2'" 
* 4
eststo: reghdfe Fd3credit_to_gdp R_B R_P R_B_P R_N if smp_credit_to_gdp==1 , cluster(ccode year) absorb(ccode )
local aR2: display %5.2f e(r2_a_within)
estadd local CFE  "\checkmark"
estadd local aR2 = "`aR2'" 
* 5
eststo: reghdfe Fd3credit_to_gdp R_B R_P R_B_P R_N $X1 if smp_credit_to_gdp==1 , cluster(ccode year) absorb(ccode)
local aR2: display %5.2f e(r2_a_within)
estadd local CFE  "\checkmark"
estadd local cont "\checkmark"
estadd local aR2 = "`aR2'" 
		lincom R_B+R_P+R_B_P
* 6
eststo: reghdfe Fd3credit_to_gdp R_B R_P R_B_P R_N $X1 if smp_credit_to_gdp==1 , cluster(ccode year) absorb(ccode year)
local aR2: display %5.2f e(r2_a_within)
estadd local CFE  "\checkmark"
estadd local cont "\checkmark"
estadd local YFE  "\checkmark"
estadd local aR2 = "`aR2'" 

#delimit ;
esttab using "$output/Table 2A.tex", 
	    replace compress b(a2) t(a2)  star(* 0.10 ** 0.05 *** 0.01 ) noconstant  brackets		
	    noobs booktabs  nonotes  nomtitles
	   mgroups("Real GDP  \(\text{growth}_{t,t+3}\)" "Credit-GDP \(\text{change}_{t,t+3}\)",
	          pattern(1 0 0 1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )	
	    scalars("CFE Country fixed effects" "cont Controls" "YFE Year fixed effects" "aR2 Adj. \(R^2 \) (within)" "N N")
		keep(R_B R_P R_B_P R_N) order(R_B R_P R_B_P R_N)
		label substitute(\_ _);
#delimit cr

** (B) Conditioning on bank failures
cap est drop *
* 1
eststo: reghdfe Fd3y R_B_2 R_P_2 R_B_P_2 R_N if smp_y==1 , cluster(ccode year) absorb(ccode)
	local aR2: display %5.2f e(r2_a_within)
	estadd local CFE  "\checkmark"
	estadd local aR2 = "`aR2'" 
* 2	
eststo: reghdfe Fd3y R_B_2 R_P_2 R_B_P_2 R_N $X1_2 if smp_y==1 , cluster(ccode year) absorb(ccode)
	local aR2: display %5.2f e(r2_a_within)
	estadd local CFE  "\checkmark"
	estadd local cont "\checkmark"
	estadd local aR2 = "`aR2'" 
	lincom R_B_2 + R_P_2 + R_B_P_2
* 3
eststo: reghdfe Fd3y R_B_2 R_P_2 R_B_P_2 R_N $X1_2 if smp_y==1 , cluster(ccode year) absorb(ccode year)
	local aR2: display %5.2f e(r2_a_within)
	estadd local CFE  "\checkmark"
	estadd local cont "\checkmark"
	estadd local YFE  "\checkmark"
	estadd local aR2 = "`aR2'" 
	local aR2: display %5.2f e(r2_a_within)
* 4
eststo: reghdfe Fd3credit_to_gdp R_B_2 R_P_2 R_B_P_2 R_N if smp_credit_to_gdp==1 , cluster(ccode year) absorb(ccode year)
	local aR2: display %5.2f e(r2_a_within)
	estadd local CFE  "\checkmark"
	estadd local aR2 = "`aR2'" 
	local aR2: display %5.2f e(r2_a_within)
* 5
eststo: reghdfe Fd3credit_to_gdp R_B_2 R_P_2 R_B_P_2 R_N $X1_2 if smp_credit_to_gdp==1 , cluster(ccode year) absorb(ccode)
	local aR2: display %5.2f e(r2_a_within)
	estadd local CFE  "\checkmark"
	estadd local cont "\checkmark"
	estadd local aR2 = "`aR2'" 
	local aR2: display %5.2f e(r2_a_within)
	lincom R_B_2 +R_P_2+ R_B_P_2
* 6
eststo: reghdfe Fd3credit_to_gdp R_B_2 R_P_2 R_B_P_2 R_N $X1_2 if smp_credit_to_gdp==1 , cluster(ccode year) absorb(ccode year)
	local aR2: display %5.2f e(r2_a_within)
	estadd local CFE  "\checkmark"
	estadd local cont "\checkmark"
	estadd local YFE  "\checkmark"
	estadd local aR2 = "`aR2'" 


#delimit ;
esttab using "$output/Table 2B.tex", 
	    replace compress b(a2) t(a2)  star(* 0.10 ** 0.05 *** 0.01 ) noconstant  brackets		
	    noobs booktabs  nonotes  nomtitles
	   mgroups("Real GDP \(\text{growth}_{t,t+3}\)" "Credit-GDP \(\text{change}_{t,t+3}\)",
	          pattern(1 0 0 1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )	
	    scalars("CFE Country fixed effects" "cont Controls" "YFE Year fixed effects" "aR2 Adj. \(R^2 \) (within)" "N N")
		keep(R_B_2 R_P_2 R_B_P_2 R_N) order( R_B_2 R_P_2 R_B_P_2 R_N)
		label substitute(\_ _);
#delimit cr







 
/*------------------------------------------------------------------------

 Figure IV: 
 Impact of Bank Equity Crashes Outside of Episodes 
 with Either a Panic or Widespread Bank Failures
 
-------------------------------------------------------------------------*/

use "$datapath/BVX_annual_regdata.dta", clear

*** Prepare Variables
gen  R_B = Rtot_real_w
gen  R_N = Rtot_nonfin_real_w

foreach X in R_B R_N {
	gen B1_`X' = ( `X' <= -.45)              if `X'~=.
	gen B2_`X' = ( `X' >  -.45 & `X' <=-.30) if `X'~=.
	gen B3_`X' = ( `X' >  -.30 & `X' <=-.15) if `X'~=.
	gen B4_`X' = ( `X' >  -.15 & `X' <=0.0)  if `X'~=.	
	gen B5_`X' = ( `X' >  0.0 & `X' <=.15)   if `X'~=.
	gen B6_`X' = ( `X' >  0.15 & `X' <=.30)  if `X'~=.
	gen B7_`X' = ( `X' >  0.30 & `X' <=.45)  if `X'~=.
	gen B8_`X' = ( `X' >  0.45 ) if `X'~=.
}
		
foreach X of varlist B?_R_B B?_R_N {
	forv i=1/3{
		gen L`i'`X' = L`i'.`X'
	}
}
foreach X of varlist g0y g0credit_to_gdp {
	forv i=0/3{
		gen L`i'`X' = L`i'.`X'
	}
}

drop if RC==1 | L.RC==1 | L2.RC==1 | L3.RC==1 | F.RC==1 | F2.RC==1 | F3.RC==1 // drop BVX crisis episodes (episodes with panics or widespread bank failures associated with BE declines)

global lags3 "L?B?_R_B L?B?_R_N"
local X0 ""
local X1 "$lags3	L0g0y L1g0y L2g0y L3g0y  L0g0credit_to_gdp L1g0credit_to_gdp L2g0credit_to_gdp L3g0credit_to_gdp"

global RHS        "B1_R_B B2_R_B B3_R_B B4_R_B 	      B6_R_B B7_R_B B8_R_B  B1_R_N B2_R_N B3_R_N B4_R_N        B6_R_N B7_R_N B8_R_N"
global RHS_noomit "B1_R_B B2_R_B B3_R_B B4_R_B B5_R_B B6_R_B B7_R_B B8_R_B  B1_R_N B2_R_N B3_R_N B4_R_N B5_R_N B6_R_N B7_R_N B8_R_N"

foreach LHS in "y" "credit_to_gdp" {
	qui reghdfe Fd3`LHS'   $RHS `X1' , absorb(ccode year) cluster(ccode year)
	gen smp_`LHS' = e(sample)
}
tempfile dta_crash
save    `dta_crash'

global H = 10
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in $RHS_noomit {
			gen b`k'`LHS'   = 0 
			gen b`k'`LHS'se = 0 
			gen b`k'`LHS'u  = 0 
			gen b`k'`LHS'l  = 0 	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash', clear
			
			reghdfe Fd`i'`LHS'   $RHS `X`j'' if smp_`LHS'==1  , absorb(ccode) cluster(ccode year)
			
			use `EST_`LHS'_`j'', clear
			foreach k in $RHS {
				replace b`k'`LHS'   = _b[`k']  if h ==`i'
				replace b`k'`LHS'se = _se[`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h


* Credit
#delimit ;
twoway connected bB1_R_Bcredit_to_gdp bB2_R_Bcredit_to_gdp bB3_R_Bcredit_to_gdp bB4_R_Bcredit_to_gdp
				 bB6_R_Bcredit_to_gdp bB7_R_Bcredit_to_gdp	bB8_R_Bcredit_to_gdp
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 				 
				 title("Bank equity returns") name(credit_R_B_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.06(0.02)0.06))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%") rows(2));
#delimit cr

#delimit ;
twoway connected bB1_R_Ncredit_to_gdp bB2_R_Ncredit_to_gdp bB3_R_Ncredit_to_gdp	bB4_R_Ncredit_to_gdp 
                 bB6_R_Ncredit_to_gdp bB7_R_Ncredit_to_gdp bB8_R_Ncredit_to_gdp	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 				 				 
				 title("Nonfin. equity returns") name(credit_R_N_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.06(0.02)0.06))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%") rows(2));
#delimit cr
grc1leg credit_R_B_bins credit_R_N_bins, name(cred_bins,replace) altshrink 
graph display cred_bins, ysize(5.5) xsize(9.0)
graph export "$output/Figure 4B.pdf", replace as(pdf) 


* GDP
#delimit ;
twoway connected bB1_R_By bB2_R_By bB3_R_By bB4_R_By
				 bB6_R_By bB7_R_By bB8_R_By	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 				 				 
				 title("Bank equity returns") name(y_R_B_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.06(0.02)0.02))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%") rows(2));
#delimit cr
#delimit ;
twoway connected bB1_R_Ny bB2_R_Ny bB3_R_Ny bB4_R_Ny
				 bB6_R_Ny bB7_R_Ny bB8_R_Ny	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 				 				 
				 title("Nonfin. equity returns") name(y_R_N_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.06(0.02)0.02))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%") rows(2));
#delimit cr
grc1leg y_R_B_bins y_R_N_bins, name(y_bins,replace) altshrink 
graph display y_bins, ysize(5.5) xsize(9.0)
graph export "$output/Figure 4A.pdf", replace as(pdf) 







/*------------------------------------------------------------------------

 Figure A2: 
 Bank Equity Returns Provide the Best Real Time Signal 
 of Narrative Banking Crises: ROC Analysis
 
-------------------------------------------------------------------------*/

use "$datapath/BVX_annual_regdata.dta", clear

* merge in broad market index, annual credit spreads, and unemployment just for this analysis
merge 1:1 country ccode year using "$root/data/additional data/broad equity market indexes.dta", nogen
merge 1:1 country ccode year using "$root/data/additional data/additional macro data.dta", nogen

xtset ccode year
keep if Rtot_real ~=. & JC~=.

foreach X of varlist Rtot_real_w Rtot_broad_real_w Rtot_nonfin_real_w rgdp_gr {
	gen `X'_neg = - `X'
}
foreach X of varlist unemployment {
	gen D1`X' = `X' - L.`X'
}
ren Rtot_real_w_neg R_B
ren Rtot_nonfin_real_w_neg R_N
ren Rtot_broad_real_w_neg R_M
ren corp_credit_spread c_spr
ren bank_credit_spread b_spr
gen Fd5d_y = -(Fd5credit_to_gdp)
ren D1unemployment D1u
ren rgdp_gr_neg D1y

tempfile tmp
save    `tmp'


* Panel: Macro variables
keep if R_B~=. & Fd5d_y~=. & D1u~=. & D1y~=.

foreach X of varlist  R_B Fd5d_y D1u D1y  {
	rocreg JC `X', nobootstrap
	dis `auc_`X''
	ren _roc_`X' roc_`X'
	ren _fpr_`X' fpr_`X'
}


#delimit ;
twoway scatter roc_R_B fpr_R_B , connect(J) sort m(i) lcolor(blue)
	|| scatter roc_D1u fpr_D1u , connect(J) sort m(i) lcolor(red)	
	|| scatter roc_D1y fpr_D1y , connect(J) sort m(i) lcolor(dkgreen)
	|| scatter roc_Fd5d_y fpr_Fd5d_y , connect(J) sort m(i) 
	|| scatter roc_R_B roc_R_B ,  connect(l) sort m(i) lcolor(black) lpattern(solid)
	xtitle("False positive rate") ytitle("True positive rate")
	yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
	legend(order(1 "Bank equity return" 2 "Unemp. rate change" 3 "GDP growth" 4 "Credit contraction (t,t+5)"))
	name(roc_macro,replace);
#delimit cr
graph export "$output/Appendix Figure 2C.pdf", replace as(pdf)

* Panel: Financial Variables 1
use `tmp', clear
gen R_B_R_N = R_B - R_N
keep if R_B~=. & R_M~=. & R_N~=. 
foreach X of varlist  R_B R_N R_M R_B_R_N  {
	rocreg JC `X', nobootstrap
	ren _roc_`X' roc_`X'
	ren _fpr_`X' fpr_`X'
}

#delimit ;
twoway scatter roc_R_B fpr_R_B , connect(J) sort m(i) lcolor(blue)
    || scatter roc_R_N fpr_R_N , connect(J) sort m(i) lcolor(red)
	|| scatter roc_R_M fpr_R_M , connect(J) sort m(i) lcolor(dkgreen)
	|| scatter roc_R_B_R_N fpr_R_B_R_N , connect(J) sort m(i) lcolor(teal)	
	|| scatter roc_R_B roc_R_B ,  connect(l) sort m(i) lcolor(black) lpattern(solid)
	xtitle("False positive rate") ytitle("True positive rate")
	yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
	legend(order(1 "Bank equity return" 2 "Nonfin. equity return" 3 "Market return" 4 "Banks minus nonfin. return" ))
	name(roc_fin,replace);
#delimit cr
graph export "$output/Appendix Figure 2A.pdf", replace as(pdf)


* Panel: Financial Variables 2
use `tmp', clear
keep if R_B~=. & c_spr~=. & b_spr~=.
foreach X of varlist  R_B  b_spr c_spr  {
	rocreg JC `X', nobootstrap
	ren _roc_`X' roc_`X'
	ren _fpr_`X' fpr_`X'
}
#delimit ;
twoway scatter roc_R_B fpr_R_B , connect(J) sort m(i) lcolor(blue)
	|| scatter roc_c_spr fpr_c_spr , connect(J) sort m(i) lcolor(black)
	|| scatter roc_b_spr fpr_b_spr , connect(J) sort m(i) lcolor(orange_red)
	|| scatter fpr_R_B fpr_R_B ,  connect(l) sort m(i) lcolor(black) lpattern(solid)
	xtitle("False positive rate") ytitle("True positive rate")
	yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
	legend(order(1 "Bank equity return" 2 "Corp. credit spread" 3 "Bank credit spread"))
	name(roc_fin_spr,replace);
#delimit cr
graph export "$output/Appendix Figure 2B.pdf", replace as(pdf)






/*------------------------------------------------------------------------
 
 Figure A3: 
 Distribution of Bank and Nonfinancial Equity Returns:

-------------------------------------------------------------------------*/

use "$datapath/BVX_annual_regdata.dta", clear

gen CRISIS = JC
gen  R_B = Rtot_real_w
gen  R_N = Rtot_nonfin_real_w

keep if R_B~=. & R_N~=.

twoway (histogram R_B if CRISIS==1, color(blue) width(0.05) )  ///
       (histogram R_B if CRISIS==0,  width(0.05) ///
	   fcolor(none) lcolor(black)), legend(order(1 "Narrative Crisis in t" 2 "No crisis" )) ///
	   title("Bank equity return") xtitle("") name(R_B_crisis,replace)


twoway (histogram R_N if CRISIS==1, color(blue) width(0.05) )  ///
       (histogram R_N if CRISIS==0,  width(0.05) ///
	   fcolor(none) lcolor(black)), legend(order(1 "Narrative Crisis in t" 2 "No crisis" )) ///
	   title("Nonfinancial equity return") xtitle("") name(R_N_crisis,replace)

graph combine R_B_crisis R_N_crisis, rows(2) xcommon ycommon xsize(4) ysize(6)
graph export "$output/Appendix Figure 3.pdf", replace as(pdf) 








/*------------------------------------------------------------------------

 Figure A5: 
 Bank Equity Crashes Predict Output Gaps and Credit Contraction: 
 Alternative Timing
 
-------------------------------------------------------------------------*/

use "$datapath/BVX_annual_regdata.dta", clear

gen  R_B = Rtot_real_w
gen  R_N = Rtot_nonfin_real_w

foreach X in R_B R_N {
	gen B1_`X' = ( `X' <= -.45)              if `X'~=.
	gen B2_`X' = ( `X' >  -.45 & `X' <=-.30) if `X'~=.
	gen B3_`X' = ( `X' >  -.30 & `X' <=-.15) if `X'~=.
	gen B4_`X' = ( `X' >  -.15 & `X' <=0.0)  if `X'~=.	
	gen B5_`X' = ( `X' >  0.0 & `X' <=.15)   if `X'~=.
	gen B6_`X' = ( `X' >  0.15 & `X' <=.30)  if `X'~=.
	gen B7_`X' = ( `X' >  0.30 & `X' <=.45)  if `X'~=.
	gen B8_`X' = ( `X' >  0.45 ) if `X'~=.
}
	

foreach X of varlist B?_R_B B?_R_N {
	forv i=1/3{
		gen L`i'`X' = L`i'.`X'
	}
}
foreach X of varlist g0y g0credit_to_gdp {
	forv i=0/3{
		gen L`i'`X' = L`i'.`X'
	}
}

global lags3 "L?B?_R_B L?B?_R_N"
local X0 ""
local X1 "$lags3	L(1/3).g0y  L(1/3).g0credit_to_gdp "

global RHS        "B1_R_B B2_R_B B3_R_B B4_R_B 	      B6_R_B B7_R_B B8_R_B  B1_R_N B2_R_N B3_R_N B4_R_N        B6_R_N B7_R_N B8_R_N"
global RHS_noomit "B1_R_B B2_R_B B3_R_B B4_R_B B5_R_B B6_R_B B7_R_B B8_R_B  B1_R_N B2_R_N B3_R_N B4_R_N B5_R_N B6_R_N B7_R_N B8_R_N"

foreach LHS in "y" "credit_to_gdp" {
	qui reghdfe g3`LHS'   $RHS `X1' , absorb(ccode) cluster(ccode year)
	gen smp_`LHS' = e(sample)
}
tempfile dta_crash
save    `dta_crash'

global H = 6
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j' 
		gen h  = _n - 1 - 1
		foreach k in $RHS_noomit {
			gen b`k'`LHS'   = 0 
			gen b`k'`LHS'se = 0 
			gen b`k'`LHS'u  = 0 
			gen b`k'`LHS'l  = 0 	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=0/$H{
			use `dta_crash',clear
			
			* LHS: t-1 to t+i change
			reghdfe g`i'`LHS'   $RHS `X`j'' if smp_`LHS'==1  , absorb(ccode) cluster(ccode year)
			
			use `EST_`LHS'_`j'', clear
			foreach k in $RHS {
				replace b`k'`LHS'   = _b[`k']  if h ==`i'
				replace b`k'`LHS'se = _se[`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h


* Credit
#delimit ;
twoway connected bB1_R_Bcredit_to_gdp bB2_R_Bcredit_to_gdp bB3_R_Bcredit_to_gdp bB4_R_Bcredit_to_gdp
				 bB6_R_Bcredit_to_gdp bB7_R_Bcredit_to_gdp	bB8_R_Bcredit_to_gdp
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 				 				 
				 title("Bank equity returns") name(credit_R_B_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.12(0.02)0.04)) xlabel(#7) 
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%") rows(2));
#delimit cr

#delimit ;
twoway connected bB1_R_Ncredit_to_gdp bB2_R_Ncredit_to_gdp bB3_R_Ncredit_to_gdp	bB4_R_Ncredit_to_gdp 
                 bB6_R_Ncredit_to_gdp bB7_R_Ncredit_to_gdp bB8_R_Ncredit_to_gdp	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 				 				 
				 title("Nonfin. equity returns") name(credit_R_N_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.12(0.02)0.04)) xlabel(#7) 
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%") rows(2));
#delimit cr
grc1leg credit_R_B_bins credit_R_N_bins, name(cred_bins,replace) altshrink  
graph display cred_bins, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 5B.pdf", replace as(pdf) 


* GDP
#delimit ;
twoway connected bB1_R_By bB2_R_By bB3_R_By bB4_R_By
				 bB6_R_By bB7_R_By bB8_R_By	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 				 				 
				 title("Bank equity returns") name(y_R_B_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.06(0.02)0.02)) xlabel(#7) 
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%") rows(2));
#delimit cr
#delimit ;
twoway connected bB1_R_Ny bB2_R_Ny bB3_R_Ny bB4_R_Ny
				 bB6_R_Ny bB7_R_Ny bB8_R_Ny	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 				 				 
				 title("Nonfin. equity returns") name(y_R_N_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) yscale(range(-.06(0.02)0.02)) xlabel(#7) 
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%") rows(2));
#delimit cr
grc1leg y_R_B_bins y_R_N_bins, name(y_bins,replace) altshrink xsize(4) ysize(2) 
graph display y_bins, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 5A.pdf", replace as(pdf) 












/*------------------------------------------------------------------------
 
 Figure A6: 
 Bank Equity and Subsequent Macroeconomic Outcomes: 
 Robustness to Alternative Specifications

-------------------------------------------------------------------------*/


**** Panel (A): 30% Crashes ****

use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill

gen  R_B = C_B30
gen  R_N = C_N30

*** controls
local X0 ""
local X1 "L(1/3).R_B  L(1/3).R_N	  L(0/3).g0y  L(0/3).g0credit_to_gdp"

qui reghdfe Fd3y   R_B R_N `X1' , absorb(ccode) cluster(ccode year) 
gen smp_y=e(sample)
qui reghdfe Fd3credit_to_gdp   R_B R_N `X1' , absorb(ccode) cluster(ccode year) 
gen smp_credit_to_gdp=e(sample)
			
global H = 6
global Hp1 = $H + 1

tempfile dta_crash
save    `dta_crash'

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N"{
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash',clear

			reghdfe Fd`i'`LHS'   R_B R_N `X`j'' if smp_`LHS'==1 , absorb(ccode) cluster(ccode year)			
			
			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" {
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h


/*--- Plot Responses to Bank/Nonfin Equity Crashes ---*/

* Output Response
#delimit ;
twoway connected bBy bByu bByl bNy bNyu bNyl 
				 h if j==0, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 ytitle("Real GDP response") name(gdp_on_return_nocont,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) 
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash"));
#delimit cr

#delimit ;
twoway connected bBy bByu bByl bNy bNyu bNyl 
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP response")  name(gdp_on_return_cont,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) 
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1) );
#delimit cr


* Credit to GDP Response
#delimit ;
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bNcredit_to_gdp bNcredit_to_gdpu bNcredit_to_gdpl 
				 h if j==0, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 ytitle("Credit-to-GDP response")  name(credit_on_return_nocont,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) 
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash"));
#delimit cr

#delimit ;
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bNcredit_to_gdp bNcredit_to_gdpu bNcredit_to_gdpl
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP response") name(credit_on_return_cont,replace) plotregion(margin(zero))
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))  
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1)  );
#delimit cr


grc1leg gdp_on_return_cont credit_on_return_cont, name(return_cont,replace) altshrink xsize(4) ysize(2) 
graph display return_cont, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 6A.pdf", replace as(pdf) 



**** Panel (B): Continuous Returns ****

use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill
gen  R_B = -Rtot_real_w // negative returns
gen  R_N = -Rtot_nonfin_real_w


tempfile dta_crash
save    `dta_crash'

*** controls
local X0 ""
local X1 "L(1/3).R_B	 L(1/3).R_N 	L(0/3).g0y 	L(0/3).g0credit_to_gdp"
global H = 6
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N"{
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash',clear

			reghdfe Fd`i'`LHS'   R_B R_N `X`j'' , absorb(ccode) cluster(ccode year)		
			
			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" {
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h



/*--- Plot Responses to Bank/Nonfin Equity Return ---*/

* Output Response
#delimit ;
twoway connected bBy bByu bByl bNy bNyu bNyl 
				 h if j==0, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP response") name(gdp_on_return_nocont,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 legend(order(1 "Bank equity return shock" 4 "Nonfinancial equity return shock"));
#delimit cr

#delimit ;
twoway connected bBy bByu bByl bNy bNyu bNyl 
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP response")  name(gdp_on_return_cont,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 legend(order(1 "Bank equity return shock" 4 "Nonfinancial equity return shock") rows(1) );
#delimit cr



* Credit to GDP Response
#delimit ;
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bNcredit_to_gdp bNcredit_to_gdpu bNcredit_to_gdpl 
				 h if j==0, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP response")  name(credit_on_return_nocont,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 legend(order(1 "Bank equity return shock" 4 "Nonfinancial equity return shock"));
#delimit cr

#delimit ;
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bNcredit_to_gdp bNcredit_to_gdpu bNcredit_to_gdpl
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP response") name(credit_on_return_cont,replace) plotregion(margin(zero))
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) 
				 legend(order(1 "Bank equity return shock" 4 "Nonfinancial equity return shock") rows(1) );
#delimit cr
 
grc1leg gdp_on_return_cont credit_on_return_cont, name(return_cont,replace) altshrink xsize(4) ysize(2) //title("Real GDP Response")
graph display return_cont, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 6B.pdf", replace as(pdf) 










	
/*------------------------------------------------------------------------

 Figure A7: 
 Bank Equity Crashes and Subsequent Macroeconomic Outcomes: 
 Subsamples
 
-------------------------------------------------------------------------*/


**** Panel (A): Excluding Great Recession and Great Depression ****

use "$datapath/BVX_annual_regdata.dta", clear
gen  R_B = C_B30
gen  R_N = C_N30
foreach X of varlist R_B R_N g0y g0credit_to_gdp {
	forv i=0/3{
		gen L`i'`X'=L`i'.`X'
	}
}
drop if inrange(year,1927,1937)   // nb: this restricts data to not include bank equity crashes during these periods, but long differences still imply that some data from e.g. 2005 onward is used 
drop if inrange(year,2005,2015)
xtset ccode year
tsfill
tempfile dta_crash
save    `dta_crash'

*** controls // nb: defining controls this way allows controls to be drawn from restricted years
local X0 ""
local X1 "   L1R_B L2R_B L3R_B	 L1R_N L2R_N L3R_N	  L0g0y L1g0y L2g0y L3g0y L0g0credit_to_gdp L1g0credit_to_gdp L2g0credit_to_gdp L3g0credit_to_gdp"

global H = 10
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N"{
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash',clear

			reghdfe Fd`i'`LHS'   R_B R_N `X`j'' , absorb(ccode) cluster(ccode year)			
			
			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" {
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h

* Output Response
#delimit ;
twoway connected bBy bByu bByl bNy bNyu bNyl 
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP response")  name(output,replace)
				 xtitle("Years after shock", size(medlarge)) ylabel(#6, format(%03.2f) labsize(medlarge))   xlabel(#4, labsize(medlarge))
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1)  );
#delimit cr

* Credit to GDP Response
#delimit ;
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bNcredit_to_gdp bNcredit_to_gdpu bNcredit_to_gdpl
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP response") name(credit,replace) plotregion(margin(zero))
				 xtitle("Years after shock", size(medlarge)) ylabel(#6, format(%03.2f) labsize(medlarge))   xlabel(#4, labsize(medlarge))
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1)  );
#delimit cr


grc1leg output credit, name(ex_GreatD_GreatR,replace) altshrink xsize(6)  //title("Excluding Great Depression and Great Recession")
graph display ex_GreatD_GreatR, ysize(6.5) xsize(9.0)
graph export "$output/Appendix Figure 7A.pdf", as(pdf) replace



**** Panel (B): Pre-war ****
use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill

gen  R_B = C_B30
gen  R_N = C_N30
foreach X of varlist R_B R_N g0y g0credit_to_gdp {
	forv i=0/3{
		gen L`i'`X'=L`i'.`X'
	}
}
keep if year<=1939    
tempfile dta_crash
save    `dta_crash'

*** controls
local X0 ""
local X1 "L1R_B L2R_B L3R_B	 L1R_N L2R_N L3R_N	  L0g0y L1g0y L2g0y L3g0y   L0g0credit_to_gdp L1g0credit_to_gdp L2g0credit_to_gdp L3g0credit_to_gdp"

global H = 10
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N"{
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash',clear

			reghdfe Fd`i'`LHS'   R_B R_N `X`j'' , absorb(ccode) cluster(ccode year)			
			
			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" {
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h

* Output Response
#delimit ;
twoway connected bBy bByu bByl bNy bNyu bNyl 
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP response")  name(output,replace)
				 xtitle("Years after shock", size(medlarge)) ylabel(#6, format(%03.2f) labsize(medlarge))   xlabel(#4, labsize(medlarge))
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1) );
#delimit cr


* Credit to GDP Response
#delimit ;
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bNcredit_to_gdp bNcredit_to_gdpu bNcredit_to_gdpl
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP response") name(credit,replace) plotregion(margin(zero))
				 xtitle("Years after shock", size(medlarge)) ylabel(#6, format(%03.2f) labsize(medlarge))   xlabel(#4, labsize(medlarge))
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1)  );
#delimit cr


grc1leg output credit, name(prewar,replace) altshrink //title("Pre-1939 Data")
graph display prewar, ysize(6.5) xsize(9.0)
graph export "$output/Appendix Figure 7B.pdf", as(pdf) replace




**** Panel (C): Post-war ****
use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill

gen  R_B = C_B30
gen  R_N = C_N30
foreach X of varlist R_B R_N g0y g0credit_to_gdp {
	forv i=0/3{
		gen L`i'`X'=L`i'.`X'
	}
}
tab year, gen(year_)
keep if year>=1945    // POST-WAR
tempfile dta_crash
save    `dta_crash'

*** controls
local X0 ""
local X1 "   L1R_B L2R_B L3R_B	 L1R_N L2R_N L3R_N	  L0g0y L1g0y L2g0y L3g0y L0g0credit_to_gdp L1g0credit_to_gdp L2g0credit_to_gdp L3g0credit_to_gdp"

global H = 10
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N"{
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash',clear

			reghdfe Fd`i'`LHS'   R_B R_N `X`j'' , absorb(ccode) cluster(ccode year) 
			
			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" {
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h

* Output Response
#delimit ;
twoway connected bBy bByu bByl bNy bNyu bNyl 
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP response")  name(output,replace)
				 xtitle("Years after shock", size(medlarge)) ylabel(#6, format(%03.2f) labsize(medlarge))   xlabel(#4, labsize(medlarge))
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1) );
#delimit cr


* Credit to GDP Response
#delimit ;
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bNcredit_to_gdp bNcredit_to_gdpu bNcredit_to_gdpl
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP response") name(credit,replace) plotregion(margin(zero))
				 xtitle("Years after shock", size(medlarge)) ylabel(#6, format(%03.2f) labsize(medlarge))   xlabel(#4, labsize(medlarge))
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1)  );
#delimit cr

grc1leg output credit, name(postwar,replace) altshrink //title("Post-1945 Data")
graph display postwar, ysize(6.5) xsize(9.0)
graph export "$output/Appendix Figure 7C.pdf", as(pdf) replace



**** Panel (D): Post-war - Bretton Woods ****

use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill
gen  R_B = C_B30
gen  R_N = C_N30
foreach X of varlist R_B R_N g0y g0credit_to_gdp {
	forv i=0/3{
		gen L`i'`X'=L`i'.`X'
	}
}
keep if year>=1946 & year<=1970    
tempfile dta_crash
save    `dta_crash'

*** controls
local X0 ""
local X1 "L1R_B L2R_B L3R_B	 L1R_N L2R_N L3R_N	  L0g0y L1g0y L2g0y L3g0y L0g0credit_to_gdp L1g0credit_to_gdp L2g0credit_to_gdp L3g0credit_to_gdp"

global H = 10
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N"{
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash',clear

			reghdfe Fd`i'`LHS'   R_B R_N `X`j'' , absorb(ccode) cluster(ccode year)			
			
			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" {
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h

* Output Response
#delimit ;
twoway connected bBy bByu bByl bNy bNyu bNyl 
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP response")  name(output,replace)
				 xtitle("Years after shock", size(medlarge)) ylabel(#6, format(%03.2f) labsize(medlarge))   xlabel(#4, labsize(medlarge))
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1) );
#delimit cr


* Credit to GDP Response
#delimit ;
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bNcredit_to_gdp bNcredit_to_gdpu bNcredit_to_gdpl
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP response") name(credit,replace) plotregion(margin(zero))
				 xtitle("Years after shock", size(medlarge)) ylabel(#6, format(%03.2f) labsize(medlarge))   xlabel(#4, labsize(medlarge))
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1)  );
#delimit cr

grc1leg output credit, name(Fig_crash_postwar_1946_70,replace) altshrink //title("Post-1945 Data")
graph display Fig_crash_postwar_1946_70, ysize(6.5) xsize(9.0)
graph export "$output/Appendix Figure 7D.pdf", as(pdf) replace



**** Panel (E): Post-war - After Bretton Woods ****

use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill
gen  R_B = C_B30
gen  R_N = C_N30
foreach X of varlist R_B R_N g0y g0credit_to_gdp {
	forv i=0/3{
		gen L`i'`X'=L`i'.`X'
	}
}
tab year, gen(year_)
keep if year>1970    
tempfile dta_crash
save    `dta_crash'

*** controls
local X0 ""
local X1 "   L1R_B L2R_B L3R_B	 L1R_N L2R_N L3R_N	  L0g0y L1g0y L2g0y L3g0y L0g0credit_to_gdp L1g0credit_to_gdp L2g0credit_to_gdp L3g0credit_to_gdp"

global H = 10
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N"{
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash',clear

			reghdfe Fd`i'`LHS'   R_B R_N `X`j'' , absorb(ccode) cluster(ccode year)
			
			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" {
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h

* Output Response
#delimit ;
twoway connected bBy bByu bByl bNy bNyu bNyl 
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP response")  name(output,replace)
				 xtitle("Years after shock", size(medlarge)) ylabel(#6, format(%03.2f) labsize(medlarge))   xlabel(#4, labsize(medlarge))
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1) );
#delimit cr


* Credit to GDP Response
#delimit ;
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bNcredit_to_gdp bNcredit_to_gdpu bNcredit_to_gdpl
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP response") name(credit,replace) plotregion(margin(zero))
				 xtitle("Years after shock", size(medlarge)) ylabel(#6, format(%03.2f) labsize(medlarge))   xlabel(#4, labsize(medlarge))
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1)  );
#delimit cr

grc1leg output credit, name(Fig_crash_postwar_1971_2016,replace)  altshrink //title("Post-1945 Data")
graph display Fig_crash_postwar_1971_2016, ysize(6.5) xsize(9.0)
graph export "$output/Appendix Figure 7E.pdf", as(pdf) replace










/*------------------------------------------------------------------------
 
 Figure A8: 
 Bank Equity Crashes and Subsequent Macroeconomic Outcomes: U.S. Only
 
-------------------------------------------------------------------------*/


***** Panel (A) USA full sample *****
use "$datapath/BVX_annual_regdata.dta", clear
keep if country=="US"
tsset year
gen  R_B = C_B30
gen  R_N = C_N30

tab R_B R_N
list year R_B R_N PANIC if R_B==1 | R_N==1

foreach X of varlist R_B R_N g0y g0credit_to_gdp {
	forv i=0/3{
		gen L`i'`X'=L`i'.`X'
	}
}

*** controls
local X0 ""
local X1 "L1R_B L2R_B L3R_B	 L1R_N L2R_N L3R_N	  L0g0y L1g0y L2g0y L3g0y L0g0credit_to_gdp L1g0credit_to_gdp L2g0credit_to_gdp L3g0credit_to_gdp "

* Use observations with at least t to t+3 years window
foreach LHS in "y" "credit_to_gdp" {
	qui reg Fd3`LHS'   R_B R_N `X1' 
	gen smp_`LHS'=e(sample)
}
			
global H = 6
global Hp1 = $H + 1

tempfile dta_crash
save    `dta_crash'

*ssc install newey2

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N"{
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash',clear
			
			newey2 Fd`i'`LHS'   R_B R_N `X`j'' if smp_`LHS'==1 , lag(6) force
			
			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" {
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h


* Output Response
#delimit ;
twoway connected bBy bByu bByl bNy bNyu bNyl 
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP Response")  name(gdp_usa,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) 
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1) );
#delimit cr


* Credit to GDP Response
#delimit ;
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bNcredit_to_gdp bNcredit_to_gdpu bNcredit_to_gdpl
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP Response") name(credit_usa,replace) plotregion(margin(zero))
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) 
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1)  );
#delimit cr
 
grc1leg gdp_usa credit_usa, name(usa,replace) altshrink xsize(4) ysize(2) 
graph display usa, ysize(5.0) xsize(9.0)
graph export "$output/Appendix Figure 8A.pdf", replace as(pdf) 


 


***** Panel (B): USA ex Great Recession and Great Depression *****

use "$datapath/BVX_annual_regdata.dta", clear

keep if country=="US"
drop if inrange(year,1927,1937) 
drop if inrange(year,2005,2015)

tsset year
gen  R_B = C_B30
gen  R_N = C_N30

tab R_B R_N
list year R_B R_N PANIC if R_B==1 | R_N==1

foreach X of varlist R_B R_N g0y g0credit_to_gdp {
	forv i=0/3{
		gen L`i'`X'=L`i'.`X'
	}
}

*** controls
local X0 ""
local X1 "L1R_B L2R_B L3R_B	 L1R_N L2R_N L3R_N	  L0g0y L1g0y L2g0y L3g0y L0g0credit_to_gdp L1g0credit_to_gdp L2g0credit_to_gdp L3g0credit_to_gdp "

foreach LHS in "y" "credit_to_gdp" {
	qui reg Fd3`LHS'   R_B R_N `X1' 
	gen smp_`LHS'=e(sample)
}
			
global H = 6
global Hp1 = $H + 1

tempfile dta_crash
save    `dta_crash'

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N"{
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash',clear
			
			newey2 Fd`i'`LHS'   R_B R_N `X`j'' if smp_`LHS'==1 , lag(6) force
			
			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" {
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h


/*--- Plot Responses to Bank/Nonfin Equity Crashes ---*/

* Output Response
#delimit ;
twoway connected bBy bByu bByl bNy bNyu bNyl 
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP response")  name(gdp_usa,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f)) 
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1) );
#delimit cr


* Credit to GDP Response
#delimit ;
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bNcredit_to_gdp bNcredit_to_gdpu bNcredit_to_gdpl
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP response") name(credit_usa,replace) plotregion(margin(zero))
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))  
				 legend(order(1 "Bank equity crash" 4 "Nonfinancial equity crash") rows(1)  );
#delimit cr
 
grc1leg gdp_usa credit_usa, name(usa,replace) altshrink xsize(4) ysize(2) //title("Real GDP Response")
graph display usa, ysize(5.0) xsize(9.0)
graph export "$output/Appendix Figure 8B.pdf", replace as(pdf) 





/*------------------------------------------------------------------------

 Figure A9: 
 Bank Equity Crashes Excluding Panic Episodes
 
-------------------------------------------------------------------------*/

use "$datapath/BVX_annual_regdata.dta", clear

*** Prepare Variables
gen  R_B = Rtot_real_w
gen  R_N = Rtot_nonfin_real_w
foreach X in R_B R_N {
	gen B1_`X' = ( `X' <= -.45)              if `X'~=.
	gen B2_`X' = ( `X' >  -.45 & `X' <=-.30) if `X'~=.
	gen B3_`X' = ( `X' >  -.30 & `X' <=-.15) if `X'~=.
	gen B4_`X' = ( `X' >  -.15 & `X' <=0.0)  if `X'~=.	
	gen B5_`X' = ( `X' >  0.0 & `X' <=.15)   if `X'~=.
	gen B6_`X' = ( `X' >  0.15 & `X' <=.30)  if `X'~=.
	gen B7_`X' = ( `X' >  0.30 & `X' <=.45)  if `X'~=.
	gen B8_`X' = ( `X' >  0.45 ) if `X'~=.
}

drop if PANIC==1 | L.PANIC==1 | L2.PANIC==1 | L3.PANIC==1 | F.PANIC==1 | F2.PANIC==1 | F3.PANIC==1

foreach X of varlist B?_R_B B?_R_N {
	forv i=1/3{
		gen L`i'`X' = L`i'.`X'
	}
}
foreach X of varlist g0y g0credit_to_gdp {
	forv i=0/3{
		gen L`i'`X' = L`i'.`X'
	}
}

global lags3 "L?B?_R_B L?B?_R_N"
local X0 ""
local X1 "$lags3	L0g0y L1g0y L2g0y L3g0y L0g0credit_to_gdp L1g0credit_to_gdp L2g0credit_to_gdp L3g0credit_to_gdp"

global RHS        "B1_R_B B2_R_B B3_R_B B4_R_B 	      B6_R_B B7_R_B B8_R_B  B1_R_N B2_R_N B3_R_N B4_R_N        B6_R_N B7_R_N B8_R_N"
global RHS_noomit "B1_R_B B2_R_B B3_R_B B4_R_B B5_R_B B6_R_B B7_R_B B8_R_B  B1_R_N B2_R_N B3_R_N B4_R_N B5_R_N B6_R_N B7_R_N B8_R_N"

foreach LHS in "y" "credit_to_gdp" {
	qui reghdfe Fd3`LHS'   $RHS `X1' , absorb(ccode year) cluster(ccode year)
	gen smp_`LHS' = e(sample)
}
tempfile dta_crash
save    `dta_crash'


*** Local Projection
global H = 6
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in $RHS_noomit {
			gen b`k'`LHS'   = 0 
			gen b`k'`LHS'se = 0 
			gen b`k'`LHS'u  = 0 
			gen b`k'`LHS'l  = 0 	
		}
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_crash',clear
			
			reghdfe Fd`i'`LHS'   $RHS `X`j'' if smp_`LHS'==1  , absorb(ccode) cluster(ccode year)
			
			use `EST_`LHS'_`j'', clear
			foreach k in $RHS {
				replace b`k'`LHS'   = _b[`k']  if h ==`i'
				replace b`k'`LHS'se = _se[`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h


* Credit
#delimit ;
twoway connected bB1_R_Bcredit_to_gdp bB2_R_Bcredit_to_gdp bB3_R_Bcredit_to_gdp bB4_R_Bcredit_to_gdp
				 bB6_R_Bcredit_to_gdp bB7_R_Bcredit_to_gdp	bB8_R_Bcredit_to_gdp
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Bank equity returns") name(credit_R_B_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))   yscale(range(-.06(0.02)0.06))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%") rows(2));
#delimit cr

#delimit ;
twoway connected bB1_R_Ncredit_to_gdp bB2_R_Ncredit_to_gdp bB3_R_Ncredit_to_gdp	bB4_R_Ncredit_to_gdp 
                 bB6_R_Ncredit_to_gdp bB7_R_Ncredit_to_gdp bB8_R_Ncredit_to_gdp	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Nonfin. equity returns") name(credit_R_N_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))   yscale(range(-.06(0.02)0.06))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%") rows(2));
#delimit cr
grc1leg credit_R_B_bins credit_R_N_bins, name(cred_bins,replace) altshrink 
graph display cred_bins, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 9B.pdf", replace as(pdf) 


* GDP
#delimit ;
twoway connected bB1_R_By bB2_R_By bB3_R_By bB4_R_By
				 bB6_R_By bB7_R_By bB8_R_By	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Bank equity returns") name(y_R_B_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))   yscale(range(-.06(0.02)0.02))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%") rows(2));
#delimit cr
#delimit ;
twoway connected bB1_R_Ny bB2_R_Ny bB3_R_Ny bB4_R_Ny
				 bB6_R_Ny bB7_R_Ny bB8_R_Ny	
				 h if j==1 & h<=6, color(blue blue blue blue red red red red) m(s c t d  i i i) lpattern(solid dash shortdash dot solid dash shortdash dot)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Nonfin. equity returns") name(y_R_N_bins,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))   yscale(range(-.06(0.02)0.02))
				 legend(order(1 "{&le}-45%" 2 "(-45%,-30%]" 3 "(-30%,-15%]" 4 "(-15%,0%]" 5 "(15%,30%]" 6 "(30%,45%]" 7 ">45%") rows(2));
#delimit cr
grc1leg y_R_B_bins y_R_N_bins, name(y_bins,replace) altshrink 
graph display y_bins, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 9A.pdf", replace as(pdf) 







/*------------------------------------------------------------------------

 Figure A10:
 Impact of BVX Crises With and Without Panics
 
-------------------------------------------------------------------------*/

use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill

*** Prepare Data
gen     R_B   = RC     // (i)  BVX Crisis Indicator
gen     R_P   = PANIC  // (ii) Panic Indicator
gen     R_N   = C_N30  // 30% non-fin equity crash control
gen g0d_y = g0credit_to_gdp
* NB: Panic indicator is a subset of BVX crisis indicator:
tab R_B R_P 

*** controls
local X0 ""
local X1 "L(1/3).R_B   L(1/3).R_P  L(1/3).R_N  L(0/3).g0y   L(0/3).g0d_y  "
foreach LHS in "y" "credit_to_gdp" {
	reghdfe Fd3`LHS'   R_B R_P R_N   `X1' , absorb(ccode year) cluster(ccode year)   
	gen smp_`LHS'=e(sample)
}

tempfile  dta_PANIC
save     `dta_PANIC'

*** Estimation 
global H = 6
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N" "P" {
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
			gen bBOTH`LHS' = 0 if h==0
			gen bBOTH`LHS'u = 0 if h==0
			gen bBOTH`LHS'l = 0 if h==0
			
		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_PANIC',clear
			
			reghdfe Fd`i'`LHS'   R_B R_P R_N `X`j''  if smp_`LHS'==1, absorb(ccode) cluster(ccode year)  
			lincom R_B + R_P
			local b_sum = r(estimate)
			local se_sum = r(se)

			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" "P" {
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
				replace bBOTH`LHS'  = `b_sum' if h ==`i'
				replace bBOTH`LHS'u = `b_sum' + 1.96*`se_sum'  if h ==`i'
				replace bBOTH`LHS'l = `b_sum' - 1.96*`se_sum'  if h ==`i'
				
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h

#delimit ;
* Output Response;
twoway connected bBy bByu bByl  bBOTHy bBOTHyu bBOTHyl 
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP Response") name(gdp,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 legend(order(1 "BVX Banking Crisis without Panic" 4 "BVX Banking Crisis with Panic"));
* Credit to GDP Response;				 
twoway connected bBcredit_to_gdp bBcredit_to_gdpu bBcredit_to_gdpl bBOTHcredit_to_gdp bBOTHcredit_to_gdpu bBOTHcredit_to_gdpl 
				 h if j==1 & h<=6, color(blue blue blue red red red) lpattern(solid dot dot solid dot dot) m(s i i c i i) 
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP Response") name(credit,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 legend(order(1 "BVX Banking Crisis without Panic" 4 "BVX Banking Crisis with Panic"));
#delimit cr

grc1leg gdp credit, name(baseline,replace) altshrink xsize(5) ysize(2) 
graph display baseline, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 10.pdf", replace as(pdf) 




 

/*------------------------------------------------------------------------

 Figure A11: 
 Banking Distress With and Without Banking Panics: Finer Panics Classification
		
-------------------------------------------------------------------------*/

**** Panel (A): Baseline ****

use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill

tab PANIC_finer_f , gen(FP_f_)

*** Prepare Data
gen     R_B    = C_B30    
gen     R_B_P2 = C_B30*FP_f_2  // bank equity crash x isolated runs / borderline cases
gen     R_B_P3 = C_B30*FP_f_3  // bank equity crash x clear-cut panic
gen     R_P2   = FP_f_2        // isolated runs / borderline cases
gen     R_P3   = FP_f_3        // clear-cut panic
gen     R_N    = C_N30

gen g0d_y = g0credit_to_gdp

*** controls
local X0 ""
local X1 "L(1/3).R_B_P?   L(1/3).R_B   L(1/3).R_P?  L(1/3).R_N  L(0/3).g0y   L(0/3).g0d_y"
foreach LHS in "y" "credit_to_gdp" {
	reghdfe Fd3`LHS'   R_B  R_B_P2 R_B_P3  R_P2 R_P3  R_N   `X1' , absorb(ccode) cluster(ccode year)   
	gen smp_`LHS'=e(sample)
}

tempfile  dta_PANIC
save     `dta_PANIC'

global H = 6
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N" "P2" "P3" "B_P2" "B_P3" {
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
			gen bBOTH2`LHS' = 0 if h==0
			gen bBOTH3`LHS' = 0 if h==0

		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_PANIC',clear
			
			reghdfe Fd`i'`LHS'   R_B R_B_P2 R_B_P3  R_P2 R_P3  R_N `X`j''  if smp_`LHS'==1, absorb(ccode) cluster(ccode year)

			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" "P2" "P3" "B_P2" "B_P3" {
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
				replace bBOTH2`LHS' = bB`LHS' + bP2`LHS' + bB_P2`LHS' 
				replace bBOTH3`LHS' = bB`LHS' + bP3`LHS' + bB_P3`LHS' 
				
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h


#delimit ;
* Output Response;
twoway connected bBy bP2y bP3y  bBOTH2y bBOTH3y  
				 h if j==1 & h<=6, color(blue red red dkgreen dkgreen) lpattern(solid dash solid dash solid) m(s c c t t ) 
				 title("Real GDP response") name(gdp,replace)
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 legend(order(1 "Bank equity crash, no panic" 2 "Isolated runs, no crash" 3 "Clear-cut panic, no crash" 
				              4 "Bank equity crash, isolated runs" 5 "Bank equity crash, clear-cut panic" ) holes(2) );
							  
* Credit to GDP Response;				 
twoway connected bBcredit_to_gdp bP2credit_to_gdp bP3credit_to_gdp  bBOTH2credit_to_gdp bBOTH3credit_to_gdp  
				 h if j==1 & h<=6, color(blue red red dkgreen dkgreen) lpattern(solid dash solid dash solid) m(s c c t t ) 
				 title("Credit-to-GDP response") name(credit,replace)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 legend(order(1 "Bank equity crash, no panic" 2 "Isolated runs, no crash" 3 "Clear-cut panic, no crash" 
				              4 "Bank equity crash, isolated runs" 5 "Bank equity crash, clear-cut panic" ) holes(2) );							  
#delimit cr

grc1leg gdp credit, name(finer2,replace) altshrink xsize(5) ysize(2) 
graph display finer2, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 11A.pdf", replace as(pdf) 





**** Panel (B): 2 categories of panic severity, conditioning on bank faiures ****

use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill

tab PANIC_finer_f , gen(FP_f_)

*** Prepare Data
gen R_B   = 0 if Rtot_real!=.
gen R_B_P2 = 0 if Rtot_real!=.
gen R_B_P3 = 0 if Rtot_real!=.
gen R_P2   = 0 if Rtot_real!=.
gen R_P3   = 0 if Rtot_real!=.
replace R_B    = 1 if C_B30==1 & bankfailure_narrative_f==1
replace R_B_P2 = 1 if C_B30==1 & bankfailure_narrative_f==1 & FP_f_2==1
replace R_B_P3 = 1 if C_B30==1 & bankfailure_narrative_f==1 & FP_f_3==1
replace R_P2   = 1 if FP_f_2==1 
replace R_P3   = 1 if FP_f_3==1 

gen     R_N   = C_N30
gen g0d_y = g0credit_to_gdp

*** controls
local X0 ""
local X1 "L(1/3).R_B_P?   L(1/3).R_B   L(1/3).R_P?  L(1/3).R_N  L(0/3).g0y   L(0/3).g0d_y  "
foreach LHS in "y" "credit_to_gdp" {
	reghdfe Fd3`LHS'   R_B  R_B_P2 R_B_P3  R_P2 R_P3  R_N   `X1' , absorb(ccode) cluster(ccode year)   
	gen smp_`LHS'=e(sample)
}

tempfile  dta_PANIC
save     `dta_PANIC'

global H = 6
global Hp1 = $H + 1

foreach LHS in "y" "credit_to_gdp" {
	foreach j in "0" "1" {

		clear
		set obs $Hp1 
		gen j = `j'
		gen h  = _n - 1 
		foreach k in "B" "N" "P2" "P3" "B_P2" "B_P3" {
			gen b`k'`LHS'  = 0 if h==0
			gen b`k'`LHS'se= 0 if h==0
			gen b`k'`LHS'u = 0 if h==0
			gen b`k'`LHS'l = 0 if h==0	
		}
			gen bBOTH2`LHS' = 0 if h==0
			gen bBOTH3`LHS' = 0 if h==0

		gen Nobs`LHS'=. 
		tempfile EST_`LHS'_`j'
		save    `EST_`LHS'_`j''

		forv i=1/$H{
			use `dta_PANIC',clear
			
			reghdfe Fd`i'`LHS'   R_B R_B_P2 R_B_P3  R_P2 R_P3  R_N `X`j''  if smp_`LHS'==1 , absorb(ccode) cluster(ccode year)

			use `EST_`LHS'_`j'', clear
			foreach k in "B" "N" "P2" "P3" "B_P2" "B_P3" {
				replace b`k'`LHS'   = _b[R_`k']  if h ==`i'
				replace b`k'`LHS'se = _se[R_`k'] if h ==`i'
				replace b`k'`LHS'u  = b`k'`LHS' + 1.96*b`k'`LHS'se if h ==`i'
				replace b`k'`LHS'l  = b`k'`LHS' - 1.96*b`k'`LHS'se if h ==`i'
			}
				replace bBOTH2`LHS' = bB`LHS' + bP2`LHS' + bB_P2`LHS' 
				replace bBOTH3`LHS' = bB`LHS' + bP3`LHS' + bB_P3`LHS' 
				
			replace Nobs`LHS'= `e(N)'  if h ==`i'
			save `EST_`LHS'_`j'', replace	
		}

	}
	
	use          `EST_`LHS'_0', clear
	append using `EST_`LHS'_1'
	tempfile EST_`LHS'
	save    `EST_`LHS''
}

use `EST_y', clear
merge 1:1 j h using `EST_credit_to_gdp', nogen

sort j h


#delimit ;
* Output Response;
twoway connected bBy bP2y bP3y  bBOTH2y bBOTH3y  
				 h if j==1 & h<=6, color(blue red red dkgreen dkgreen) lpattern(solid dash solid dash solid) m(s c c t t ) 
				 title("Real GDP response") name(gdp,replace)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 legend(order(1 "Bank eq. crash and failures, no panic" 2 "Isolated runs, no bank failures" 3 "Clear-cut panic, no bank failures" 
				              4 "Bank eq. crash and failures, isolated runs" 5 "Bank eq. crash and failures, clear-cut panic" ) holes(2) );
							  
* Credit to GDP Response;				 
twoway connected bBcredit_to_gdp bP2credit_to_gdp bP3credit_to_gdp  bBOTH2credit_to_gdp bBOTH3credit_to_gdp  
				 h if j==1 & h<=6, color(blue red red dkgreen dkgreen) lpattern(solid dash solid dash solid) m(s c c t t ) 
				 title("Credit-to-GDP response") name(credit,replace)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))
				 legend(order(1 "Bank eq. crash and failures, no panic" 2 "Isolated runs, no crash" 3 "Clear-cut panic, no crash" 
				              4 "Bank eq. crash and failures, isolated runs" 5 "Bank eq. crash and failures, clear-cut panic" ) holes(2) );							  
#delimit cr

grc1leg gdp credit, name(finer2_failures,replace) altshrink xsize(5) ysize(2) 
graph display finer2_failures, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 11B.pdf", replace as(pdf) 






/*------------------------------------------------------------------------

 Figure A12: 
 Frequency of Panic and Non-Panic Crises Across Decades
 
-------------------------------------------------------------------------*/

use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill

** Create variables

* Quiet crisis on BVX list
gen     QUIET = 0 
replace QUIET = 1 if RC==1 & PANIC_ind==0

* Quiet crisis: Bank 30% cumulative equity decline
gen     QUIET2 = 0 
replace QUIET2 = 1 if bankeqdecline==1 & PANIC_ind==0


collapse (mean) RC QUIET QUIET2 PANIC_ind, by(decade)
drop if decade==.

twoway connected RC PANIC_ind QUIET QUIET2 decade, ///
	mcolor(blue black red dkgreen) lcolor(blue black red dkgreen) ///
	legend(order(1 "BVX banking crisis" 2 "BVX panic crisis" 3 "BVX non-panic crisis" 4 "All 30% bank equity declines without panic")) ///
	xlabel(#8) ylabel(#6, format(%03.2f)) ytitle("Frequency of crisis by decade") name(ts,replace)
graph export "$output/Appendix Figure 12.pdf", as(pdf) replace








/*------------------------------------------------------------------------

 Figure A14: 
 Bank and Nonfinancial Equity Around Banking Crises and Normal Recessions
 
-------------------------------------------------------------------------*/


use "$datapath/BVX_annual_regdata.dta", clear

* Prepare variables
gen R_B = Rtot_real_w
gen R_N = Rtot_nonfin_real_w

gen R_Bl = ln(1+R_B) // log returns
gen R_Nl = ln(1+R_N)

foreach X of varlist R_B R_N {
		
	* cumulative log return
	gen Fd1`X'l = F.`X'l
	forv i=2/15 {
		local im1 = `i'-1
		gen Fd`i'`X'l = F`i'.`X'l + Fd`im1'`X'l
	}	
}
local W=5
foreach X in R_Bl R_Nl {
	gen Fd0`X'=0
}
foreach X of varlist Fd*R_Bl Fd*R_Nl {
	gen L_`X' = L`W'.`X'
}


* (A) "Nonfinancial recessions"
gen     NF_REC = 0 if rgdp_gr~=.
replace NF_REC = 1 if P2==1 & (RC~=1 & L.RC~=1 & F.RC~=1) // peak that doesn't coincide with crisis within a year

* (B) "Financial recessions"
gen     F_REC = 0 if rgdp_gr~=.
replace F_REC = 1 if P2==1 & (RC==1 | L.RC==1 | F.RC==1)

tempfile EVENTDATA
save    `EVENTDATA'

foreach XX in "NF_REC" "F_REC"{

	use `EVENTDATA', clear
	
	gen TREAT = `XX'
	
	collapse (mean) L_Fd*R_Bl L_Fd*R_Nl    , by(TREAT)
	drop if TREAT==.
	ren L_Fd* Fd*
	foreach X in  R_Bl R_Nl{
		ren Fd*`X' `X'*
	}
	reshape long  R_Bl R_Nl, i(TREAT) j(time)
	replace time=time-`W'
	sum time
	reshape wide  R_Bl R_Nl , i(time) j(TREAT)

	keep if inrange(time,-5,5)
	
	tempfile XX_`XX'
	save    `XX_`XX''
}
*


* Figure: Annual, Nonfinancial recession
use `XX_NF_REC', clear

#delimit ;
twoway connected R_Bl1 R_Nl1 time if time>=-5, 
yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
xtitle("Years after start of ordinary recession") ytitle("Cumulative log return relative to t=-5")
mcolor(blue red) lcolor(blue red) xlabel(#10)
legend(order(1 "Bank equity" 2 "Nonfin. equity")) name(NF_REC_log,replace);
#delimit cr	
graph export "$output/Appendix Figure 14A.pdf", as(pdf) replace

	
* Figure: Annual, financial recession
use `XX_F_REC', clear

#delimit ;
twoway connected R_Bl1 R_Nl1 time if time>=-5, 
yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
xtitle("Years after start of banking crisis recession") ytitle("Cumulative log return relative to t=-5")
mcolor(blue red) lcolor(blue red) xlabel(#10)
legend(order(1 "Bank equity" 2 "Nonfin. equity")) name(F_REC_log,replace);
#delimit cr		
graph export "$output/Appendix Figure 14B.pdf", as(pdf) replace







/*------------------------------------------------------------------------

Tables 7 and A13:
Comparisons of banking crisis chronologies	

------------------------------------------------------------------------*/


use "$root/data/additional data/Narrative Crisis List, Panics List, and BVX List.dta", clear
drop if jointlist==. & add==.
replace jointlist = add if add!=.
rename jointlist JointCrisisList
merge 1:1 country JointCrisisList using "$root/data/additional data/Narrative chronologies.dta"
drop if _merge==2
drop _merge year_bankeqdecline exclude_war panic bankeqdecline bankfailures_widespread
rename JointCrisisList year
replace country="US" if country=="U.S."
replace country="UK" if country=="U.K."
save "$root/data/temp/BVX list and other chronologies combined.dta", replace



capture log close
log using "$root/output/Tables 7 and A13.txt", replace text


quietly forval col1 = 1/8 {


	use "$root/data/additional data/peak to trough measures.dta", clear

	merge 1:1 country year using "$root/data/temp/BVX list and other chronologies combined.dta"
	drop _merge
	rename add addedcrisis
	rename delete deletedcrisis
	rename revised revisedchronology
	rename ReinhartRogoff reinhartrogoff
	rename SchularickTaylor schularicktaylor 
	rename LaevenValencia laevenvalencia
	rename Bordo bordo
	rename CaprioKlingebiel caprioklingebiel
	rename DemirgucKuntDetrag demirguckuntdetragiache


	replace log_rgdp_gr_pptdecline = -log_rgdp_gr_pptdecline

	* Table 7
	if "`col1'"=="1" noisily display _n _n "BVX Crisis List"
	if "`col1'"=="2" noisily display _n _n "BVX Crisis List (conditional on BE decline > 30%)"
	if "`col1'"=="3" noisily display _n _n "Newly uncovered crises"
	if "`col1'"=="4" noisily display _n _n "Removed crises"

	* Table A13
	if "`col1'"=="5" noisily display _n _n "Reinhart Rogoff crises"
	if "`col1'"=="6" noisily display _n _n "Laeven Valencia crises"
	if "`col1'"=="7" noisily display _n _n "BVX Crises (on Laeven-Valencia sample)"
	if "`col1'"=="8" noisily display _n _n "BVX Crises (conditional on BE decline > 30%, on Laeven-Valencia sample)"

	*** BVX Chronology
	if "`col1'"=="1" gen jointcrisis = 1 if revisedchronology!=.
	*** BVX Chronology conditional on Bank equity decline
	if "`col1'"=="2" gen jointcrisis = 1 if revisedchronology!=. & log_bank_ret_pktr!=. & log_bank_ret_pktr<log(1-0.30)
	*** Added episodes
	if "`col1'"=="3" gen jointcrisis = 1 if addedcrisis!=.
	*** Deleted episodes
	if "`col1'"=="4" gen jointcrisis = 1 if deletedcrisis!=.
	*** Reinhart-Rogoff
	if "`col1'"=="5" gen jointcrisis = 1 if reinhartrogoff!=.
	*** Laeven-Valencia
	if "`col1'"=="6" gen jointcrisis = 1 if laevenvalencia!=.
	*** BVX Chronology (on Laeven-Valencia sample)
	if "`col1'"=="7" gen jointcrisis = 1 if inrange(year,1970,2012) & revisedchronology!=.
	*** BVX Chronology conditional on Bank equity decline (on Laeven-Valencia sample)
	if "`col1'"=="8" gen jointcrisis = 1 if inrange(year,1970,2012) & revisedchronology!=. & log_bank_ret_pktr!=. & log_bank_ret_pktr<log(1-0.30)

	*** Intersection of Reinhart-Rogoff and deleted
	*gen jointcrisis = 1 if reinhartrogoff!=. & deletedcrisis!=.

	*** Romer-Romer
	gen romersample = 1 if inrange(year,1967,2012) & (country=="Australia" | country=="Austria" | country=="Belgium" | country=="Canada" | ///
	country=="Denmark" | country=="Finland" | country=="France" | country=="Germany" | country=="Greece" | country=="Iceland" | ///
	country=="Ireland" | country=="Italy" | country=="Japan" | country=="Luxembourg" | country=="Netherlands" | country=="New Zealand" | ///
	country=="Norway" | country=="Portugal" | country=="Spain" | country=="Sweden" | country=="Switzerland" | country=="Turkey" | /// 
	country=="UK" | country=="US")
	*gen jointcrisis = 1 if romersample==1 & romerromer==1
	*** Reinhart-Rogoff (on Romer-Romer sample)
	*gen jointcrisis = 1 if romersample==1 & reinhartrogoff==1
	*** Revised Chronology (on Romer-Romer sample)
	*gen jointcrisis = 1 if romersample==1 & revisedchronology!=.



	sort ccode year
	drop if ccode==.
	xtset ccode year
	gen jointcrisis_3years = max(jointcrisis, L.jointcrisis, L2.jointcrisis)
	gen crisis_number = sum(jointcrisis)
	recast str20 country
	keep if jointcrisis_3years==1


	collapse (firstnm) country year log_bank_ret_pktr log_abnormal_ret_pktr log_mktcap_ret_pktr log_rgdp_gr_pktr log_rgdp_gr_pptdecline ///
	log_rgdp_gr_maxdeviation, by(crisis_number)

	sort country year


	foreach var1 in log_abnormal_ret_pktr log_mktcap_ret_pktr log_rgdp_gr_pktr log_rgdp_gr_pptdecline ///
	log_rgdp_gr_maxdeviation {
		replace `var1' = exp(`var1')-1
	}
	rename log_* *



	merge 1:1 country year using "$root/data/additional data/crisis characteristics.dta"
	drop if _merge==2

	rename (significantliabilitiesguarantees significantliquiditysupport ///
	nplatpeak accountforwhatpercentofbankingas lossindepositspktotr) ///
	(sigliabguarantees sigliqsupport npl_peak failedbankspct lossindeposits)



	noisily summ bank_ret_pktr  abnormal_ret_pktr mktcap_ret_pktr ///
	rgdp_gr_pktr rgdp_gr_pptdecline rgdp_gr_maxdeviation  ///
	failedbankspct	 npl_peak lossindeposits sigliabguarantees sigliqsupport
		
}

log close




/*------------------------------------------------------------------------

Figure A15:
Global Great Depression analysis

-------------------------------------------------------------------------*/


use "$root/data/additional data/peak to trough measures.dta", clear

merge 1:1 country year using "$root/data/temp/BVX list and other chronologies combined.dta"
drop _merge

keep if log_bank_ret_pktr!=.
drop if revised==.

keep if inrange(year,1928,1934)
sort country year
by country: keep if _n==1

gen bank_ret_pktr =exp(log_bank_ret_pktr)-1
gen rgdp_gr_pktr =exp(log_rgdp_gr_pktr)-1
keep country year revised rgdp_gr_pktr bank_ret_pktr

drop if rgdp_gr_pktr<-0.30 // Exclude Chile from graph (as noted in the caption)
append using "$root/data/additional data/noncrisis countries in Great Depression.dta"

scatter rgdp_gr_pktr bank_ret_pktr, mlabel(country) || lfit rgdp_gr_pktr bank_ret_pktr, ///
legend(off) ylabel(-0.4(0.1)0) xlabel(-0.6(0.2)0.1) ytitle("Real GDP decline (peak to trough)") ///
xtitle("Bank equity decline (peak to trough)") yline(0, lcolor(k) lw(thin)) scheme(s1color)

graph export "$root/output/Appendix Figure 15.pdf", replace


 
 
/*------------------------------------------------------------------------

 Figure A16: 
 Comparisons with Other Banking Crisis Chronologies
 
-------------------------------------------------------------------------*/


/*--- Panel (A): BVX Revised List and Reinhart Rogoff Chronology ---*/

use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill

*** Prepare Variables
gen g0d_y = g0credit_to_gdp
gen RR = (ReinhartRogoffCrisis==1)
foreach X of varlist RC RR g0y g0d_y {
	forv i=0/3{
		gen L`i'`X' = L`i'.`X'
	}
}
local X0     ""
local X1_RC  "L1RC L2RC L3RC	L0g0y L1g0y L2g0y L3g0y   L0g0d_y L1g0d_y L2g0d_y L3g0d_y"


foreach LHS in "y" "credit_to_gdp" {
	qui reghdfe Fd3`LHS' RC RR `X1_RC' L1RR L2RR L3RR , absorb(ccode year) cluster(ccode year)
	gen smp_`LHS' = e(sample)
}

tempfile dta_RR_comp
save    `dta_RR_comp'


*** Local Projection -- BVX Crisis Indicator and Reinhart Rogoff
global H = 6
global Hp1 = $H + 1

foreach SHOCK in "RC" "RR"{
	foreach LHS in "y" "credit_to_gdp" {
		foreach j in "0" "1" {
			
			local X1 "L1`SHOCK' L2`SHOCK' L3`SHOCK'	  L0g0y L1g0y L2g0y L3g0y   L0g0d_y L1g0d_y L2g0d_y L3g0d_y"
			clear
			set obs $Hp1 
			gen j = `j'
			gen h  = _n - 1 
				gen b`SHOCK'`LHS'   = 0 
				gen b`SHOCK'`LHS'se = 0 
				gen b`SHOCK'`LHS'u  = 0 
				gen b`SHOCK'`LHS'l  = 0 	
			gen Nobs`SHOCK'`LHS'    = . 
			tempfile EST_`SHOCK'_`LHS'_`j'
			save    `EST_`SHOCK'_`LHS'_`j''

			forv i=1/$H{
				use `dta_RR_comp',clear

				reghdfe Fd`i'`LHS'  `SHOCK' `X`j'' if smp_`LHS'==1  , absorb(ccode) cluster(ccode year)				
				
				use `EST_`SHOCK'_`LHS'_`j'', clear
					replace b`SHOCK'`LHS'   = _b[`SHOCK']  if h ==`i'
					replace b`SHOCK'`LHS'se = _se[`SHOCK'] if h ==`i'
					replace b`SHOCK'`LHS'u  = b`SHOCK'`LHS' + 1.96*b`SHOCK'`LHS'se if h ==`i'
					replace b`SHOCK'`LHS'l  = b`SHOCK'`LHS' - 1.96*b`SHOCK'`LHS'se if h ==`i'
				replace Nobs`SHOCK'`LHS'= `e(N)'  if h ==`i'
				save `EST_`SHOCK'_`LHS'_`j'', replace			
			}
		}
		
		use          `EST_`SHOCK'_`LHS'_0', clear
		append using `EST_`SHOCK'_`LHS'_1'
		tempfile EST_`SHOCK'_`LHS'
		save    `EST_`SHOCK'_`LHS''
	}
}

use `EST_RC_y', clear
merge 1:1 j h using `EST_RC_credit_to_gdp', nogen
merge 1:1 j h using `EST_RR_y', nogen
merge 1:1 j h using `EST_RR_credit_to_gdp', nogen

sort j h


*** GDP
* controls
#delimit ;
twoway connected bRCy bRCyu bRCyl
				 bRRy bRRyu bRRyl	
				 h if j==1 & h<=6, color(blue blue blue red red red) m(s i i c i i ) lpattern(solid dot dot solid dot dot)
				 name(BVX_vs_ReinRog_y,replace)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP Response")
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))   yscale(range(-.06(0.02)0.02))
				 legend(order(1 "BVX Crisis List" 4 "Reinhart Rogoff") rows(1));
#delimit cr

*** Credit
* controls
#delimit ;
twoway connected bRCcredit_to_gdp bRCcredit_to_gdpu bRCcredit_to_gdpl
				 bRRcredit_to_gdp bRRcredit_to_gdpu bRRcredit_to_gdpl	
				 h if j==1 & h<=6, color(blue blue blue red red red) m(s i i c i i ) lpattern(solid dot dot solid dot dot)
				 name(BVX_vs_ReinRog_credit,replace)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP Response")
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))   yscale(range(-.06(0.02)0.02))
				 legend(order(1 "BVX Crisis List" 4 "Reinhart Rogoff") rows(1));
#delimit cr

grc1leg BVX_vs_ReinRog_y BVX_vs_ReinRog_credit, name(BVX_vs_ReinRog,replace) altshrink xsize(4) ysize(2) 
graph display BVX_vs_ReinRog, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 16A.pdf", replace as(pdf) 



/*--- BVX RCL vs Laeven and Valencia ---*/

use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill
*** Prepare Variables
gen LV = (LaevenValenciaCrisis==1)
gen g0d_y = g0credit_to_gdp
foreach X of varlist RC LV g0y g0d_y {
	forv i=0/3{
		gen L`i'`X' = L`i'.`X'
	}
}
local X0     ""
local X1_RC  "L1RC L2RC L3RC	L0g0y L1g0y L2g0y L3g0y   L0g0d_y L1g0d_y L2g0d_y L3g0d_y"


foreach LHS in "y" "credit_to_gdp" {
	qui reghdfe Fd3`LHS' RC LV `X1_RC' L1LV L2LV L3LV , absorb(ccode) cluster(ccode year)
	gen smp_`LHS' = e(sample)
}

keep if year>=1970  // compare on same sample

tempfile dta_LV_comp
save    `dta_LV_comp'



*** Local Projection -- BVX Crisis Indicator
global H = 6
global Hp1 = $H + 1

foreach SHOCK in "RC" "LV"{
	foreach LHS in "y" "credit_to_gdp" {
		foreach j in "0" "1" {
			
			local X1 "L1`SHOCK' L2`SHOCK' L3`SHOCK'	  L0g0y L1g0y L2g0y L3g0y   L0g0d_y L1g0d_y L2g0d_y L3g0d_y"
			clear
			set obs $Hp1 
			gen j = `j'
			gen h  = _n - 1 
				gen b`SHOCK'`LHS'   = 0 
				gen b`SHOCK'`LHS'se = 0 
				gen b`SHOCK'`LHS'u  = 0 
				gen b`SHOCK'`LHS'l  = 0 	
				gen Nobs`SHOCK'`LHS'= . 
			tempfile EST_`SHOCK'_`LHS'_`j'
			save    `EST_`SHOCK'_`LHS'_`j''

			forv i=1/$H{
				use `dta_LV_comp',clear
				
				reghdfe Fd`i'`LHS'  `SHOCK' `X`j'' if smp_`LHS'==1  , absorb(ccode) cluster(ccode year)
				
				use `EST_`SHOCK'_`LHS'_`j'', clear
					replace b`SHOCK'`LHS'   = _b[`SHOCK']  if h ==`i'
					replace b`SHOCK'`LHS'se = _se[`SHOCK'] if h ==`i'
					replace b`SHOCK'`LHS'u  = b`SHOCK'`LHS' + 1.96*b`SHOCK'`LHS'se if h ==`i'
					replace b`SHOCK'`LHS'l  = b`SHOCK'`LHS' - 1.96*b`SHOCK'`LHS'se if h ==`i'
				replace Nobs`SHOCK'`LHS'= `e(N)'  if h ==`i'
				save `EST_`SHOCK'_`LHS'_`j'', replace			
			}
		}
		
		use          `EST_`SHOCK'_`LHS'_0', clear
		append using `EST_`SHOCK'_`LHS'_1'
		tempfile EST_`SHOCK'_`LHS'
		save    `EST_`SHOCK'_`LHS''
	}
}

use `EST_RC_y', clear
merge 1:1 j h using `EST_RC_credit_to_gdp', nogen
merge 1:1 j h using `EST_LV_y', nogen
merge 1:1 j h using `EST_LV_credit_to_gdp', nogen

sort j h


*** GDP
* controls
#delimit ;
twoway connected bRCy bRCyu bRCyl
				 bLVy bLVyu bLVyl	
				 h if j==1 & h<=6, color(blue blue blue red red red) m(s i i c i i ) lpattern(solid dot dot solid dot dot)
				 name(BVX_vs_LV_y,replace)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Real GDP Response")
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))   yscale(range(-.06(0.02)0.02))
				 legend(order(1 "BVX Crisis List" 4 "Laeven Valencia") rows(1));
#delimit cr

*** Credit
* controls
#delimit ;
twoway connected bRCcredit_to_gdp bRCcredit_to_gdpu bRCcredit_to_gdpl
				 bLVcredit_to_gdp bLVcredit_to_gdpu bLVcredit_to_gdpl	
				 h if j==1 & h<=6, color(blue blue blue red red red) m(s i i c i i ) lpattern(solid dot dot solid dot dot)
				 name(BVX_vs_LV_credit,replace)
				 yline(0, lcolor(black) lpattern(solid) lwidth(thin)) 
				 title("Credit-to-GDP Response")
				 xtitle("Years after shock") ylabel(#6, format(%03.2f))   yscale(range(-.06(0.02)0.02))
				 legend(order(1 "BVX Crisis List" 4 "Laeven Valencia") rows(1));
#delimit cr

grc1leg BVX_vs_LV_y BVX_vs_LV_credit, name(BVX_vs_LV,replace) altshrink xsize(4) ysize(2) 
graph display BVX_vs_LV, ysize(5.5) xsize(9.0)
graph export "$output/Appendix Figure 16B.pdf", replace as(pdf) 









/*------------------------------------------------------------------------

 Table A6: 
 Bank Equity Captures the Symptoms and Severity of Banking Crises

 Table A7: 
 Alternative Measures of Bank Equity Declines
 
-------------------------------------------------------------------------*/


**** Prepare Data

use "$datapath/BVX_annual_regdata.dta", clear
merge 1:1 country year using "$root/data/additional data/peak to trough measures.dta", keep(1 3) nogen
sort ccode year
xtset ccode year
replace log_rgdp_gr_pptdecline = -log_rgdp_gr_pptdecline

gen jointcrisis_3years = max(jointcrisis, L.jointcrisis, L2.jointcrisis)
sort ccode year
gen crisis_number = sum(jointcrisis)
recast str20 country

keep if jointcrisis_3years==1




collapse (firstnm) country year log_bank_ret_pktr log_abnormal_ret_pktr log_mktcap_ret_pktr log_rgdp_gr_pktr log_rgdp_gr_pptdecline ///
log_rgdp_gr_maxdeviation  log_bank_ret_recovery , by(crisis_number)


sort country year
foreach var1 in log_bank_ret_pktr log_abnormal_ret_pktr log_mktcap_ret_pktr log_rgdp_gr_pktr log_rgdp_gr_pptdecline ///
log_rgdp_gr_maxdeviation  log_bank_ret_recovery {
	replace `var1' = exp(`var1')-1
}
rename log_* *



replace bank_ret_recovery = -bank_ret_pktr if bank_ret_recovery>(-bank_ret_pktr)
gen prewar = (year<1945)
label var bank_ret_pktr "Bank equity decline"

merge 1:1 country year using "$root/data/additional data/crisis characteristics.dta", nogen

rename (significantliabilitiesguarantees significantliquiditysupport nationalizations  govtequityinjection ///
nplatpeak accountforwhatpercentofbankingas largestbankssignifinvolved lossindepositspktotr) ///
(sigliabguarantees sigliqsupport  nationaliz  govtequity ///
npl_peak failedbankspct largestbanks lossindeposits)



*--- drop war years ---*
drop if inrange(year,1914,1918) 
drop if inrange(year,1939,1945)




**** Table: Symptoms
cap est drop *
local j=1
foreach Y in lossindeposits lossindeposits lossindeposits failedbankspct largestbanks npl_peak sigliabguarantees ///
sigliqsupport nationaliz govtequity {
	eststo d_`j' : reg `Y' bank_ret_pktr, robust
	eststo m_`j' : reghdfe `Y' bank_ret_pktr, absorb(prewar) vce(robust)
	estadd local PW "\checkmark"
	local j=`j'+1
}

estout *, cells(b(star fmt(%7.3f)) t(par([ ]) fmt(%7.3f))) stats(r2 r2_a N, fmt(%5.3f %5.3f %9.0g) labels(R2 "Adj. R2" N)) starl(* 0.10 ** 0.05 *** 0.01) varlabels(`lhs') collabels(none) var(18) model(9)  mlabels(none)




#delimit ;
esttab m_2 m_4 m_5 m_6 using "$output/Appendix Table 6B-a.tex", 
	   replace compress b(a3) t(a3)  star(* 0.10 ** 0.05 *** 0.01 ) noconstant  brackets
	   mgroups("\shortstack{Decline in \\ deposits \\ (prewar \\ only)}" "\shortstack{Failed banks \\ (\% of total \\ bank assets)}"
	           "\shortstack{Largest \\ banks \\ failing}" "\shortstack{NPL at \\ peak}",
	          pattern(1 1 1 1 1 1 1 1 1 1 1) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
	    noobs booktabs  nonotes  nomtitles
	    scalars("PW Post-1945 dummy" "r2_a_within Adj. \(R^2 \) (within)" "N N")
		keep(bank_ret_pktr) order(bank_ret_pktr)
		label substitute(\_ _);
#delimit cr
#delimit ;
esttab m_7 m_8 m_9 m_10 using "$output/Appendix Table 6B-b.tex", 
	   replace compress b(a3) t(a3)  star(* 0.10 ** 0.05 *** 0.01 ) noconstant  brackets
	   mgroups( "\shortstack{Significant \\ liability \\ guarantees}" "\shortstack{Significant \\ liquidity \\ support}"
			   "\shortstack{Banks \\ nationalized}" "\shortstack{Govt \\ equity \\ injections}",
	          pattern(1 1 1 1 1 1 1 1 1 1 1) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
		noobs booktabs  nonotes  nomtitles
	    scalars("PW Post-1945 dummy" "r2_a_within Adj. \(R^2 \) (within)" "N N")
		keep(bank_ret_pktr) order(bank_ret_pktr)
		label substitute(\_ _);
#delimit cr


**** Table: Severity of Crises on Bank equity return peak to trough

gen smp = 1 if rgdp_gr_pktr~=. & rgdp_gr_pptdecline~=. & rgdp_gr_maxdeviation~=.
cap est drop *
foreach Y of varlist rgdp_gr_pktr rgdp_gr_pptdecline rgdp_gr_maxdeviation {
	eststo m_`Y':	reg     `Y' bank_ret_pktr if smp==1 ,  robust 
	eststo d_`Y': reghdfe `Y' bank_ret_pktr if smp==1 , absorb(prewar) vce(robust)
	estadd local PW "\checkmark"
}
sum bank_ret_pktr rgdp_gr_pktr if smp==1

#delimit ;
esttab d_* using "$output/Appendix Table 6A.tex", 
	   replace compress b(a3) t(a3)  star(* 0.10 ** 0.05 *** 0.01 ) noconstant  brackets
	   mgroups("\shortstack{Real GDP \\ (peak-to-trough \\ decline)}" "\shortstack{Real GDP growth  \\ (\%.-pt. decline, \\ peak-to-trough)}" "\shortstack{Real GDP growth \\ (max deviation \\ from trend)}",
	          pattern(1 1 1 ) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
	    noobs booktabs  nonotes  nomtitles
	    scalars("PW Post-1945 dummy" "r2_a_within Adj. \(R^2 \) (within)" "N N")
		keep(bank_ret_pktr) order(bank_ret_pktr)
		label substitute(\_ _);
#delimit cr





*** Appendix Table: Bank Abnormal Decline

cap est drop *
label var abnormal_ret_pktr "Abnormal bank decline"
foreach Y of varlist rgdp_gr_pktr rgdp_gr_pptdecline rgdp_gr_maxdeviation {
	eststo: reghdfe `Y' abnormal_ret_pktr if smp==1 , absorb(prewar) vce(robust)
	estadd local PW "\checkmark"
}
#delimit ;
esttab using "$output/Appendix Table 7A.tex", 
	   replace compress b(a3) t(a3)  star(* 0.10 ** 0.05 *** 0.01 ) noconstant  brackets
	   mgroups("\shortstack{Real GDP \\ (peak-to-trough \\ decline)}" "\shortstack{Real GDP growth  \\ (\%.-pt. decline, \\ peak-to-trough)}" "\shortstack{Real GDP growth \\ (max deviation \\ from trend)}",
	          pattern(1 1 1 ) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
	    noobs booktabs  nonotes  nomtitles
	    scalars("PW Post-1945 dummy" "r2_a_within Adj. \(R^2 \) (within)" "N N")
		keep(abnormal_ret_pktr) order(abnormal_ret_pktr)
		label substitute(\_ _);
#delimit cr





*** Appendix Table: Bank market cap decline

cap est drop *
label var mktcap_ret_pktr "Bank market cap decline"
foreach Y of varlist rgdp_gr_pktr rgdp_gr_pptdecline rgdp_gr_maxdeviation {
	eststo: reghdfe `Y' mktcap_ret_pktr if smp==1 , absorb(prewar) vce(robust)
	estadd local PW "\checkmark"
}
#delimit ;
esttab using "$output/Appendix Table 7B.tex", 
	   replace compress b(a3) t(a3)  star(* 0.10 ** 0.05 *** 0.01 ) noconstant  brackets
	   mgroups("\shortstack{Real GDP \\ (peak-to-trough \\ decline)}" "\shortstack{Real GDP growth  \\ (\%.-pt. decline, \\ peak-to-trough)}" "\shortstack{Real GDP growth \\ (max deviation \\ from trend)}",
	          pattern(1 1 1 ) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
	    noobs booktabs  nonotes  nomtitles
	    scalars("PW Post-1945 dummy" "r2_a_within Adj. \(R^2 \) (within)" "N N")
		keep(mktcap_ret_pktr) order(mktcap_ret_pktr)
		label substitute(\_ _);
#delimit cr


* Note for text: R2 is higher for bank market cap decline also when comparing on exact same sample
areg rgdp_gr_pktr mktcap_ret_pktr if smp==1 , absorb(prewar) r
areg rgdp_gr_pktr bank_ret_pktr   if e(sample), absorb(prewar) r
	
	

*** Appendix Table: Bank Equity Decline and Recovery

cap est drop *
label var bank_ret_recovery "Bank equity recovery" 
foreach Y of varlist rgdp_gr_pktr rgdp_gr_pptdecline rgdp_gr_maxdeviation {
	eststo: reghdfe `Y' bank_ret_pktr bank_ret_recovery if smp==1 , absorb(prewar) vce(robust)
	estadd local PW "\checkmark"
}
#delimit ;
esttab using "$output/Appendix Table 7C.tex", 
	   replace compress b(a3) t(a3)  star(* 0.10 ** 0.05 *** 0.01 ) noconstant  brackets
	   mgroups("\shortstack{Real GDP \\ (peak-to-trough \\ decline)}" "\shortstack{Real GDP growth  \\ (\%.-pt. decline, \\ peak-to-trough)}" "\shortstack{Real GDP growth \\ (max deviation \\ from trend)}",
	          pattern(1 1 1 ) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
	    noobs booktabs  nonotes  nomtitles
	    scalars("PW Post-1945 dummy" "r2_a_within Adj. \(R^2 \) (within)" "N N")
		keep(bank_ret_pktr bank_ret_recovery) order(bank_ret_pktr bank_ret_recovery)
		label substitute(\_ _);
#delimit cr







/*-----------------------------------------------------------------------
 
 Table A8: 
 Impact of Bank Equity Crashes Outside of Narrative Crises
 
-------------------------------------------------------------------------*/

use "$datapath/BVX_annual_regdata.dta", clear
xtset ccode year
tsfill
set matsize 5000

* bank equity crash variables
gen R_B = C_B30        
gen R_N = C_N30  

* crisis variable
foreach X of varlist JC {
	gen `X'_adj = `X'
	replace `X'_adj = 1 if L3.`X'==1 | L2.`X'==1 | L.`X'==1 | F.`X'==1 | F2.`X'==1 | F3.`X'==1
}
drop JC
ren JC_adj JC  
gen R_BxJC = R_B*JC

label var R_B "Bank equity crash"
label var R_BxJC "Bank eq. crash \( \times \) Narrative crisis"
label var JC    "Narrative crisis"
label var R_N "Nonfinancial equity crash"
label var C_B30 "Bank equity crash"
label var C_N30 "Nonfinancial equity crash"

gen g0d_y = g0credit_to_gdp

cap est drop *

global X1 "L(1/3).R_B L(1/3).JC L(1/3).R_N L(0/3).g0y  L(0/3).g0d_y"


reghdfe Fd3y R_BxJC C_B30 JC C_N30 $X1, absorb(ccode year) cluster(ccode year)
gen smp=e(sample)

* 1
eststo: reghdfe Fd1y C_B30 C_N30 JC $X1 if smp==1, absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
	//xtsccfixedb Fd1y C_B30 C_N30 JC $X1 if smp==1, fe lag($L)
estadd local CFE  "\checkmark"
estadd local cont "\checkmark"
estadd local aR2 = "`aR2'" 
* 2
eststo: reghdfe Fd1y R_BxJC C_B30 JC C_N30 $X1 if smp==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
	//xtsccfixedb Fd1y R_BxJC C_B30 JC C_N30 $X1 if smp==1 , fe lag($L)
estadd local CFE  "\checkmark"
estadd local cont "\checkmark"
estadd local aR2 = "`aR2'" 
* 3
eststo: reghdfe Fd3y C_B30 C_N30 JC $X1 if smp==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
//xtsccfixedb Fd3y C_B30 C_N30 JC $X1 if smp==1 , fe lag($L)
estadd local CFE  "\checkmark"
estadd local cont "\checkmark"
estadd local aR2 = "`aR2'" 
* 4
eststo: reghdfe Fd3y R_BxJC C_B30 JC C_N30 $X1 if smp==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
//xtsccfixedb Fd3y R_BxJC C_B30 JC C_N30 $X1 if smp==1 , fe lag($L)
estadd local CFE  "\checkmark"
estadd local cont "\checkmark"
estadd local aR2 = "`aR2'" 

#delimit ;
esttab using "$output/Appendix Table 8A.tex", 
	    replace compress b(a2) t(a2)  star(* 0.10 ** 0.05 *** 0.01 ) noconstant  brackets
	   mgroups("\shortstack{Real GDP \\ \(\text{growth}_{t,t+1}\)}" "\shortstack{Real GDP \\ \(\text{growth}_{t,t+3}\)}",
	          pattern(1 0 1 0 ) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
	    noobs booktabs  nonotes  nomtitles
	    scalars("CFE Country fixed effects" "cont Controls" "aR2 Adj. \(R^2 \) (within)" "N N")
		keep(C_B30 JC R_BxJC C_N30) order(C_B30 JC R_BxJC C_N30)
		label substitute(\_ _);
#delimit cr

cap est drop *
reghdfe Fd3credit_to_gdp R_BxJC C_B30 JC C_N30 $X1, absorb(ccode) cluster(ccode year)
gen smp2=e(sample)

* 1
eststo: reghdfe Fd1credit_to_gdp C_B30 C_N30 JC $X1 if smp2==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
//xtsccfixedb Fd1credit_to_gdp C_B30 C_N30 JC $X1 if smp2==1 , fe lag($L)
estadd local CFE  "\checkmark"
estadd local cont "\checkmark"
estadd local aR2 = "`aR2'" 
* 2
eststo: reghdfe Fd1credit_to_gdp R_BxJC C_B30 JC C_N30 $X1 if smp2==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
//xtsccfixedb Fd1credit_to_gdp R_BxJC C_B30 JC C_N30 $X1 if smp2==1 , fe lag($L)
estadd local CFE  "\checkmark"
estadd local cont "\checkmark"
estadd local aR2 = "`aR2'" 
* 3
eststo: reghdfe Fd3credit_to_gdp C_B30 C_N30 JC $X1 if smp2==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
//xtsccfixedb Fd3credit_to_gdp C_B30 C_N30 JC $X1 if smp2==1 , fe lag($L)
estadd local CFE  "\checkmark"
estadd local cont "\checkmark"
estadd local aR2 = "`aR2'" 
* 4
eststo: reghdfe Fd3credit_to_gdp R_BxJC C_B30 JC C_N30 $X1 if smp2==1 , absorb(ccode) cluster(ccode year)
local aR2: display %5.2f e(r2_a_within)
//xtsccfixedb Fd3credit_to_gdp R_BxJC C_B30 JC C_N30 $X1 if smp2==1 , fe lag($L)
estadd local CFE  "\checkmark"
estadd local cont "\checkmark"
estadd local aR2 = "`aR2'" 


#delimit ;
esttab using "$output/Appendix Table 8B.tex", 
	    replace compress b(a2) t(a2) star(* 0.10 ** 0.05 *** 0.01 ) noconstant  brackets
	   mgroups("\shortstack{Credit/GDP \\ \(\text{change}_{t,t+1}\)}" "\shortstack{Credit/GDP \\ \(\text{change}_{t,t+3}\)}",
	          pattern(1 0 1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}) )		
	    noobs booktabs  nonotes  nomtitles
	    scalars("CFE Country fixed effects" "cont Controls"  "aR2 Adj. \(R^2 \) (within)" "N N")
		keep(C_B30 JC R_BxJC C_N30) order(C_B30 JC R_BxJC C_N30)
		label substitute(\_ _);
#delimit cr









/*------------------------------------------------------------------------

 Table A14: 
 Area Under the ROC Curve for BVX Crises and Other Crisis Chronologies
 
-------------------------------------------------------------------------*/

*** Comparison with Reinhart Rogoff

* Preallocate results table
clear
set obs 15
gen obs =_n
gen iclass=.
gen AUC = .
gen SE    = .
gen classifier=""
gen chronology=""
tempfile RESULTS
save    `RESULTS'
local j=1
local k=0

use "$datapath/BVX_annual_regdata.dta", clear

gen m_g0y = - g0y // negative annual real GDP growth
gen m_R_B = - Rtot_real_w // negative bank equity returns
gen m_R_N = - Rtot_nonfin_real_w // negative nonfin equity returns
gen m_Fd5cr= -(F5.credit_to_gdp - credit_to_gdp) // credit contraction: t to t+5

local classifiers m_g0y m_R_B m_R_N m_Fd5cr  

gen LV = (LaevenValenciaCrisis==1) if year>=1970
gen RR = (ReinhartRogoffCrisis==1)

tempfile AUC_DTA
save    `AUC_DTA'


* Full sample
foreach X in `classifiers' {
	local k =`k'+1
	foreach Chron in RC RR  {
		
		* Calculate ROC
		use `AUC_DTA', clear
		roctab `Chron' `X',
		
		* Save results
		use `RESULTS', clear
		replace classifier = "`X'"  if obs == `j'
		replace iclass     = `k'    if obs ==`j'
		replace chronology = "`Chron'" if obs==`j'
		replace AUC = `r(area)' if obs == `j'
		replace SE    = `r(se)'   if obs == `j'
		local j = `j'+1
		save `RESULTS', replace
	}
}

use `RESULTS', clear
drop obs
drop if SE==.
replace AUC = round(AUC,0.01)
replace SE = round(SE,0.01)
format AUC %7.3f
format SE    %7.3f

ren AUC AUC_
ren SE SE_
reshape wide  AUC_ SE_, i(iclass classifier) j(chronology) string

gsort iclass
drop iclass

replace classifier = "Bank eq. return, \(t-1\) to \(t\)"       if classifier == "m_R_B"
replace classifier = "Nonfin. eq. return, \(t-1\) to \(t\)"    if classifier == "m_R_N"
replace classifier = "Real GDP growth, \(t-1\) to \(t\)"       if classifier == "m_g0y"
replace classifier = "Credit-to-GDP change, \(t\) to \(t+5\)"  if classifier == "m_Fd5cr"

label var AUC_RC "AUC - BVX Crisis" 
label var SE_RC  "se(AUC) - BVX Crisis" 
label var AUC_RR "AUC - Reinhart-Rogoff" 
label var SE_RR "se(AUC) - Reinhart-Rogoff" 

order classifier AUC_RC SE_RC 


texsave using "$output/Appendix Table 14A.tex", ///
replace varlabels frag  ///
size(small) align(l C C C C ) width(6in)


*** Comparison with Laeven Valencia

* Preallocate results table
clear
set obs 15
gen obs =_n
gen iclass=.
gen AUC = .
gen SE    = .
gen classifier=""
gen chronology=""
tempfile RESULTS
save    `RESULTS'
local j=1
local k=0

use `AUC_DTA', clear
keep if year>=1970
save    `AUC_DTA', replace


* Full sample
foreach X in `classifiers' {
	local k =`k'+1
	foreach Chron in RC LV  {
		
		* Calculate ROC
		use `AUC_DTA', clear
		roctab `Chron' `X',
		
		* Save results
		use `RESULTS', clear
		replace classifier = "`X'"  if obs == `j'
		replace iclass     = `k'    if obs ==`j'
		replace chronology = "`Chron'" if obs==`j'
		replace AUC = `r(area)' if obs == `j'
		replace SE    = `r(se)'   if obs == `j'
		local j = `j'+1
		save `RESULTS', replace
	}
}

use `RESULTS', clear
drop obs
drop if SE==.
replace AUC = round(AUC,0.01)
replace SE = round(SE,0.01)
format AUC %7.3f
format SE    %7.3f

ren AUC AUC_
ren SE SE_
reshape wide  AUC_ SE_, i(iclass classifier) j(chronology) string

gsort iclass
drop iclass

replace classifier = "Bank eq. return, \(t-1\) to \(t\)"       if classifier == "m_R_B"
replace classifier = "Nonfin. eq. return, \(t-1\) to \(t\)"    if classifier == "m_R_N"
replace classifier = "Real GDP growth, \(t-1\) to \(t\)"       if classifier == "m_g0y"
replace classifier = "Credit-to-GDP change, \(t\) to \(t+5\)"  if classifier == "m_Fd5cr"

label var AUC_RC "AUC - BVX Crisis" 
label var SE_RC  "se(AUC) - BVX Crisis" 
label var AUC_LV "AUC - Laeven-Valencia" 
label var SE_LV "se(AUC) - Laeven-Valencia" 

order classifier AUC_RC SE_RC 


texsave using "$output/Appendix Table 14B.tex", ///
replace varlabels frag  ///
size(small) align(l C C C C ) width(6in)



