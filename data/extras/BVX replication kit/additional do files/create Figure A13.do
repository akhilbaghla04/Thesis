
/*======================================================================

	Create Figure A13
	
	Robustness analysis for event studies: around Narrative Crises
	
	
	Note: need to first run the code "create Tables A9 and A10.do" before running this file

========================================================================*/



quietly {


use "$root/data/additional data/Timing of stocks and bonds.dta", clear
drop if peak_B_year==.
drop if narrative_year==.
replace narrative_month = 1
save "$root/data/temp/Timing of stocks and bonds - peak_B only - around Narrative Crises", replace



*/

/*------------------------------------------------
 Construct Event Study Dataset
--------------------------------------------------*/

global H = 36    // event study window


use "$root/data/master_data_monthly.dta", clear

gen panic_tm = tm
merge 1:1 country panic_tm using "$root/data/temp/panic_dates_narrative"
gen panic_now = 1 if _merge==3
sort ccode tm
drop panic_year panic_month _merge

global vars     sC sB  // compute average y(t) - y(0)
global cumvars  rB rN     // computed cumulated return from t=0
global lev_vars panic_now    // compute average of level y(t)

gen date = dofm(tm)
gen year = year(date)



drop date year


* variable transformations
foreach X in $cumvars {	
	gen `X'_F1  = F1.`X'
	gen `X'_L0  = 0
	gen `X'_L1  = - `X'
	forv t=2/$H {
		local tm1 = `t'-1
		gen `X'_F`t' = `X'_F`tm1' + F`t'.`X'  
		gen `X'_L`t' = `X'_L`tm1' - L`tm1'.`X'
	}
}

foreach X in $vars {	
	forv t=1/$H {
		gen `X'_F`t' = F`t'.`X' -      `X'
		gen `X'_L`t' =    -(`X' - L`t'.`X')
	}
}

foreach X in $lev_vars {	
		gen `X'_L0   = `X'
	forv t=1/$H {
		gen `X'_F`t' = F`t'.`X'
		gen `X'_L`t' = L`t'.`X'
	}
}

save "$root/data/temp/master_data_monthly - with lags - narrative crises", replace
*/



use "$root/data/temp/master_data_monthly - with lags - narrative crises", clear

drop year_narrative

gen date = dofm(tm)
gen year = year(date)
gen month = month(date)
drop if year<1870
drop date
rename (year month) (narrative_year narrative_month)


merge 1:1 country narrative_year narrative_month using "$root/data/temp/Timing of stocks and bonds - peak_B only - around Narrative Crises"

*** keep only events on the narrative list
keep if _merge==3


*** keep only events with a bank stock decline
foreach var1 in rB rN sB sC {
	replace `var1' = . if peak_B_year==.
	forval nn = 1/$H {
		replace `var1'_F`nn'=. if peak_B_year==.
		replace `var1'_L`nn'=. if peak_B_year==.
	} 
}


*** keep only bank credit spread data if trough is marked 
replace sB = . if trough_bankcredit_year!=.
forval nn = 1/$H {
	replace sB_F`nn'=. if trough_bankcredit_year==.
	replace sB_L`nn'=. if trough_bankcredit_year==.
}

*** keep only corp credit spread data if trough is marked 
replace sC = . if trough_corpcredit_year!=.
forval nn = 1/$H {
	replace sC_F`nn'=. if trough_corpcredit_year==.
	replace sC_L`nn'=. if trough_corpcredit_year==.
}


* drop extreme returns due to hyperinflation episodes
drop if country=="Germany" & tm==ym(1925,1) 
drop if country=="Chile" & tm==ym(1976,1)


rename (narrative_year narrative_month) (year month)
keep country year month tm *_F* *_L* trough_bankcredit*

save  "$root/data/temp/tmp_event_all_MB - narrative crises", replace




*/





foreach panel1 in "A" "B" "C" "D" "E" {

	use "$root/data/temp/tmp_event_all_MB - narrative crises", clear

	egen panic_anytime = rowtotal(panic_now_*), missing
	order panic_anytime

	forv t=1/$H {
		replace panic_now_L`t' = 0 if panic_now_L`t'==. & panic_anytime==1
		replace panic_now_F`t' = 0 if panic_now_F`t'==. & panic_anytime==1

		
		foreach X in sC sB {
			drop if inrange(`X'_L`t',0.50,9999) | `X'_L`t'<-0.10
			drop if inrange(`X'_F`t',0.50,9999)	| `X'_F`t'<-0.10	
		}
	}


	replace panic_now_L0 = 0 if panic_now_L0==. & panic_anytime==1

	****** SUBSAMPLES **************	
	if "`panel1'"=="B" drop if sB_L1==.
	if "`panel1'"=="C" keep if inrange(year,1870,1939)
	if "`panel1'"=="D" keep if inrange(year,1940,2016)
	if "`panel1'"=="E" keep if inrange(year,1940,2006)
	

	count
	global Ncrisis_max `r(N)'
	count if rB_L0!=.
	global N_bankstock `r(N)'
	count if rN_L0!=.
	global N_nonfinstock `r(N)'
	count if sB_L1!=.
	global N_bankspreads `r(N)'
	count if sC_L1!=.
	global N_corpspreads `r(N)'
	count if panic_now_L1!=.
	global N_panic_now `r(N)'


	tempfile tmp_event_Nmax
	save    `tmp_event_Nmax'



	global vars     sC sB 		// compute average y(t) - y(0)
	global cumvars  rB rN       // computed cumulated return from t=0
	global lev_vars panic_now   // compute average of level y(t)

	foreach X in $lev_vars $vars $cumvars {	
		use `tmp_event_Nmax', clear
		keep `X'_F* `X'_L*
		collapse `X'_*
		gen tmp=1
		reshape long `X'_F `X'_L , i(tmp) j(lead)
		reshape long `X'_ , i(lead) j(sign) string
		replace tmp = -1 if sign=="L"
		gen t = lead*tmp
		drop if sign=="F" & lead==0
		drop lead tmp sign
		ren `X'_ `X'
		dis in red "`X'"
		tempfile mean_`X'
		save    `mean_`X''
	}

	use `mean_rB', clear
	foreach X in $vars $cumvars $lev_vars {	
		merge 1:1 t using `mean_`X'', nogen
	}
	foreach X in $vars   {	
		replace `X' = 0 if t==0
	}

	sort t
	replace panic_now = panic_now/20

	#delimit ;
	twoway line rB rN t , xline(0, lcolor(black) lpattern(solid) lwidth(thin))  
		lcolor(blue red) lpattern(solid dash) 
		xtitle("Event time (months)") xlabel(-36(12)36) 
		legend(order(1 "Bank equity (N = $N_bankstock)" 3  "Bank credit spread (N = $N_bankspreads)" 
		2  "Nonfinancial equity (N = $N_nonfinstock)" 4 "Corporate credit spread (N = $N_corpspreads)" 5 "Start of panic (N = $N_panic_now)")) ||
		   line sB sC panic_now t, yline(0, lcolor(black) lpattern(solid) lwidth(thin) axis(2)) lcolor(gs10 gs10 black) lpattern(solid shortdash solid) lwidth(. . medthick)
		ytitle("", axis(2)) graphregion(color(white)) yaxis(2) name(fullsample,replace);
	#delimit cr
	gr_edit .legend.plotregion1.key[2].DragBy 0 -1
	gr_edit .legend.plotregion1.label[2].DragBy 0 -1
	gr_edit .legend.plotregion1.key[4].DragBy 0 -1
	gr_edit .legend.plotregion1.label[4].DragBy 0 -1
	gr_edit .legend.plotregion1.key[5].DragBy 0 33
	gr_edit .legend.plotregion1.label[5].DragBy 0 33
	graph export "$root/output/Appendix Figure 13`panel1'.pdf", replace
	

}


*/





}
