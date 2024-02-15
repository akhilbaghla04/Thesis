

/*======================================================================

	Create Figure 5
	
	U.S. 2008 plot

========================================================================*/


use "$root/data/additional data/US monthly data.dta", clear

local range1 = "2004, 2012"
replace baa = baa/100
replace LIBOR_OIS = LIBOR_OIS/100

replace cumul_bank_equity_return = exp(cumul_bank_equity_return)-1
replace cumul_nonfin_equity_return = exp(cumul_nonfin_equity_return)-1

gen date2 = year + (month-1)/12
label variable date2 "year"
label variable cumul_nonfin_equity_return "Nonfin equity index"
label variable cumul_bank_equity_return "Bank equity index"

twoway line cumul_nonfin_equity_return cumul_bank_equity_return  date2 if inrange(year,`range1'), yline(0, lcolor(k) lw(vthin)) ///
ylabel(-1.5(0.25)0) ytitle("total returns index") lcolor(red blue) || ///
line nonfin_bond_spread baa LIBOR_OIS date2 if inrange(year,`range1'), xlabel(2004(1)2013) ///
yaxis(2) ylabel(0(0.01)0.08, axis(2)) yline(0, lcolor(k) lw(vthin) axis(2)) ytitle("bond spread", axis(2)) name(p1) legend(r(2)) ///
lcolor (black black green) lp(- . .)

graph export "$root/output/Figure 5.pdf", replace
