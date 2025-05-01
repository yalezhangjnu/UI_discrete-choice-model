**********************************************************************************
* File name:    lobby_clean.do
* Description:  This file is used to deal with lobbying raw dataset from opensecrets.org websites
* Date:         2024.1.22
* Author：      Yi Zhang
**********************************************************************************
* Setup global enviroment for lobbying data cleaning
* For replication purpose, please change this section corresponding working directory to the local path.
clear all
global base_path "C:\Users\YI\Desktop\RobustLR1030"  
cd "$base_path\Lobbydata\stata"
*************************************************
***************bank lobby expenditure************
*1.Transfer all .csv data files to .dta data files from 2008 to 2019
forvalues i=2008/2019{
	import delimited "$base_path\Lobbydata\stata\bank lobby expenditure\lobbyexpenditure`i'.csv",clear
rename total total`i'
save lobbyexpenditure`i',replace
}
*2.Merge datasets(2008-2019)
use lobbyexpenditure2008,clear
forvalues i=2009/2019{
	merge 1:1 clientparent using lobbyexpenditure`i',nogen
}
*Use "merge" instead of "append" because "merge" can generate missing values in years 
*when there is no expenditure data, which ensures tha all years are included in the resulting data.
*We do not need to keep subsidiaryaffiliate in current data set
drop subsidiaryaffiliate
* reshape dataset to what we need
reshape long total,i(clientparent) j(year) 
replace total =substr(total,2,.)
destring total,replace 
* save the merge dataset
cd "$base_path\Lobbydata\stata"
save lobbyexpenditure08-19,replace

************************************************
***************bank lobby represents************
*1.Merge datasets and transfer the .csv data to .dta data(1998-2019)
global d "$base_path\Lobbydata\stata\bank lobby represents"
cd "$d"
qui rcd
return list
* compress all lobby represents' files together
local k = 1998
forvalues i = 2(1)`r(tdirs)'{
	cd "$d"
	qui rcd
	cd "`r(ndir`i')'"
	local ff: dir . files "*.csv",respectcase 
	foreach f in `ff'{
		import delimited "`f'",varnames(1) clear
		
		rename client client_clue
		
		gen client = substr("`f'",24,.)
		replace client = substr(client,1,strlen(client)-4) 
		
		gen year = `k' 
		save "$base_path\Lobbydata\stata\represents_clean\_`f'`k'.dta",replace
		
	}
	
	local k = `k'+1
	
}	
	
clear	
cd "$base_path\Lobbydata\stata\represents_clean"
local ff: dir . file "*.dta" 
foreach f in `ff'{
append using "`f'",force
} 
*extract a substring from the variable 'totalamount' and replace all occurrences of commas in the variable totalamount with an empty string.
replace totalamount =substr(totalamount,2,.)
replace totalamount = subinstr(totalamount,",","",.) 
destring totalamount,replace 
*create the bankrepresents 98-19
replace client_clue = lobbyingfirmhired if missing(client_clue) 
order client client_clue year 
sort client year
save "$base_path\Lobbydata\stata\represents98-19",replace
export delimited "$base_path\Lobbydata\stata\represents98-19.csv", replace

* keep the bank represents record after 2008
keep if year >= 2008
sort client year
save "$base_path\Lobbydata\stata\represents08-19",replace
export delimited "$base_path\Lobbydata\stata\represents08-19.csv", replace

*************************************************
***use expenditure data and represents data to double check active lobby clients****

cd "$base_path\Lobbydata\stata"
use lobbyexpenditure08-19,clear
rename clientparent client
merge 1:m client year using represents08-19
sort client year

gen lobby_status = 1  if (!missing(total) & total != 0 | !missing(lobbyingfirmhired)) 
replace lobby_status = 0 if mi(lobby_status)
*A dummy variable equal to one if bank i is active in lobbying during the year t, and zero otherwise."Active" means that the bank has at least hired once a lobbying firm or filed a lobbying report.
save lobby_repre08-19,replace
export delimited lobby_repre08-19.csv, replace
***generate unique lobby bank list between 08-19***
use "$base_path\Lobbydata\stata\lobby_repre08-19",clear
keep if lobby_status == 1
duplicates drop client,force
sort client
keep client
save "lobby_id0819", replace
export delimited lobby_id0819.csv, replace

*** Next we need to check mannually in capitalIQ pro platform, we update each parent ID in the file with unique ID.
*** The file with full name and MIKEY is store as file: lobby_id0819.

*************************************************
*******Get lobby experience from bank list*******

*1.Merge datasets and transfer the .csv data to .dta data(1998-2019)

cd "$base_path\Lobbydata\stata\experience"

forvalues i=1998/2019{
	import delimited "$base_path\Lobbydata\rawdata1030\bank list\bank list `i'.csv",clear
rename total total`i'
save list`i',replace

}


use  list1998,clear
forvalues i=1999/2019{
	merge 1:1 clientparent using list`i',nogen
}

drop subsidiaryaffiliate
reshape long total,i(clientparent) j(year) 
rename clientparent client
replace total =substr(total,2,.)
destring total,replace 

cd "$base_path\Lobbydata\stata"
save list98-19,replace

*2.Merge "list98-19" and "represents98-19"
cd "$base_path\Lobbydata\stata"
use list98-19,clear

merge 1:m client year using "$base_path\Lobbydata\stata\represents98-19"

sort client year
order client year

gen lobby_status = 1  if (!missing(total) | !missing(lobbyingfirmhired)) 
replace lobby_status = 0 if mi(lobby_status)
egen cl_year = tag(client year) 


*3.Calculate experience

bys client: gen temp_status = _n if lobby_status == 1 
bys client: egen m = min(temp_status)
bys client: gen earliest_year = year[m]
drop m temp_status

gen experience = year - earliest_year
replace experience = 0 if experience <0


duplicates drop client year,force 
cd "$base_path\Lobbydata\stata"
save experience98-19,replace


**** Merge "experience" into "expend_repre08-19" ****

cd "$base_path\Lobbydata\stata"
use lobby_repre08-19,clear

merge m:1 client year using experience98-19,keepusing(experience) gen(merge2)

drop if year<2008

drop if merge == 2


*************************************************
*1.revolvingdoorprofiles："Revolving Door Profiles"=1
replace revolvingdoorprofiles = "1" if revolvingdoorprofiles == "Revolving Door Profiles"
replace revolvingdoorprofiles = "0" if revolvingdoorprofiles == "No Revolving Door Profiles"

destring revolvingdoorprofiles,replace

*2.formermembersofcongress："Former Members of Congress"=1
replace formermembersofcongress = "1" if formermembersofcongress == "Former Members of Congress"
replace formermembersofcongress = "0" if formermembersofcongress == "Non Former Members of Congress"

destring formermembersofcongress,replace
save expend_repre_exper08-19 ,replace
export delimited expend_repre_exper08-19.csv, replace


*************************************************
********Targeted lobbying on regulators******

*1.Transfer the .csv data to .dta data
cd "$base_path\Lobbydata\rawdata1030\Targeted lobbying on regulators"

local a FDIC Fed OCC
forvalues i=2008/2019{
	foreach v in `a'{
		import delimited "`i'\Clients Lobbying `v'.csv",clear
		gen year = `i'
		gen target = "`v'"

		save "$base_path\Lobbydata\stata\Target\target`i'`v'",replace
	}
}

*2.Merge datasets
cd "$base_path\Lobbydata\stata\Target"
openall * 
order client year
sort client year

duplicates drop client year,force
replace target = "1"
destring target,replace

cd "$base_path\Lobbydata\stata"
save target08-19,replace
export delimited target0819.csv, replace


***** Merge "target" into "expend_repre_exper08-19" ****

cd "$base_path\Lobbydata\stata"
use expend_repre_exper08-19,clear

merge m:1 client year using target08-19,keepusing(target) keep(1 3) nogen
replace target = 0 if mi(target)

save finaldata08-19,replace
export delimited finaldata0819.csv, replace
****** generate finalversion of data ************
drop if _merge==1
drop _merge 
drop merge2
export delimited lobby0819test.csv, replace








