**** File Name:     dataprocess040925.do
**** Description:   This file is used to replicate the final dataset we need for Kaido and Zhang (2024)
**** Author:        Yi Zhang
**** Date:          2024/08/05  final check: 2025/04/09
**** Data resource: 1. Capital IQ pro 2. opensecret.org/lobby/Commecial Banks 3. BEA for per capita personal income

********** Part 0: Setworking directory **********
clear all
global base_path "C:\Users\YI\Desktop\RobustLR1030\Bankdata"
global raw_data_path "$base_path\raw_data"
global processed_data_path "$base_path\replicate_data"
global result_path "$base_path\results"
cd "$base_path"

********** PART 1:  Bankdata: For Financial and Demographic Covariates **************
********** This part is corresponding to data we explain in Online Appendix D.4 *****
import excel "$raw_data_path\bank0720.xlsx", sheet("Sheet1") firstrow clear
tostring REG_SNL_INSTN_KEY , replace
gen status = 1
sort REG_SNL_INSTN_KEY year

********** get three processed data we need to use: mf_bank_id, mf_bank_id_year, mf_bank_zipcode
preserve
keep REG_SNL_INSTN_KEY year status TotalAssets000 Initial_Scale ZipCode
save "$processed_data_path\mf_bank_id_year",replace
keep REG_SNL_INSTN_KEY status
duplicates drop REG_SNL_INSTN_KEY, force
save "$processed_data_path\mf_bank_id",replace
restore

preserve
duplicates drop REG_SNL_INSTN_KEY,force
drop if ZipCode==""
keep REG_SNL_INSTN_KEY ZipCode 
export excel using "$processed_data_path/mf_bank_zipcode.xlsx", firstrow(variables) replace
restore

preserve
keep ParentInstitutionKey year status TotalAssets000 Initial_Scale ZipCode
rename ParentInstitutionKey REG_SNL_INSTN_KEY
save "$processed_data_path\mf_bank_id_year_Parent",replace
keep REG_SNL_INSTN_KEY status
duplicates drop REG_SNL_INSTN_KEY,force
save "$processed_data_path\mf_bank_id_Parent",replace
*restore the original dataset to its initial state
restore

*********** setup criterion for sample selection*******************
drop if CompanyType == "Savings Bank"
*drop if TotalAssets000 == . this two lines are shifted to 372-373
*drop if Initial_Scale==.

save "$processed_data_path\mf_bank_data.dta", replace
export delimited "$processed_data_path\mf_bank_data.csv", replace

******** risk-taking variables is not used for this application: cash out all codes ********************
* Z-score and ROAA volatility
*bys REG_SNL_INSTN_KEY : gen a1 = ROAA[_n-1]
*bys REG_SNL_INSTN_KEY : gen a2 = ROAA[_n+1]
*egen roaa_std=rowsd(ROAA a1 a2)
*gen z_score = (ROAA + GRBTotalEquityCapital000/TotalAssets000)/roaa_std
*replace z_score = ln(z_score+1)

* Unuse commitment_growth
*bys REG_SNL_INSTN_KEY : gen u1 = OffBSTotUnusedCommitments[_n-1]
*gen unused_commitment_growth = 100*(OffBSTotUnusedCommitments - u1)/u1

* Loan_growth
*gen total_loan = ConTotalRealEstateLoans0+ConTotCommIndLoans000+ConTotConsumerLoans000
*bys REG_SNL_INSTN_KEY : gen l1 = total_loan[_n-1]
*gen loan_growth = 100*(total_loan - l1)/l1

* Nonperforming_loans ratio
*gen nonperforming_loans = 100*(PDTotalLoans000 + NAccrTotalLoans000)/total_loan

* Non-acrual loans ratio
*gen nonaccural_loans = 100*NAccrTotalLoans000/total_loan


******** generate financial variables used in W_i ********************
* Capital Adequacy: multiply 100 as percentage measure, original numerator and denominar need to multiply 1000
gen capital_adequacy = 100*Tier1Capital000/RiskWeightedAssets000
* Asset quality: this one does not need to multiply 100, data is directly downloaded with percentage as unit.
gen asset_quality = -LoanandLeaseAllowanceTotal
* Earning: previous version forget to multiply 100, it is the ratio of netinterstincome/totalassets
gen earning = NetInterestIncome000/TotalAssets000
* Mangement Quality needs to be generated based on the number of enforcement actions
* It is calculated in part 2 
* Liquidity: multiply by 100 as percentage measure
gen liquidity = 100*TotalCashBalsDueDepInst/TotalDepositsInclDomFor
* Sensitivity to market risk: NetShorttermLiabilitiesAsse is a percentage measure,therefore, this tranformation is percentage measure.
gen sensitivity_to_market_risk = abs(NetShorttermLiabilitiesAsse)*TotalAssets000/TOTAL_EARN_ASSETS

******** generate financial variables used in W_i ********************
*Deposit to asset ratio: multiply by 100 as percentage measure
gen deposit_to_asset_ratio = 100*TotalDepositsInclDomFor/TotalAssets000
*Leverage: this is directly downloaded from database as percentage measure
gen leverage = LeverageRatio
*Total Core Deposit 000usd
gen total_core_deposit = REG_CORE_DEP
*Size 000usd
gen size = TotalAssets000
*Age years
gen age= year - YearEstablished

global bank_num "capital_adequacy asset_quality earning liquidity sensitivity_to_market_risk deposit_to_asset_ratio leverage  total_core_deposit size age"

global bank_char "REG_SNL_INSTN_KEY year CompanyName CompanyType PrimaryRegulator YearEstablished ParentName ParentInstitutionKey CountyandState ZipCode State Initial_Scale status"

save "$processed_data_path\mf_bankvar_clean",replace
export delimited "$processed_data_path\mf_bankvar_clean.csv", replace

**************Part 2： enforcement action sample data *********************************
********** This part is corresponding to data we exlained in Online Appendix D.2 *****

clear all
import excel "$raw_data_path\enforcement0519.xlsx", firstrow clear
sort KeyInstitution IssueDate
rename KeyInstitution REG_SNL_INSTN_KEY
tostring REG_SNL_INSTN_KEY , replace
gen year = year( IssueDate )
************** drop observatrions that are not belongs to commercial banks*************
drop if InstitutionType == "Credit Union"|InstitutionType == "Savings & Loan Assoc"|InstitutionType == "Savings & Loan HC"|InstitutionType == "Savings Bank"
merge m:1 REG_SNL_INSTN_KEY year using "$processed_data_path\mf_bank_id_year"
************** keep different merge types ****************************
************** store record from enforcement file that is not matched with bank data ***********************************
preserve
keep REG_SNL_INSTN_KEY InstitutionName RegulatoryAgency InstitutionType Current Regactiontype IssueDate ModificationDate TerminationDate year status _merge TotalAssets000 Initial_Scale ZipCode
keep if _merge==1
sort year
export excel using "$processed_data_path\mf_unmatch.xlsx", firstrow(variables) replace
restore

preserve
keep REG_SNL_INSTN_KEY InstitutionName RegulatoryAgency InstitutionType Current Regactiontype IssueDate ModificationDate TerminationDate year status _merge TotalAssets000 Initial_Scale ZipCode
keep if _merge==3
sort year
export excel using "$processed_data_path\mf_match.xlsx", firstrow(variables) replace
restore

clear
import excel "$processed_data_path\\mf_unmatch.xlsx", sheet("Sheet1") firstrow clear 
drop _merge TotalAssets000 Initial_Scale ZipCode
merge m:m REG_SNL_INSTN_KEY year using "$processed_data_path\mf_bank_id_year_Parent"
keep if _merge==3
duplicates drop REG_SNL_INSTN_KEY,force
replace status=1
drop _merge
save "$processed_data_path\mf_data_Parent.dta", replace
export delimited "$processed_data_path\mf_data_Parent.csv", replace

****divide the unmatch into two part****
import excel "$processed_data_path\\mf_unmatch.xlsx", sheet("Sheet1") firstrow clear 
drop _merge TotalAssets000 Initial_Scale ZipCode
merge m:1 REG_SNL_INSTN_KEY year Regactiontype using "$processed_data_path\mf_data_Parent"

preserve
drop if _merge==3
drop _merge
save "$processed_data_path\mf_unmatch_unmatch",replace
restore

preserve
drop if _merge==1
drop _merge
replace status=1
save "$processed_data_path\mf_unmatch_match",replace
restore

import excel "$processed_data_path\mf_match.xlsx", firstrow clear
append using  "$processed_data_path\mf_unmatch_unmatch"
append using  "$processed_data_path\mf_unmatch_match"
gen enforcement_status = 1
save "$processed_data_path\mf_enforcementresult", replace
export delimited "$processed_data_path\mf_enforcementresult.csv", replace

gen against_person = (Regactiontype =="Cease and Desist Against a Person"| Regactiontype=="Fine Levied Against a Person"|Regactiontype=="Other Actions Against a Person"|Regactiontype=="Restitution by a Person" |Regactiontype=="Sanctions Against Personnel") 
bys REG_SNL_INSTN_KEY year: egen management_quality = sum(against_person)
bys year: egen any_action = total(!missing(REG_SNL_INSTN_KEY))
sort REG_SNL_INSTN_KEY IssueDate

gen severe = (Regactiontype == "Cease and Desist"| Regactiontype == "Deposit Insurance Threat"|Regactiontype == "Formal Agreement/Consent Order" |  Regactiontype == "Prompt Corrective Action")
replace severe = 0 if mi(severe)
drop _merge
gen Formal_Agreement = (Regactiontype == "Formal Agreement/Consent Order"&status==1)
gen Cease_and_Desist = (Regactiontype == "Cease and Desist"&status==1)
gen Prompt_Corrective_Action = (Regactiontype == "Prompt Corrective Action"&status==1)
gen Deposit_Insurance_Threat = (Regactiontype == "Deposit Insurance Threat"&status==1)
save "$processed_data_path\mfnd_enforcement_year",replace
export delimited "$processed_data_path\mfnd_enforcement_year.csv", replace

local action "severe Cease_and_Desist Formal_Agreement Deposit_Insurance_Threat Prompt_Corrective_Action"
foreach i of local action{
	capture drop t_`i'
	bys year : egen t_`i' = total(`i')
}
duplicates drop REG_SNL_INSTN_KEY  year ,force
keep REG_SNL_INSTN_KEY year any_action t_*  enforcement_status against_person

rename t_* * 
order year any_action severe Formal_Agreement Cease_and_Desist Prompt_Corrective_Action Deposit_Insurance_Threat
save "$processed_data_path\mfnd_enforcement",replace

use "$processed_data_path\mfnd_enforcement",clear 
drop if year<2008

cd $base_path
graph twoway connect any_action year || connect severe year,xlabel(2008(1)2019,angle(45)) 
graph export action_trend_nd.png, as(png) replace

preserve
drop if year<2008 
drop if year>2012
keep year any_action severe Formal_Agreement Cease_and_Desist Prompt_Corrective_Action Deposit_Insurance_Threat
export excel using "$processed_data_path\mfnd_enforcement_table08-12.xlsx", firstrow(variables) replace
restore

preserve
drop if year<2013
keep year any_action severe Formal_Agreement Cease_and_Desist Prompt_Corrective_Action Deposit_Insurance_Threat
export excel using "$processed_data_path\mfnd_enforcement_table13-19.xlsx", firstrow(variables) replace
restore

************************* for matching case：this is for checking why do we use 2010 YEAR DATA *********************************
clear
use "$processed_data_path\mf_enforcementresult",clear
gen against_person = (Regactiontype =="Cease and Desist Against a Person"| Regactiontype=="Fine Levied Against a Person"|Regactiontype=="Other Actions Against a Person"|Regactiontype=="Restitution by a Person" |Regactiontype=="Sanctions Against Personnel") 
bys REG_SNL_INSTN_KEY year: egen management_quality = sum(against_person)
drop if TotalAssets000 == . 
drop if Initial_Scale == .
drop if ZipCode == ""
bys year: egen any_action = total(!missing(REG_SNL_INSTN_KEY))
sort REG_SNL_INSTN_KEY IssueDate
gen severe = (Regactiontype == "Cease and Desist"| Regactiontype == "Deposit Insurance Threat"|Regactiontype == "Formal Agreement/Consent Order" |  Regactiontype == "Prompt Corrective Action")
replace severe = 0 if mi(severe)
drop _merge
gen Formal_Agreement = (Regactiontype == "Formal Agreement/Consent Order"&status==1)
gen Cease_and_Desist = (Regactiontype == "Cease and Desist"&status==1)
gen Prompt_Corrective_Action = (Regactiontype == "Prompt Corrective Action"&status==1)
gen Deposit_Insurance_Threat = (Regactiontype == "Deposit Insurance Threat"&status==1)

save "$processed_data_path\mfd_enforcement_year",replace
export delimited "$processed_data_path\mfd_enforcement_year.csv", replace

local action "severe Cease_and_Desist Formal_Agreement Deposit_Insurance_Threat Prompt_Corrective_Action"
foreach i of local action{
	capture drop t_`i'
	bys year : egen t_`i' = total(`i')
}
duplicates drop REG_SNL_INSTN_KEY  year ,force
keep REG_SNL_INSTN_KEY year any_action t_*  enforcement_status against_person

rename t_* * 
order year any_action severe Formal_Agreement Cease_and_Desist Prompt_Corrective_Action Deposit_Insurance_Threat
save "$processed_data_path\mfd_enforcement",replace

use "$processed_data_path\mfd_enforcement",clear 
drop if year<2008

cd $base_path
graph twoway connect any_action year || connect severe year,xlabel(2008(1)2019,angle(45)) 
graph export action_trend_nd.png, as(png) replace

preserve
drop if year<2008 
drop if year>2012
keep year any_action severe Formal_Agreement Cease_and_Desist Prompt_Corrective_Action Deposit_Insurance_Threat
export excel using "$processed_data_path\mfd_enforcement_table08-12.xlsx", firstrow(variables) replace
restore

preserve
drop if year<2013
keep year any_action severe Formal_Agreement Cease_and_Desist Prompt_Corrective_Action Deposit_Insurance_Threat
export excel using "$processed_data_path\mfd_enforcement_table13-19.xlsx", firstrow(variables) replace
restore

************** Part 3： Lobbying sample data  ***************************
**** import the mannual matching parent financial institution list *****

import delimited "$raw_data_path\lobby_id0819.csv",clear
tostring reg_snl_instn_key,replace
rename reg_snl_instn_key REG_SNL_INSTN_KEY
drop if REG_SNL_INSTN_KEY=="."
save "$processed_data_path/lobby_id0819",replace

import delimited "$raw_data_path\lobby0819.csv",clear
merge m:1 client using "$processed_data_path\lobby_id0819", nogen keep(matched)
****revolving_door_lobbyists/targeted_lobbying_on_regulators to check "active" lobbying *****
bys REG_SNL_INSTN_KEY year: egen revolving_door_lobbyists = sum(revolvingdoorprofiles)
replace revolving_door_lobbyists = 1 if revolving_door_lobbyists>0
bys REG_SNL_INSTN_KEY year: egen targeted_lobbying_on_regulators = sum(target)
replace targeted_lobbying_on_regulators = 1 if targeted_lobbying_on_regulators>0

duplicates drop REG_SNL_INSTN_KEY year,force
gen ParentInstitutionKey = REG_SNL_INSTN_KEY 
rename experience lobby_experience
rename total total_expenditure
keep ParentInstitutionKey REG_SNL_INSTN_KEY year lobby_status total_expenditure revolving_door_lobbyists targeted_lobbying_on_regulators lobby_experience
save "$processed_data_path\lobby",replace
export delimited "$processed_data_path\lobby.csv", replace


***************** Part 4: Regional Economic Variables ***********************
clear all
import excel "$raw_data_path\pcpi_growth0812.xlsx", sheet("Sheet1") firstrow clear
rename C g2008
rename D g2009
rename E g2010
rename F g2011
rename G g2012

reshape long g, i(GeoFips GeoName) j(year)
rename g growth
save "$processed_data_path/personal_income_growth0812",replace

rename GeoName CountyandState
replace CountyandState = subinstr(CountyandState ,"*","",.)
replace CountyandState = subinstr(CountyandState ,"Independent ","",.)
replace CountyandState = subinstr(CountyandState ,"Saint","St.",.)
save "$processed_data_path/growth_county0812", replace


******************* get non independent city area **********************************
gen independent = strpos( CountyandState , "(City)") > 0

preserve
keep if independent==0
drop independent
save "$processed_data_path/noinde_growthcounty",replace
restore

******************** get independent city area **************************************
keep if independent==1

gen city_part = substr( CountyandState , 1, strpos(CountyandState, "(") - 2)
gen state_part = substr( CountyandState , strpos(CountyandState, ", ") + 2, .)
gen city_type = substr( CountyandState , strpos(CountyandState, "("), strpos(CountyandState, ")") - strpos(CountyandState, "(") + 1)
gen reordered_city_name = city_part + "," +" "+ state_part + " " + city_type
replace CountyandState = reordered_city_name
drop independent city_part state_part city_type reordered_city_name
save "$processed_data_path/inde_growthcounty",replace

clear
use "$processed_data_path/noinde_growthcounty"
append using "$processed_data_path/inde_growthcounty"
replace CountyandState = "Anchorage, AK" if CountyandState == "Anchorage Municipality, AK"
replace CountyandState = "Fairbanks North Star, AK" if CountyandState == "Fairbanks North Star Borough, AK"
replace CountyandState = "Ketchikan Gateway, AK" if CountyandState == "Ketchikan Gateway Borough, AK"
replace CountyandState = "LaGrange, IN" if CountyandState == "Lagrange, IN"
replace CountyandState = "St.e Genevieve, MO" if CountyandState == "Ste. Genevieve, MO"
save "$processed_data_path/final_growth_county0812", replace

***************** Part 5: Instrumental Variables ***********************
import excel "$raw_data_path/distance.xlsx", sheet("Sheet1") firstrow clear
tostring REG_SNL_INSTN_KEY , replace
keep REG_SNL_INSTN_KEY distance
save "$processed_data_path/distance",replace

use "$processed_data_path\mf_bankvar_clean", clear
bys State: egen market_size = sum(Initial_Scale)
gen initial_market_size = Initial_Scale/market_size
duplicates drop REG_SNL_INSTN_KEY,force
keep REG_SNL_INSTN_KEY initial_market_size
save "$processed_data_path/initial_market_size",replace


****************** Part 6: merge the full dataset ***************************
****************** merge bank and lobby data ********************************
clear all
use "$processed_data_path\mf_bankvar_clean",clear
keep REG_SNL_INSTN_KEY year CompanyName CompanyType PrimaryRegulator YearEstablished ParentName ParentInstitutionKey ZipCode CountyandState State Initial_Scale status capital_adequacy asset_quality earning liquidity sensitivity_to_market_risk deposit_to_asset_ratio leverage total_core_deposit size age TotalAssets000
drop if TotalAssets000 == . 
drop if Initial_Scale == .
drop if ZipCode == ""
merge m:1 ParentInstitutionKey year using "$processed_data_path/lobby" ,nogen keep(1 3)
merge m:1 REG_SNL_INSTN_KEY year using "$processed_data_path/lobby", nogen update keep(1 5)
replace lobby_status = 0 if lobby_status==.
gen total_expenditure_missing = total_expenditure
local lobby "lobby_experience total_expenditure revolving_door_lobbyists targeted_lobbying_on_regulators"
foreach i of local lobby {
	replace `i' = 0 if `i' == .
}
save "$processed_data_path/bank_lobby",replace
****************** merge banklobby with enforcement **************************
use "$processed_data_path/bank_lobby",clear
expand 2 if year==2007,gen(a)
replace year=2006 if a==1
expand 2 if year==2007,gen(b)
replace year=2005 if b==1 
***************** notice that if bank_lobby does not drop empty total assets and empty initial scale, then we should use mfnd_enforcement **************
*merge 1:1 REG_SNL_INSTN_KEY year using "$processed_data_path\mfd_enforcement"
merge 1:1 REG_SNL_INSTN_KEY year using "$processed_data_path\mfnd_enforcement"
drop if _merge==2
drop _merge any_action
replace enforcement_status = 0 if enforcement_status == .
local action  "severe Cease_and_Desist Formal_Agreement Deposit_Insurance_Threat Prompt_Corrective_Action"
foreach i of local action {
	replace `i' = 0 if `i' == .
}
**** Management_quality: three year lags and moving average across four years.
replace against_person=0 if against_person==.
bys REG_SNL_INSTN_KEY : gen p1 = against_person[_n-1]
bys REG_SNL_INSTN_KEY : gen p2 = against_person[_n-2]
bys REG_SNL_INSTN_KEY : gen p3 = against_person[_n-3]
local lags "p1 p2 p3"
foreach i of local lags {
	replace `i' = 0 if `i' == .
}
egen management_quality = rowmean(against_person p1 p2 p3)
replace management_quality = -management_quality*4
drop p1 p2 p3 a b 
keep if year>=2007&year<=2020
replace severe=1 if severe !=0
save "$processed_data_path\bank_lobby_enforcement",replace

**** merge with personal_income_growth/distance/U.S Treasury
use  "$processed_data_path\bank_lobby_enforcement",clear
replace CountyandState = subinstr(CountyandState ,"*","",.)
replace CountyandState = subinstr(CountyandState ,"Independent ","",.)
replace CountyandState = subinstr(CountyandState ,"Saint","St.",.)
*********************** selected year based on what we are going to do *************************** 
drop if year<2008
drop if year>2012
merge m:1 year CountyandState using "$processed_data_path/final_growth_county0812"
drop if _merge==1|_merge==2
drop _merge
merge m:1 REG_SNL_INSTN_KEY using  "$processed_data_path\distance",nogen keep(1 3)
merge m:1 REG_SNL_INSTN_KEY using "$processed_data_path\initial_market_size",nogen keep(1 3)

global lobby_enforcement "lobby_status lobby_experience total_expenditure_missing total_expenditure revolving_door_lobbyists targeted_lobbying_on_regulators severe"

**** if we consider change model of covariates we use the following output variable lists
*global bank_num "z_score ROAA_volatility unused_commitment_growth loan_growth nonperforming_loans nonaccural_loans capital_adequacy asset_quality management_quality earning liquidity sensitivity_to_market_risk deposit_to_asset_ratio leverage  total_core_deposit size age ROAA"

global bank_num "capital_adequacy asset_quality management_quality earning liquidity sensitivity_to_market_risk deposit_to_asset_ratio leverage  total_core_deposit size age"
*global bank_char "REG_SNL_INSTN_KEY year CompanyName CompanyType PrimaryRegulator YearEstablished ParentName ParentInstitutionKey Location ZipCode State Initial_Scale status"
global bank_char "REG_SNL_INSTN_KEY year CompanyName CompanyType PrimaryRegulator YearEstablished ParentName ParentInstitutionKey CountyandState ZipCode State Initial_Scale status"

global others "growth distance initial_market_size"
winsor2  $bank_num , cuts(1 99) replace
order $bank_char $lobby_enforcement $bank_num
keep $bank_char $lobby_enforcement $bank_num $others
drop lobby_experience total_expenditure_missing total_expenditure revolving_door_lobbyists targeted_lobbying_on_regulators
save "$processed_data_path\rawdatafinal_all",replace
export delimited "$processed_data_path\rawdatafinal_all.csv",replace

keep if year == 2010
***** not sure if we should drop missing value directly or report them in the table,but make sure we drop what we need here.
*drop total_expenditure_missing total_expenditure revolving_door_lobbyists targeted_lobbying_on_regulators
*drop if asset_quality ==.   
save "$processed_data_path\rawdatafinal_2010",replace
export delimited "$processed_data_path\rawdatafinal_2010.csv",replace
















