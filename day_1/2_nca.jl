
# Call Necessary Packages 
using CSV
using Chain
using DataFrames
using Dates
using NCA
using NCAUtilities
using NCA.Unitful
using PumasUtilities
using CairoMakie





###################################################################
#                                                                 #
#                  IV BOLUS SINGLE DOSE EXAMPLE                   #
#                                                                 #
###################################################################



# Load Data
df_bolus_sd = CSV.read("data/iv_bolus_sd.csv", DataFrame, missingstring=["NA", ".", ""])


# Important to: 
# 1. Understand how the data must be formatted 
# 2. Understand the primary functions used to perform NCA 
#         - read_nca() 
#         - run_nca()



# Define Units 
timeu = u"hr"
concu = u"mg/L"
amtu  = u"mg"



# Map Dataframe to NCA Population
pop_bolus_sd = read_nca(df_bolus_sd,
                id            = :id,
                time          = :time,
                observations  = :conc,
                amt           = :amt,
                route         = :route,
                #timeu         = true, 
                #amtu          = true, 
                #concu         = true, 
                llq           = 0.001) # bioassay
#




# Preview Data 
## individual plot - linear scale 
obsvstimes = observations_vs_time(pop_bolus_sd[5])


## individual plot - semi-log scale
obsvstimes = observations_vs_time(pop_bolus_sd[1], axis = (yscale = log,))


## mean concentration-time curve of population 
summary_observations_vs_time(pop_bolus_sd,
                                axis = (xlabel = "Time (hour)", 
                                ylabel = "Drug Concentration (mg/L)"))
#


# plot means - semilog scale 
sp_log = summary_observations_vs_time(pop_bolus_sd, 
                                  axis = (xlabel = "Time (hour)", 
                                  ylabel = "Drug Concentration (mg/L)",
                                  yscale = Makie.pseudolog10))  
#




# Perform Simple NCA
nca_bolus_sd = run_nca(pop_bolus_sd, sigdigits=3)



# Run Annotated NCA for Final Report 
nca_bolus_sd_report = run_nca(pop_bolus_sd, sigdigits=3,
                        studyid="STUDY-001",
                        studytitle="Phase 1 Drug Trial: IV Bolus Single Dose", # required
                        author = [("Author 1", "Author 2")], # required
                        sponsor = "PumasAI",
                        date=Dates.now(),
                        conclabel="Drug Concentration (mg/L)",
                        timelabel="Time (hr)",
                        versionnumber=v"0.1",)
#



# Summarize Results of Interest for Final Report
param_summary_bolus_sd  = summarize(nca_bolus_sd_report.reportdf, 
                            parameters = [:half_life, 
                                          :tmax, 
                                          :cmax, 
                                          :auclast, 
                                          :vz_obs, 
                                          :cl_obs, 
                                          :aucinf_obs])
#


# Generate NCA Report 
report(nca_bolus_sd_report, param_summary_bolus_sd, clean=true) #clean=false for full output









###################################################################
#                                                                 #
#               IV INFUSION SINGLE DOSE EXAMPLE                   #
#                                                                 #
###################################################################



# Load Data
df_inf_sd = CSV.read("data/iv_infusion_sd.csv", DataFrame, missingstring=["NA", ".", ""])



# Map Dataframe to NCA Population
pop_inf_sd = read_nca(df_inf_sd,
                id            = :id,
                time          = :time,
                observations  = :conc,
                amt           = :amt,
                route         = :route,
                group         = [:group],
                llq           = 0.001)
#



# Preview Data 
## mean concentration-time curve of population 
summary_observations_vs_time(pop_inf_sd,
                                axis = (xlabel = "Time (hour)", 
                                ylabel = "Drug Concentration (mg/L)"))
#

## grid of individual plots for the first 9 subjects - linear scale 
observations_vs_time(pop_inf_sd[1:9], 
                        axis = (xlabel = "Time (hour)", 
                                ylabel = "Drug Concentration (mg/L)"))


ctplots = observations_vs_time(pop_inf_sd, 
                                axis = (xlabel = "Time (hour)", 
                                        ylabel = "Drug Concentration (mg/L)"),
                                paginate = true, #creates multiple pages  
                                columns = 3, rows = 3, #number of col/rows per page 
                                facet = (combinelabels = true,)) #creates 1 label for each page
#
ctplots[1]


# grid individual observation vs time plots - semilog scale 
ot_log = observations_vs_time(pop_inf_sd, 
                                axis = (xlabel = "Time (hour)", 
                                ylabel = "Drug Concentration (mg/L)",
                                yscale = log,),
                                #xticks = [0,12,24,36,48,60,72],),
                                separate = true,
                                paginate = true, 
                                limit = 9,
                                facet = (combinelabels=true,),)
#
ot_log[1]






# Perform Simple NCA
nca_inf_sd = run_nca(pop_inf_sd, sigdigits=3)






###### Generate Individual NCA Parameters #######
# https://docs.pumas.ai/stable/nca/ncafunctions/

# Clearance (CL=dose/AUC)
## L/hr 
cl        = NCA.cl(pop_inf_sd, sigdigits=3)  


# Volume of distribution during the elimination phase (Vz=Dose/(λz*AUC))
## L 
vz        = NCA.vz(pop_inf_sd, sigdigits=3)  


# Terminal elimination rate constant (λz or kel --> estimated using linear regression of conc vs time on log scale)
## 1/hr or hr-1
## kel = CL/V 
lambdaz   = NCA.lambdaz(pop_inf_sd, threshold=3, sigdigits=3)  #threshold=3 specifies the max no. of time point used for calculation


# Terminal half life (t1/2 = ln(2)/λz) --> terminal slope of the natural log of concentration vs time data 
## the time required for 50% of the drug to be eliminated 
## hr 
thalf     = NCA.thalf(pop_inf_sd, sigdigits=3) 


# AUC - area under the curve --> calculated using trapezoidal method 
## hr*mg/L (time x concentration)
auc_inf = NCA.auc(pop_inf_sd, auctype=:inf, method=:linuplogdown, sigdigits=3)
auc_last = NCA.auc(pop_inf_sd, auctype=:last, method=:linuplogdown, sigdigits=3)


# AUMC - area under the first momement of concentration (units=time^2 × concentration)
## The first moment is calculated as concentration x time (mg/L * hr)
## The AUMC is the area under the (concentration x time) versus time curve (mg/L * hr^2)
## hr^2*mg/L 
aumc_inf       = NCA.aumc(pop_inf_sd, auctype=:inf, sigdigits=3)
aumc_last      = NCA.aumc(pop_inf_sd, auctype=:last, sigdigits=3)


# Mean residence time (MRT = AUMC_inf/AUC_inf) (MRT = AUMC_last/AUC_last)
## average time the drug remains in the body 
mrt_inf       = NCA.mrt(pop_inf_sd, auctype=:inf, sigdigits=3) 
mrt_last      = NCA.mrt(pop_inf_sd, auctype=:last, sigdigits=3) 


# Dose normalized Cmax 
cmax_d    = NCA.cmax(pop_inf_sd, normalize=true, sigdigits=3) 
auc_d     = NCA.auc(pop_inf_sd, normalize=true, sigdigits=3) 



# create a dataframe from all individual parameters 
individual_params    = innerjoin(vz,cl,lambdaz,thalf,cmax_d, on=[:id,:group], makeunique=true) # include group to innerjoin**




# Other AUC calculation options 
auc0_12   = NCA.auc(pop_inf_sd, interval=(0,12), method=:linuplogdown, sigdigits=3) #various other methods are :linear, :linlog
auc12_24  = NCA.auc(pop_inf_sd, interval=(12,24), method=:linuplogdown, sigdigits=3) #looking at auc 12 to 24 hours (can make this interval anything!)
partial_aucs = NCA.auc(pop_inf_sd, interval = [(0,12), (12,24)], method=:linuplogdown, sigdigits=3)


auc_inf = NCA.auc(pop_inf_sd, auctype=:inf, method=:linuplogdown, sigdigits=3)
auc_last = NCA.auc(pop_inf_sd, auctype=:last, method=:linuplogdown, sigdigits=3)


# If we want to look at a parameter for 1 individual 
thalf_4     = NCA.thalf(pop_inf_sd[4], sigdigits=3) # Half-life calculation for 4th individual



# Run Annotated NCA for Final Report 
nca_inf_sd_report = run_nca(pop_inf_sd, sigdigits=3,
                        studyid="STUDY-002",
                        studytitle="Phase 1 Drug Trial: IV Infusion Single Dose", # required
                        author = [("Author 1", "Author 2")], # required
                        sponsor = "PumasAI",
                        date=Dates.now(),
                        conclabel="Drug Concentration (mg/L)",
                        timelabel="Time (hr)",
                        versionnumber=v"0.1",)
#


# Summarize Results of Interest for Final Report
param_summary_inf_sd  = summarize(nca_inf_sd_report.reportdf, 
                                stratify_by=[:group,], # stratifying by group so we can compare each dose 
                                parameters = [:half_life, 
                                          :tmax, 
                                          :cmax, 
                                          :auclast, 
                                          :vz_obs, 
                                          :cl_obs, 
                                          :aucinf_obs])


# Generate NCA Report 
report(nca_inf_sd_report, param_summary_inf_sd)





# Look at Individual Fits 
individual_fits = subject_fits(nca_inf_sd,
             axis = (xlabel = "Time (hr)", 
                     ylabel = "Drug Concentration (mg/L)",
                     yscale = log10),
             separate = true, paginate = true,
             limit = 16, columns = 4, rows = 4, 
             facet = (combinelabels = true,))
#
individual_fits[1]











###################################################################
#                                                                 #
#      ORAL MULTIPLE DOSE EXAMPLE: STEADY STATE DATA ONLY         #
#                                                                 #
###################################################################




# Load Data
df_oral_ss = CSV.read("data/oral_md_ss_only.csv", DataFrame, missingstring=["NA", ".", ""])



# Map Dataframe to NCA Population
pop_oral_ss = read_nca(df_oral_ss,
                id            = :id,
                time          = :tad,
                observations  = :conc,
                amt           = :amt,
                route         = :route,
                ii            = :ii,
                ss            = :ss, 
                llq           = 0.001)
#



# Preview Data 
## mean concentration-time curve of population 
summary_observations_vs_time(pop_oral_ss,
                                axis = (xlabel = "Time (hour)", 
                                ylabel = "Drug Concentration (mg/L)"))
#

## grid of individual plots for the first 9 subjects - linear scale 
ctplots = observations_vs_time(pop_oral_ss[1:9], 
                                axis = (xlabel = "Time (hour)", 
                                        ylabel = "Drug Concentration (mg/L)"),
                                paginate = true, #creates multiple pages  
                                columns = 3, rows = 3, #number of col/rows per page 
                                facet = (combinelabels = true,)) #creates 1 label for each page
ctplots[1]


# grid individual observation vs time plots - semilog scale 
ot_log = observations_vs_time(pop_oral_ss, 
                                axis = (xlabel = "Time (hour)", 
                                ylabel = "Drug Concentration (mg/L)",
                                yscale = log,),
                                #xticks = [0,12,24,36,48,60,72],),
                                separate = true,
                                paginate = true, 
                                limit = 9,
                                facet = (combinelabels=true,),)
#
ot_log[1]










# Perform Simple NCA
nca_oral_ss = run_nca(pop_oral_ss, sigdigits=3)



# Generate Individual NCA Parameters 
vz        = NCA.vz(pop_oral_ss, sigdigits=3)  # Volume of Distribution/F, in this case since the drug is given orally
cl        = NCA.cl(pop_oral_ss, sigdigits=3)  # Clearance/F, in this case since the drug is given orally
lambdaz   = NCA.lambdaz(pop_oral_ss, threshold=3, sigdigits=3)  # Terminal Elimination Rate Constant, threshold=3 specifies the max no. of time point used for calculation
thalf     = NCA.thalf(pop_oral_ss, sigdigits=3) # Half-life calculation 
cmax_d    = NCA.cmax(pop_oral_ss, normalize=true, sigdigits=3) # Dose Normalized Cmax
mrt       = NCA.mrt(pop_oral_ss, sigdigits=3) # Mean residence time
individual_params      = innerjoin(vz,cl,lambdaz,thalf,cmax_d,mrt, on=[:id], makeunique=true)


auc0_12   = NCA.auc(pop_oral_ss, interval=(0,12), method=:linuplogdown, sigdigits=3) #various other methods are :linear, :linlog
auc12_24  = NCA.auc(pop_oral_ss, interval=(12,24), method=:linuplogdown, sigdigits=3) #looking at auc 12 to 24 hours (can make this interval anything!)
partial_aucs = NCA.auc(pop_oral_ss, interval = [(0,12), (12,24)], method=:linuplogdown, sigdigits=3)





# Run Annotated NCA for Final Report 
nca_oral_ss_report = run_nca(pop_oral_ss, sigdigits=3,
                        studyid="STUDY-004",
                        studytitle="Phase 1 Drug Trial: Oral Multiple Dosing (SS Only)", # required
                        author = [("Author 1", "Author 2")], # required
                        sponsor = "PumasAI",
                        date=Dates.now(),
                        conclabel="Drug Concentration (mg/L)",
                        timelabel="Time (hr)",
                        versionnumber=v"0.1",)
#


# Summarize Results of Interest for Final Report
param_summary_oral_ss  = summarize(nca_oral_ss_report.reportdf, 
                                        parameters = [:half_life, 
                                                        :tmax, 
                                                        :cmax, 
                                                        :auclast, 
                                                        :vz_f_obs, 
                                                        :cl_f_obs])
#


# Generate NCA Report 
report(nca_oral_ss_report, param_summary_oral_ss)

















###################################################################
#                                                                 #
#   ORAL MULTIPLE DOSE EXAMPLE: FIRST DOSE & STEADY STATE DATA    #
#                                                                 #
###################################################################





#################### OPTION #1 ####################

# Load Data
df_oral_first_ss = CSV.read("data/oral_md_first_ss.csv", DataFrame, missingstring=["NA", ".", ""])



# Map Dataframe to NCA Population
pop_oral_first_ss = read_nca(df_oral_first_ss,
                id            = :id,
                time          = :time,
                observations  = :conc,
                amt           = :amt,
                route         = :route,
                ii            = :ii,
                ss            = :ss, 
                llq           = 0.001)
#


# Preview Data 
## individual plot - linear scale 
obsvstimes = observations_vs_time(pop_oral_first_ss[1])





# Perform Simple NCA
nca_oral_first_ss = run_nca(pop_oral_first_ss, sigdigits=3)


# Run Annotated NCA for Final Report 
nca_oral_first_ss_report = run_nca(pop_oral_first_ss, sigdigits=3,
                        studyid="STUDY-003",
                        studytitle="Phase 1 Drug Trial: Oral Multiple Dosing (First & SS)", # required
                        author = [("Author 1", "Author 2")], # required
                        sponsor = "PumasAI",
                        date=Dates.now(),
                        conclabel="Drug Concentration (mg/L)",
                        timelabel="Time (hr)",
                        versionnumber=v"0.1",)
#


# Summarize Results of Interest for Final Report
param_summary_oral_first_ss  = summarize(nca_oral_first_ss_report.reportdf, 
                                        parameters = [:half_life, 
                                                        :tmax, 
                                                        :cmax, 
                                                        :auclast, 
                                                        :vz_f_obs, 
                                                        :cl_f_obs])
#


# Generate NCA Report 
report(nca_oral_first_ss_report, param_summary_oral_first_ss)















######################## option #2 ##############################

# Load Data
df_oral_first_ss = CSV.read("data/oral_md_first_ss.csv", DataFrame, missingstring=["NA", ".", ""])


# Data wrangle 
df_oral_first_ss = @rtransform df_oral_first_ss :evid = ismissing(:amt) == false ? 1 : 0 
df_oral_first_ss = @chain df_oral_first_ss begin
    groupby(_, [:id]) 
    transform(_, :evid => (x -> cumsum(x)) => :occ)
end
  

# Map Dataframe to NCA Population
pop_oral_first_ss = read_nca(df_oral_first_ss,
                id            = :id,
                time          = :time,
                observations  = :conc,
                amt           = :amt,
                route         = :route,
                ii            = :ii,
                ss            = :ss, 
                group         = [:occ,],
                llq           = 0.001)
#


# Preview Data 
## individual plot - linear scale 
obsvstimes = observations_vs_time(pop_oral_first_ss[1])
obsvstimes = observations_vs_time(pop_oral_first_ss[2])
obsvstimes = observations_vs_time(pop_oral_first_ss[11])




# Perform Simple NCA
nca_oral_first_ss = run_nca(pop_oral_first_ss, sigdigits=3)

df_nca = DataFrame(nca_oral_first_ss.reportdf)
df_nca[!,:id] = parse.(Int,df_nca[!,:id])
df_nca[!,:occ] = parse.(Int,df_nca[!,:occ])
df_nca = @orderby df_nca :id :occ







# Run Annotated NCA for Final Report 
nca_oral_first_ss_report = run_nca(pop_oral_first_ss, sigdigits=3,
                        studyid="STUDY-003",
                        studytitle="Second - Phase 1 Drug Trial: Oral Multiple Dosing (First & SS)", # required
                        author = [("Author 1", "Author 2")], # required
                        sponsor = "PumasAI",
                        date=Dates.now(),
                        conclabel="Drug Concentration (mg/L)",
                        timelabel="Time (hr)",
                        versionnumber=v"0.1",)
#


# Summarize Results of Interest for Final Report
param_summary_oral_first_ss  = summarize(nca_oral_first_ss_report.reportdf, 
                                        stratify_by = [:occ,],
                                        parameters = [:half_life, 
                                                        :tmax, 
                                                        :cmax, 
                                                        :auclast, 
                                                        :vz_f_obs, 
                                                        :cl_f_obs])
#


# Generate NCA Report 
report(nca_oral_first_ss_report, param_summary_oral_first_ss)