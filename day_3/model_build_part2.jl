using Pumas
using PumasUtilities
using CSV
using Random
using DataFrames 
using DataFramesMeta
using CairoMakie
using Chain
using StatsBase
using CategoricalArrays
using Dates
using Serialization 
using Bioequivalence.GLM: lm, @formula
using Pumas.Latexify




## Read Data
pkdata = CSV.read("data/phmx_demo.csv", DataFrame, missingstring=[""])

# DV = concentration (mcg/L)
# AMT = amount (mg)
pkdata[!,:amt_mcg].=pkdata[!,:amt].*1000

#
# change column type to categorical for popPK modeling purposes 
@chain pkdata begin
     transform!(_, [:doselevel, :isPM, :isfed, :sex] .=> categorical, renamecols = false)
end



## prep data for poppk analysis
pop = read_pumas(pkdata,
                id = :id,
                time = :time, 
                amt = :amt_mcg, 
                evid = :evid, 
                cmt = :cmt,
                route = :route, 
                observations = [:dv], 
                covariates = [:age, :wt, :doselevel, :isPM, :isfed, :sex])
#


# final structural model from yesterday: 
pkfit_2cmp_prop = deserialize("pkfit_2cmp_prop.jls")
infer_2cmp_prop = infer(pkfit_2cmp_prop)
insp_2cmp_prop = inspect(pkfit_2cmp_prop)


metrics_2cmp_prop = metrics_table(pkfit_2cmp_prop)
    rename!(metrics_2cmp_prop,:Value => :TWO_CMP_PROP)
#





################# ASSESSMENT OF BSV ##################
# 7. Empirical Bayes Distribution 
empirical_bayes_dist(insp_2cmp_prop)


# 8. Shrinkage 
#ηshrinkage(pkfit_2cmp_prop)
#ϵshrinkage(pkfit_2cmp_prop)


# 9. Correlation of Random Effects 
df_insp = DataFrame(insp_2cmp_prop)
ebes = @chain df_insp begin
  select(r"^η")
  dropmissing
end

pp_2cmp_corr = PlottingUtilities.pair_plot(ebes)
#save("pp_2cmp_corr.png", pp_2cmp_corr)



# 10. Empirical Bayes vs Covariates 
eb_cov = empirical_bayes_vs_covariates(insp_2cmp_prop,
                              categorical = [:doselevel, :isPM, 
                              :isfed, :sex],
                              paginate = true,
                              limit = 3)
#

eb_cov[1]
eb_cov[2]
eb_cov[3]
eb_cov[4]

# pre block: 
#CL = tvcl * exp(η[1])
#Vc = tvvc * exp(η[2])
#Ka = tvka * exp(η[3])
#Vp = tvvp * exp(η[5])
#Q  = tvq  * exp(η[6])
#F --> eta4

# Apparent trends: 
# 1. poor metabolizer on CL 
# 2. fasting on Ka 
# 3. BW and disposition parameters (CLs and Vs)

# Suspected collinearity: 
# 1. isPM and dose group (apparent during data qual)
# 2. isfed and wt 



# 11. Collinearity between impactful covariates 
pkdata_cov = @rsubset pkdata :time==0.0
pkdata_cov = @rtransform pkdata_cov :isfed_NUM = :isfed == "no" ? 0 : 1
pkdata_cov = @rtransform pkdata_cov :isPM_NUM = :isPM == "no" ? 0 : 1
pkdata_cov = @rtransform pkdata_cov :sex_NUM = :sex == "male" ? 0 : 1


boxplot(pkdata_cov[!,:isfed_NUM], pkdata_cov[!,:wt], 
                show_outliers=true,
                axis = (xlabel = "Fed Status", 
                        ylabel = "Weight (kg)",
                        xticks = ([0,1],["Not Fed", "Yes Fed"])))
#
boxplot(pkdata_cov[!,:isPM_NUM], pkdata_cov[!,:wt], 
                show_outliers=true,
                axis = (xlabel = "Poor Metabolizer Status", 
                        ylabel = "Weight (kg)",
                        xticks = ([0,1],["Normal Metab", "Poor Met"])))
#
boxplot(pkdata_cov[!,:sex_NUM], pkdata_cov[!,:wt], 
                show_outliers=true,
                axis = (xlabel = "Sex", 
                        ylabel = "Weight (kg)",
                        xticks = ([0,1],["Male", "Female"])))
#







# going to start by assessing poor metabolizer effect
# to assess magnitude: 
df_insp = DataFrame(insp_2cmp_prop)
cov_check_cl = dropmissing(select(df_insp, [:id, :isPM, :CL]))

avg_cov_check_cl = @chain cov_check_cl begin
    groupby(_, [:isPM])
    combine(_, vec([:CL] .=> [mean median minimum maximum std]))
end
# poor appear metabolizers on average have about a 50% reduction in CL 
















#################################################
#                                               #
#           Covariate Model Building            #
#                 ISPM on CL                    #
#                                               #
#################################################

mdl_base_ispm = @model begin

  @metadata begin
    desc = "BASE+isPM"
    timeu = u"hr"
  end

  @param begin
    "Clearance (L/hr)"
    tvcl ∈ RealDomain(lower = 0.0001)
    "Volume (L)"
    tvvc  ∈ RealDomain(lower = 0.0001)
    "Peripheral Volume (L)"
    tvvp ∈ RealDomain(lower = 0.0001)
    "Distributional Clearance (L/hr)"
    tvq  ∈ RealDomain(lower = 0.0001)
    "Absorption rate constant (h-1)"
    tvka ∈ RealDomain(lower = 0.0001)
    "Bioavailability"
    tvbio ∈ RealDomain(lower = 0.001, upper=1.0)
    "Proportional change in CL due to PM"
    ispmoncl ∈ RealDomain(lower=-1.00, upper=1.00)
    """
    - ΩCL
    - ΩVc
    - ΩKa
    - ΩF
    - ΩVp
    - ΩQ
    """
    Ω    ∈ PDiagDomain(6)
    "Proportional RUV"
    σ²_prop  ∈ RealDomain(lower = 0.0001)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @covariates begin
    "Dose (mg)" 
    doselevel
    "Poor Metabolizer"
    isPM
    "Fed status"
    isfed
    "Sex"
    sex
    "Age (years)"
    age
    "Weight (kg)"
    wt
  end

  @pre begin
    CL = tvcl * (1 + ispmoncl * (isPM == "yes")) * exp(η[1])
    Vc = tvvc * exp(η[2])
    Ka = tvka * exp(η[3])
    Vp = tvvp * exp(η[5])
    Q  = tvq  * exp(η[6])
  end

  @dosecontrol begin
    bioav = (Depot = tvbio* exp(η[4]), Central=1.0)
  end

  @dynamics Depots1Central1Periph1

  @derived begin
    cp := @. (Central/Vc)
    """
    DrugY Concentration (ng/mL)
    """
    dv ~ @. Normal(cp, sqrt((abs(cp)^2)*σ²_prop))
  end

end


#params_base_ispm = (coef(pkfit_2cmp_prop)..., ispmoncl = -0.5)
params_base_ispm = (ispmoncl = -0.5,
                    tvcl = 2.6,
                    tvvc = 69,
                    tvka = 1,
                    tvbio = 0.9,
                    tvq = 3.8,
                    tvvp = 44,
                    Ω = Diagonal([0.04,0.04,0.04,0.04,0.04,0.04]),
                    σ²_prop = 0.1)
#

## Maximum likelihood estimation
pkfit_base_ispm = fit(mdl_base_ispm,
                pop,
                params_base_ispm,
                Pumas.FOCE())
#
serialize("pkfit_base_ispm.jls", pkfit_base_ispm)
#pkfit_base_ispm = deserialize("pkfit_base_ispm.jls")

## perform a LRT
lr_ispm = lrtest(pkfit_2cmp_prop, pkfit_base_ispm)
pvalue(lr_ispm)






################# ASSESSMENT OF ISPM COVARIATE EFFECT ON CL ######################
insp_base_ispm = inspect(pkfit_base_ispm)
infer_base_ispm = infer(pkfit_base_ispm)


res_base_ispm = evaluate_diagnostics([
                                (pkfit_2cmp_prop, infer_2cmp_prop, 
                                insp_2cmp_prop),
                                (pkfit_base_ispm, infer_base_ispm, 
                                insp_base_ispm)],
                                categorical =[:doselevel, :isPM, 
                                              :isfed,     :sex])
#        



# 1. Parameter Plausability 
metrics_base_ispm = metrics_table(pkfit_base_ispm)
    rename!(metrics_base_ispm,:Value => :BASE_ISPM)
df_comp = innerjoin(metrics_2cmp_prop, metrics_base_ispm, on=:Metric, makeunique=true)

comp_est = compare_estimates(;pkfit_2cmp_prop, pkfit_base_ispm)

# how much of the BSV on CL was explained? 
# before: CL BSV = sqrt(0.148904)*100% = 38.6%
# after: CL BSV = sqrt(0.0021953)*100% = 4.69%
# ~34% of the variability was explained! 😺



# 2. CWRES vs TIME 
figure2 = Figure(resolution = (1500, 500); fontsize = 20)
wresiduals_vs_time(figure2[1,1], insp_2cmp_prop; observations = :dv)
wresiduals_vs_time(figure2[1,2], insp_base_ispm; observations = :dv)
figure2
#save("wres_time_ispm.png", figure2)



# 3. CWRES vs PRED 
figure1 = Figure(resolution = (1500, 500); fontsize = 20)
wresiduals_vs_predictions(figure1[1,1], insp_2cmp_prop; observations = :dv)
wresiduals_vs_predictions(figure1[1,2], insp_base_ispm; observations = :dv)
figure1
#save("wres_pred_ispm.png", figure1)




# 3. Individual Subject Fits 
pred_base_ispm = predict(pkfit_base_ispm, obstimes=0:0.1:72)

#
figures_base_ispm = 
    subject_fits(
        pred_base_ispm;
        observations=[:dv],
        axis = (xlabel = "Time (hour)", 
                ylabel = "Concentration (mcg/L)",),
        separate = true,
        paginate = true,
        limit=4,
        facet = (combinelabels = true,),)
#
figures_base_ispm[1]



# 5. GOF - Observed vs Population Predicted 
figure4 = Figure(resolution = (1500, 500); fontsize = 20)
observations_vs_predictions(figure4[1,1], insp_2cmp_prop; observations = :dv)
observations_vs_predictions(figure4[1,2], insp_base_ispm; observations = :dv)
figure4
#save("dv_pred_ispm.png", figure4)


# 6. Empirical Bayes Distribution 
empirical_bayes_dist(insp_2cmp_prop)
empirical_bayes_dist(insp_base_ispm)



# 7. Empirical Bayes vs Covariates 
eb_cov_ispm = empirical_bayes_vs_covariates(insp_base_ispm,
                              categorical = [:doselevel, :isPM, 
                              :isfed, :sex],
                              paginate = true,
                              limit = 3)
#

# comparing before/after addition of isPM on CL 
eb_cov[2]
eb_cov_ispm[2]

# assess if prior observed trends are still present 
eb_cov_ispm[1]
eb_cov_ispm[3]
eb_cov_ispm[4]



# next going to assess addition of fed state on Ka 
# to assess magnitude: 
df_insp_ispm = DataFrame(insp_base_ispm)
cov_check_ka = dropmissing(select(df_insp_ispm, [:id, :isfed, :Ka]))

avg_cov_check_ka = @chain cov_check_ka begin
    groupby(_, [:isfed])
    combine(_, vec([:Ka] .=> [mean median minimum maximum std]))
end
# tvka_fed = 0.4,
# tvka_fasted = 1.5

# can either estimate 2 separate params or an effect on Ka 














#################################################
#                                               #
#           Covariate Model Building            #
#           isPM on CL + isFED on Ka            #
#                                               #
#################################################


mdl_base_ispm_isfed = @model begin

  @metadata begin
    desc = "BASE+isPM+isFED"
    timeu = u"hr"
  end

  @param begin
    "Clearance (L/hr)"
    tvcl ∈ RealDomain(lower = 0.0001)
    "Volume (L)"
    tvvc  ∈ RealDomain(lower = 0.0001)
    "Peripheral Volume (L)"
    tvvp ∈ RealDomain(lower = 0.0001)
    "Distributional Clearance (L/hr)"
    tvq  ∈ RealDomain(lower = 0.0001)
    #"Absorption rate constant (h-1)"
    #tvka ∈ RealDomain(lower = 0.0001)
    "Bioavailability"
    tvbio ∈ RealDomain(lower = 0.001, upper=1.0)
    "Proportional change in CL due to PM"
    ispmoncl ∈ RealDomain(lower=-1.00, upper=1.00)
    "Fed - Absorption rate constant (h-1)"
    tvkafed      ∈ RealDomain(lower=0.0001)
    "Fasted - Absorption rate constant (h-1)"
    tvkafasted   ∈ RealDomain(lower=0.0001)
    #"Proportional change in Ka due to Fasting"
    #isfedonka ∈ RealDomain(lower=-1.00, upper=1.00)
    """
    - ΩCL
    - ΩVc
    - ΩKa
    - ΩF
    - ΩVp
    - ΩQ
    """
    Ω    ∈ PDiagDomain(6)
    "Proportional RUV"
    σ²_prop  ∈ RealDomain(lower = 0.0001)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @covariates begin
    "Dose (mg)" 
    doselevel
    "Poor Metabolizer"
    isPM
    "Fed status"
    isfed
    "Sex"
    sex
    "Age (years)"
    age
    "Weight (kg)"
    wt
  end

  @pre begin
    CL = tvcl * (1 + ispmoncl * (isPM == "yes")) * exp(η[1])
    Vc = tvvc * exp(η[2])
    Ka = isfed == "yes" ? tvkafed * exp(η[3]) : tvkafasted * exp(η[3])
    Vp = tvvp * exp(η[5])
    Q  = tvq  * exp(η[6])
  end

  @dosecontrol begin
    bioav = (Depot = tvbio* exp(η[4]), Central=1.0)
  end

  @dynamics Depots1Central1Periph1

  @derived begin
    cp := @. (Central/Vc)
    """
    DrugY Concentration (ng/mL)
    """
    dv ~ @. Normal(cp, sqrt((abs(cp)^2)*σ²_prop))
  end

end

params_base_ispm_isfed = (tvkafed = 0.5,
                          tvkafasted = 1.5,
                          ispmoncl = -0.55,
                          tvcl = 3.7,
                          tvvc = 78,
                          tvq = 3.9,
                          tvvp = 50,
                          tvbio = 0.9,
                          Ω = Diagonal([0.04,0.04,0.04,0.04,0.04,0.04]),
                          σ²_prop = 0.1)
#


## Maximum likelihood estimation
pkfit_base_ispm_isfed = fit(mdl_base_ispm_isfed,
                pop,
                params_base_ispm_isfed,
                Pumas.FOCE())
#


serialize("pkfit_base_ispm_isfed.jls", pkfit_base_ispm_isfed)
#pkfit_base_ispm_isfed = deserialize("pkfit_base_ispm_isfed.jls")




################# ASSESSMENT OF ISFED COVARIATE EFFECT ON KA ######################
insp_base_ispm_isfed = inspect(pkfit_base_ispm_isfed)
infer_base_ispm_isfed = infer(pkfit_base_ispm_isfed)


res_base_ispm_isfed = evaluate_diagnostics([
                                (pkfit_base_ispm, infer_base_ispm, 
                                insp_base_ispm),
                                (pkfit_base_ispm_isfed, infer_base_ispm_isfed, 
                                insp_base_ispm_isfed)],
                                categorical =[:doselevel, :isPM, 
                                              :isfed,     :sex])
#        


## left off here
# 1. Parameter Plausability 
metrics_base_ispm_isfed = metrics_table(pkfit_base_ispm_isfed)
    rename!(metrics_base_ispm_isfed,:Value => :BASE_ISPM_ISFED)
df_comp = innerjoin(metrics_base_ispm, metrics_base_ispm_isfed, on=:Metric, makeunique=true)

comp_est = compare_estimates(;pkfit_base_ispm, pkfit_base_ispm_isfed)



# 2. CWRES vs TIME 
figure2 = Figure(resolution = (1500, 500); fontsize = 20)
wresiduals_vs_time(figure2[1,1], insp_base_ispm; observations = :dv)
wresiduals_vs_time(figure2[1,2], insp_base_ispm_isfed; observations = :dv)
figure2
#save("wres_time_ispm_isfed.png", figure2)



# 3. CWRES vs PRED 
figure1 = Figure(resolution = (1500, 500); fontsize = 20)
wresiduals_vs_predictions(figure1[1,1], insp_base_ispm; observations = :dv)
wresiduals_vs_predictions(figure1[1,2], insp_base_ispm_isfed; observations = :dv)
figure1
#save("wres_pred_ispm_isfed.png", figure1)




# 3. Individual Subject Fits 
pred_base_ispm_isfed = predict(pkfit_base_ispm_isfed, obstimes=0:0.1:72)

#
figures_base_ispm_isfed = 
    subject_fits(
        pred_base_ispm_isfed;
        observations=[:dv],
        axis = (xlabel = "Time (hour)", 
                ylabel = "Concentration (mcg/L)",),
        separate = true,
        paginate = true,
        limit=4,
        facet = (combinelabels = true,),)
#
figures_base_ispm_isfed[1]



# 5. GOF - Observed vs Population Predicted 
figure4 = Figure(resolution = (1500, 500); fontsize = 20)
observations_vs_predictions(figure4[1,1], insp_base_ispm; observations = :dv)
observations_vs_predictions(figure4[1,2], insp_base_ispm_isfed; observations = :dv)
figure4
#save("dv_pred_ispm_isfed.png", figure4)


# 6. Empirical Bayes Distribution 
empirical_bayes_dist(insp_base_ispm_isfed)



# 7. Empirical Bayes vs Covariates 
eb_cov_ispm_isfed = empirical_bayes_vs_covariates(insp_base_ispm_isfed,
                              categorical = [:doselevel, :isPM, 
                              :isfed, :sex],
                              paginate = true,
                              limit = 3)
#

# comparing before/after addition of isfed on Ka
eb_cov_ispm[2]
eb_cov_ispm_isfed[2]

# assess if prior observed trends are still present 
eb_cov_ispm_isfed[1]
eb_cov_ispm_isfed[3]
eb_cov_ispm_isfed[4]


## next going to add wt to PK disposition params 

















#################################################
#                                               #
#           Covariate Model Building            #
#        isPM on CL + isFED on Ka + WT          #
#                                               #
#################################################

mdl_base_ispm_isfed_wt = @model begin

  @metadata begin
    desc = "BASE+isPM+isFED+WT"
    timeu = u"hr"
  end

  @param begin
    "Clearance (L/hr)"
    tvcl ∈ RealDomain(lower = 0.0001)
    "Volume (L)"
    tvvc  ∈ RealDomain(lower = 0.0001)
    "Peripheral Volume (L)"
    tvvp ∈ RealDomain(lower = 0.0001)
    "Distributional Clearance (L/hr)"
    tvq  ∈ RealDomain(lower = 0.0001)
    "Bioavailability"
    tvbio ∈ RealDomain(lower = 0.001, upper=1.0)
    "Proportional change in CL due to PM"
    ispmoncl ∈ RealDomain(lower=-1.00, upper=1.00)
    "Fed - Absorption rate constant (h-1)"
    tvkafed      ∈ RealDomain(lower=0.0001)
    "Fasted - Absorption rate constant (h-1)"
    tvkafasted   ∈ RealDomain(lower=0.0001)
    """
    - ΩCL
    - ΩVc
    - ΩKa
    - ΩF
    - ΩVp
    - ΩQ
    """
    Ω    ∈ PDiagDomain(6)
    "Proportional RUV"
    σ²_prop  ∈ RealDomain(lower = 0.0001)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @covariates begin
    "Dose (mg)" 
    doselevel
    "Poor Metabolizer"
    isPM
    "Fed status"
    isfed
    "Sex"
    sex
    "Age (years)"
    age
    "Weight (kg)"
    wt
  end

  @pre begin
    wtcl = (wt/70)^0.75
    wtv  = (wt/70)
    CL = tvcl * wtcl * (1 + ispmoncl * (isPM == "yes")) * exp(η[1])
    Vc = tvvc * wtv  * exp(η[2])
    Ka = isfed == "yes" ? tvkafed * exp(η[3]) : tvkafasted * exp(η[3])
    Vp = tvvp * wtv  *  exp(η[5])
    Q  = tvq  * wtcl * exp(η[6])
  end

  @dosecontrol begin
    bioav = (Depot = tvbio* exp(η[4]), Central=1.0)
  end

  @dynamics Depots1Central1Periph1

  @derived begin
    cp = @. (Central/Vc)
    """
    DrugY Concentration (ng/mL)
    """
    dv ~ @. Normal(cp, sqrt((abs(cp)^2)*σ²_prop))
  end

end

params_base_ispm_isfed_wt = (coef(pkfit_base_ispm_isfed))


## Maximum likelihood estimation
pkfit_base_ispm_isfed_wt = fit(mdl_base_ispm_isfed_wt,
                pop,
                params_base_ispm_isfed_wt,
                Pumas.FOCE())
#


serialize("pkfit_base_ispm_isfed_wt.jls", pkfit_base_ispm_isfed_wt)
#pkfit_base_ispm_isfed_wt = deserialize("pkfit_base_ispm_isfed_wt.jls")


## cannot perform a LRT - these are not nested models 



################# ASSESSMENT OF WT COVARIATE EFFECT ######################
insp_base_ispm_isfed_wt = inspect(pkfit_base_ispm_isfed_wt)
infer_base_ispm_isfed_wt = infer(pkfit_base_ispm_isfed_wt)


res_base_ispm_isfed_wt = evaluate_diagnostics([
                                (pkfit_base_ispm_isfed, infer_base_ispm_isfed, 
                                insp_base_ispm_isfed),
                                (pkfit_base_ispm_isfed_wt, infer_base_ispm_isfed_wt, 
                                insp_base_ispm_isfed_wt)],
                                categorical =[:doselevel, :isPM, 
                                              :isfed,     :sex])
#        



# 1. Parameter Plausability 
metrics_base_ispm_isfed_wt = metrics_table(pkfit_base_ispm_isfed_wt)
    rename!(metrics_base_ispm_isfed_wt,:Value => :BASE_ISPM_ISFED_WT)
df_comp = innerjoin(metrics_base_ispm_isfed, metrics_base_ispm_isfed_wt, on=:Metric, makeunique=true)

comp_est = compare_estimates(;pkfit_base_ispm_isfed, pkfit_base_ispm_isfed_wt)



# 2. CWRES vs TIME 
figure2 = Figure(resolution = (1500, 500); fontsize = 20)
wresiduals_vs_time(figure2[1,1], insp_base_ispm_isfed; observations = :dv)
wresiduals_vs_time(figure2[1,2], insp_base_ispm_isfed_wt; observations = :dv)
figure2
#save("wres_time_ispm_isfed_wt.png", figure2)



# 3. CWRES vs PRED 
figure1 = Figure(resolution = (1500, 500); fontsize = 20)
wresiduals_vs_predictions(figure1[1,1], insp_base_ispm_isfed; observations = :dv)
wresiduals_vs_predictions(figure1[1,2], insp_base_ispm_isfed_wt; observations = :dv)
figure1
#save("wres_pred_ispm_isfed_wt.png", figure1)




# 3. Individual Subject Fits 
pred_base_ispm_isfed_wt = predict(pkfit_base_ispm_isfed_wt, obstimes=0:0.1:72)

#
figures_base_ispm_isfed_wt = 
    subject_fits(
        pred_base_ispm_isfed_wt;
        observations=[:dv],
        axis = (xlabel = "Time (hour)", 
                ylabel = "Concentration (mcg/L)",),
        separate = true,
        paginate = true,
        limit=4,
        facet = (combinelabels = true,),)
#
figures_base_ispm_isfed_wt[1]



# 5. GOF - Observed vs Population Predicted 
figure4 = Figure(resolution = (1500, 500); fontsize = 20)
observations_vs_predictions(figure4[1,1], insp_base_ispm_isfed; observations = :dv)
observations_vs_predictions(figure4[1,2], insp_base_ispm_isfed_wt; observations = :dv)
figure4
#save("dv_pred_ispm_isfed_wt.png", figure4)


# 6. Empirical Bayes Distribution 
empirical_bayes_dist(insp_base_ispm_isfed_wt)



# 7. Empirical Bayes vs Covariates 
eb_cov_ispm_isfed_wt = empirical_bayes_vs_covariates(insp_base_ispm_isfed_wt,
                              categorical = [:doselevel, :isPM, 
                              :isfed, :sex],
                              paginate = true,
                              limit = 3)
#
eb_cov_ispm_isfed_wt[1]
eb_cov_ispm_isfed_wt[2]
eb_cov_ispm_isfed_wt[3]
eb_cov_ispm_isfed_wt[4]






#################################################
#                                               #
#           Final Model Qualification           #
#                                               #
#################################################

## generate a visual predictive check
vpc_base_ispm_isfed_wt = vpc(pkfit_base_ispm_isfed_wt, stratify_by = [:doselevel], ensemblealg = EnsembleThreads())
vpc_plot(vpc_base_ispm_isfed_wt)

## View all the models that have been run
view_models = list_models()
selected_models(view_models)

## Generate an app to view your results interactively
compare_all_models = evaluate_diagnostics(selected_models(view_models),
                                categorical =[:doselevel, :isPM, 
                                              :isfed, :sex])
#
## REPL display of results
coefficients_table(pkfit_base_ispm_isfed_wt)
metrics_table(pkfit_base_ispm_isfed_wt)


## Bootstrap
bs_base_ispm_isfed_wt = infer(pkfit_base_ispm_isfed_wt, Pumas.Bootstrap(samples=200, stratify_by=:isPM))
DataFrame(bs_base_ispm_isfed_wt.vcov)


## SIR
sir_base_ispm_isfed_wt = infer(pkfit_base_ispm_isfed_wt, Pumas.SIR(samples=200, resamples=20))


## report
report((; base_ispm_isfed_wt = (pkfit_base_ispm_isfed_wt, 
                                  insp_base_ispm_isfed_wt, 
                                  infer_base_ispm_isfed_wt,
                                  vpc_base_ispm_isfed_wt)),
                                  categorical = [:doselevel, :isPM, 
                                  :isfed, :sex],
                                  date = Dates.now(),
                                  output = "drugy_report.pdf", 
                                  title = "DrugY Population Pharmacokinetic Analysis",
                                  author = "Allison",
                                  version = "v0.1",
                                  header = "Pumas Report",
                                  footer = "Confidential")
#














#################################################
#                                               #
#         Simulations with Final Model          #
#                                               #
#################################################

## comparisons 
# typical subject who is a poor metabolizer & fed vs 
# typical subject who is a normal metabolizer & fasted 
dose = DosageRegimen(60000, time = 0, route=NCA.EV)
choose_covariates1() = (wt = 70, doselevel=60, age = 40, sex="male", isfed="yes", isPM="yes")
subj_1 = Subject(id = 1,
            events = dose,
            covariates = choose_covariates1(),
            observations = (dv = nothing,
                            cp = nothing))
#
choose_covariates2() = (wt = 70, doselevel=60, age = 40, sex="male", isfed="no", isPM="no")
subj_2 = Subject(id = 2,
            events = dose,
            covariates = choose_covariates2(),
            observations = (dv = nothing,
                            cp = nothing))
#
# ignoring BSV & RUV 
param = (tvcl = 3.3282948681980944,
            tvvc = 63.33119898524297,
            tvvp = 43.99991526592511,
            tvq = 3.9550160877722695,
            tvbio = 0.9051695049102518,
            ispmoncl = -0.6124674772717063,
            tvkafed = 0.40949597534030757,
            tvkafasted = 1.500796375560725,
            Ω = Diagonal([0,0,0,0,0,0]),
            σ²_prop = 0)
#
pop_comp = [subj_1, subj_2]
obs = simobs(mdl_base_ispm_isfed_wt, pop_comp, param, obstimes = 0:0.1:120)

f,a = sim_plot(obs; observations=[:cp], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (hr)", 
               ylabel = "Predicted Concentration (mg/L)", 
               title = "Final Model"),
               color = :redsblues)
axislegend(a) 
f








# typical subject who is a normal metabolizer & fed vs 
# typical subject who is a normal metabolizer & fasted 
dose = DosageRegimen(60000, time = 0, route=NCA.EV)
choose_covariates1() = (wt = 70, doselevel=60, age = 40, sex="male", isfed="no", isPM="yes")
subj_1 = Subject(id = 1,
            events = dose,
            covariates = choose_covariates1(),
            observations = (dv = nothing,
                            cp = nothing))
#
choose_covariates2() = (wt = 70, doselevel=60, age = 40, sex="male", isfed="no", isPM="no")
subj_2 = Subject(id = 2,
            events = dose,
            covariates = choose_covariates2(),
            observations = (dv = nothing,
                            cp = nothing))
#
# ignoring BSV & RUV 
param = (tvcl = 3.3282948681980944,
            tvvc = 63.33119898524297,
            tvvp = 43.99991526592511,
            tvq = 3.9550160877722695,
            tvbio = 0.9051695049102518,
            ispmoncl = -0.6124674772717063,
            tvkafed = 0.40949597534030757,
            tvkafasted = 1.500796375560725,
            Ω = Diagonal([0,0,0,0,0,0]),
            σ²_prop = 0)
#
pop_comp = [subj_1, subj_2]
obs = simobs(mdl_base_ispm_isfed_wt, pop_comp, param, obstimes = 0:0.1:120)

g,a = sim_plot(obs; observations=[:cp], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (hr)", 
               ylabel = "Predicted Concentration (mg/L)", 
               title = "Final Model"),
               color = :redsblues)
axislegend(a) 
g












