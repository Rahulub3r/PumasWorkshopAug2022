

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
using AlgebraOfGraphics


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

## data summaries
## continuous covariate summary
cont_cov_summary = @chain pkdata begin
    filter(:evid => .==(1), _)
    groupby(_, [:doselevel])
    combine(_, vec([:age,:wt] .=> [mean std]))
end


## categorical covariate summary 
cat_cov_summary = @chain pkdata begin
  filter(:evid => .==(1), _)
  stack([:isPM,:isfed,:sex])
  groupby(_, [:doselevel, :variable, :value])
  combine(nrow => :count)
  groupby(_, [:doselevel, :variable])
  transform(_, :count => (x -> x / sum(x)) => :prop) 
  sort(_, [:doselevel])
end



##  NCA
ncapop = read_nca(pkdata,
                amt          = :amt_mcg,
                observations = :dv, 
                group = [:doselevel])
#

# plot means - linear scale 
sp = summary_observations_vs_time(ncapop, 
                                    xlabel = "Time (hours)", 
                                    ylabel = "CTMX Concentration (μg/L)",
                                    columns = 2, rows = 3,
                                    plot_resolution = (600, 1000))                                    
# plot means - semilog scale 
sp_log = summary_observations_vs_time(ncapop, 
                                  axis = (xlabel = "Time (hours)", 
                                  ylabel = "CTMX Concentration (μg/L)",
                                  yscale = Makie.pseudolog10))  
# 

# individual observation vs time plots - linear scale 
ot = observations_vs_time(ncapop, 
                                xlabel = "Time (hours)", 
                                ylabel = "CTMX Concentration (μg/L)",
                                separate = true,
                                paginate = true, 
                                limit = 9,
                                facet = (combinelabels=true,),)
#
ot[1]

# individual observation vs time plots - semilog scale 
ot_log = observations_vs_time(ncapop, 
                                axis = (xlabel = "Time (hours)", 
                                ylabel = "CTMX Concentration (μg/L)",
                                #yscale = log10, #base 10
                                yscale = log, #base e
                                xticks = [0,12,24,36,48,60,72],),
                                separate = true,
                                paginate = true, 
                                limit = 9,
                                facet = (combinelabels=true,),)
#
ot_log[1]


pk_nca = run_nca(ncapop, sigdig=3, 
                studyid="STUDY-001",
                studytitle="Phase 1 SAD of CTM Analgesic",
                author = [("Allison", "CTM"),("Rahul", "CTM") ],
                sponsor = "UMB",
                date=Dates.now(),
                conclabel="CTMX Concentration (μg/L)",
                grouplabel = "Dose (mg)",
                timelabel="Time (Hr)",
                versionnumber=v"0.1",)


## parameter distribution plots    
parameters_vs_group(pk_nca, parameter=[:aucinf_obs])
parameters_vs_group(pk_nca, parameter=[:cmax])



strata = [:doselevel]
parms = [:cmax, :aucinf_obs]
output = summarize(pk_nca.reportdf; stratify_by = strata, parameters = parms)


## check dose proportionality -- from bioequivalence package
# dp_cmax = lm(@formula(cmax~doselevel), pk_nca.reportdf)
# dp_auc = lm(@formula(aucinf_obs~doselevel), pk_nca.reportdf)


## generate a full report
#report(pk_nca, output, output="demo_report")

pk_init = summarize(pk_nca.reportdf;
                    parameters = [:cl_f_obs,:vz_f_obs,:kel,:tmax])
#
# cl = 2.88 L/hr 
# kel = 0.02 (1/hr) 
# V = CL/kel = 144 L 

# 3 - 5 half lifes to Tmax 
# Tmax = 4*t1/2_a --> t1/2_a = 4/3 = 1.33
# Ka = ln(2)/t1/2_a = 0.693/1.33 = 0.9

# resource for geomean: https://www.lexjansen.com/nesug/nesug11/ph/ph10.pdf 


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




## some basic plots prior to modeling 
covariates_check(pop)
covariates_check(pop, covariates=[:wt])
observations_vs_time(pop)



# Discuss
# 1. Why do we model (characterize the drugs PK at a population level, answer Qs)
# 2. Model fundamentals (compartmental models, variability)
# 3. Data setup requirements 


#################################################
#                                               #
#           NAIVE POOLED ANALYSIS               #
#                                               #
#################################################

# https://docs.pumas.ai/dev/basics/models/

mdl_1cmp_np = @model begin

  @param begin
    "Clearance (L/hr)"
    tvcl ∈ RealDomain(lower = 0.0001)
    "Volume (L)"
    tvvc  ∈ RealDomain(lower = 0.0001)
    "Absorption rate constant (h-1)"
    tvka ∈ RealDomain(lower = 0.0001)
    "Bioavailability"
    tvbio ∈ RealDomain(lower = 0.001, upper=1.0)
    "Additive RUV"
    σ²_add  ∈ RealDomain(lower = 0.0001)
    "Proportional RUV"
    σ²_prop  ∈ RealDomain(lower = 0.0001)
  end

  @pre begin
    CL = tvcl 
    Vc = tvvc 
    Ka = tvka 
  end

  @dosecontrol begin
    bioav = (Depot = tvbio, Central=1.0)
  end

  #@dynamics Depots1Central1
  @dynamics begin 
    Depot'   = -Ka*Depot
    Central' =  Ka*Depot - CL/Vc*Central
  end 

  @derived begin
    cp := @. (Central/Vc)
    """
    DrugY Concentration (mcg/L)
    """
    dv ~ @. Normal(cp, sqrt(((abs(cp)^2)*σ²_prop) + σ²_add))
  end
end



params_1cmp_np = (
              tvcl = 2.88,
              tvvc = 144,
              tvka = 1,
              tvbio = 0.9,
              σ²_add = 0.1, 
              σ²_prop = 0.1)
#

## Naive Pooled analysis
pkfit_1cmp_np = fit(mdl_1cmp_np,
               pop,
               params_1cmp_np,
               Pumas.NaivePooled())
#
df_insp_np = DataFrame(inspect(pkfit_1cmp_np))
conc = @select(df_insp_np,:tad, :doselevel, :dv_pred)
conc = unique(dropmissing(conc))

plt_conc = data(conc) * (visual(Scatter) + visual(Lines)) * mapping(:tad, :dv_pred, color=:doselevel);
d_np = draw(plt_conc; axis=(; aspect=1, 
            title="Predicted Concentration: Naive Pooled",
            xlabel="Time (hour)",
            ylabel="Concentration (mcg/L)",
            yticks = [0,200,400,600,800]))
#











#################################################
#                                               #
#             TWO STAGE ANALYSIS                #
#                                               #
#################################################

fit_2stg = [fit(mdl_1cmp_np,
                pop[i],
                params_1cmp_np) for i in 1:length(pop)]
fit_2stg[1]
output_vec = reduce(vcat, DataFrame.(icoef.(fit_2stg)))
indpars_vec = @select output_vec :id :CL :Vc :Ka


vcat_out = vcat(DataFrame.(inspect.(fit_2stg))...)
inspect_2stg = reduce(vcat, DataFrame(inspect(subj)) for subj in fit_2stg)

conc_ts = @select(inspect_2stg, :id, :tad, :doselevel, :dv_ipred)
conc_ts = unique(dropmissing(conc_ts))
plt_conc_ts = data(conc_ts) * (visual(Scatter) + visual(Lines)) * mapping(:tad, :dv_ipred, color=:id);
d_ts = draw(plt_conc_ts; axis=(; aspect=1, 
            title="Predicted Concentration: Naive Pooled",
            xlabel="Time (hour)",
            ylabel="Concentration (mcg/L)",
            yticks = [0,400,800,1200]))
#





# by dose group
conc_ts_30 = @rsubset conc_ts :doselevel == 30
conc_ts_60 = @rsubset conc_ts :doselevel == 60
conc_ts_90 = @rsubset conc_ts :doselevel == 90


plt_conc_ts_30 = data(conc_ts_30) * (visual(Scatter) + visual(Lines)) * mapping(:tad, :dv_ipred, color=:id);
d_ts_30 = draw(plt_conc_ts_30; axis=(; aspect=1, 
            title="Predicted Concentration: Naive Pooled",
            xlabel="Time (hour)",
            ylabel="Concentration (mcg/L)",
            yticks = [0,200,400,600]))
#


plt_conc_ts_60 = data(conc_ts_60) * (visual(Scatter) + visual(Lines)) * mapping(:tad, :dv_ipred, color=:id);
d_ts_30 = draw(plt_conc_ts_60; axis=(; aspect=1, 
            title="Predicted Concentration: Naive Pooled",
            xlabel="Time (hour)",
            ylabel="Concentration (mcg/L)",
            yticks = [0,200,400,600]))
#


plt_conc_ts_90 = data(conc_ts_90) * (visual(Scatter) + visual(Lines)) * mapping(:tad, :dv_ipred, color=:id);
d_ts_30 = draw(plt_conc_ts_90; axis=(; aspect=1, 
            title="Predicted Concentration: Naive Pooled",
            xlabel="Time (hour)",
            ylabel="Concentration (mcg/L)",
            yticks = [0,400,800,1200,1600]))
#






#################################################
#                                               #
#                 One Stage:                    #
#           Base Model Development              #
#                                               #
#################################################

# Goals during structural model build are to adequately capture...
# 1. ADME properties of the drug 
# 2. Between-subject variability 
# 3. Residual unexplained variability 


### ONE COMPARTMENT MODEL, 1ST ORDER ABSORPTION, 1ST ORDER ELIMINATION 
mdl_1cmp = @model begin

    @metadata begin
      desc = "base model: 1comp"
      timeu = u"hr"
    end

    @param begin
      "Clearance (L/hr)"
      tvcl ∈ RealDomain(lower = 0.0001)
      "Volume (L)"
      tvvc  ∈ RealDomain(lower = 0.0001)
      "Absorption rate constant (h-1)"
      tvka ∈ RealDomain(lower = 0.0001)
      "Bioavailability"
      tvbio ∈ RealDomain(lower = 0.001, upper=1.0)
      """
      - ΩCL
      - ΩVc
      - ΩKa
      - ΩF
      """
      Ω    ∈ PDiagDomain(4)
      "Additive RUV"
      σ²_add  ∈ RealDomain(lower = 0.0001)
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
      CL = tvcl * exp(η[1])
      Vc = tvvc * exp(η[2])
      Ka = tvka * exp(η[3])
    end

    @dosecontrol begin
      bioav = (Depot = tvbio* exp(η[4]), Central=1.0)
    end

    @dynamics Depots1Central1

    @derived begin
      cp := @. (Central/Vc)
      """
      DrugY Concentration (ng/mL)
      """
      dv ~ @. Normal(cp, sqrt(((abs(cp)^2)*σ²_prop) + σ²_add))
    end
end


params_1cmp = (
                tvcl = 2.6,
                tvvc = 97,
                tvka = 3,
                tvbio = 0.9,
                Ω = Diagonal([0.04,0.04,0.04,0.04]),
                σ²_add = 0.1, 
                σ²_prop = 0.1)
#


## Generate latex code in the REPL
latexify(mdl_1cmp, :dynamics)
# render the latex code in the plot pane
# you can right click on the rendered code and save the tex to use in Word e.g.
render(latexify(mdl_1cmp, :dynamics))


## Initial estimate explorer
ee_1cmp = explore_estimates(mdl_1cmp, 
                     pop, 
                     params_1cmp)
## update coefficients based on app
#pkparams_1cmp = coef(ee_1cmp)


## Evaluate initial loglikelihood 
pkfit_1cmp_ll = loglikelihood(mdl_1cmp,
                              pop,
                              params_1cmp,
                              Pumas.FOCE()) # FOCE = first-order conditional estimation

## explore influential individuals 
pkfit_1cmp_fi = findinfluential(mdl_1cmp,
                                  pop,
                                  params_1cmp,
                                  Pumas.FOCE())       

##



## Maximum likelihood estimation
# combination RUV 
pkfit_1cmp_comb = fit(mdl_1cmp,
                 pop,
                 params_1cmp,
                 Pumas.FOCE())
#
# additive RUV 
pkfit_1cmp_add = fit(mdl_1cmp,
                 pop,
                 params_1cmp,
                 Pumas.FOCE(),
                 constantcoef=(σ²_prop=0.0,))
#
# proportional RUV 
pkfit_1cmp_prop = fit(mdl_1cmp,
                 pop,
                 params_1cmp,
                 Pumas.FOCE(),
                 constantcoef=(σ²_add=0.0,))
#
serialize("pkfit_1cmp_add.jls", pkfit_1cmp_add)
serialize("pkfit_1cmp_prop.jls", pkfit_1cmp_prop)
serialize("pkfit_1cmp_comb.jls", pkfit_1cmp_comb)
#pkfit_1cmp_add = deserialize("pkfit_1cmp_add.jls")
#pkfit_1cmp_prop = deserialize("pkfit_1cmp_prop.jls")
#pkfit_1cmp_comb = deserialize("pkfit_1cmp_comb.jls")






#### Diagnostics - Assess RUV model

## Using interactive app 
infer_1cmp_add = infer(pkfit_1cmp_add)
infer_1cmp_prop = infer(pkfit_1cmp_prop)
infer_1cmp_comb = infer(pkfit_1cmp_comb)

insp_1cmp_add = inspect(pkfit_1cmp_add)
insp_1cmp_prop = inspect(pkfit_1cmp_prop)
insp_1cmp_comb = inspect(pkfit_1cmp_comb)


res_1cmp = evaluate_diagnostics([(pkfit_1cmp_add, infer_1cmp_add, 
                                insp_1cmp_add),
                                (pkfit_1cmp_prop, infer_1cmp_prop, 
                                insp_1cmp_prop),
                                (pkfit_1cmp_comb, infer_1cmp_comb, 
                                insp_1cmp_comb)],
                                categorical =[:doselevel, :isPM, 
                                              :isfed,     :sex])
#                 



## Manual assessment 
# 1. Parameter Plausability 
metrics_1cmp_add = metrics_table(pkfit_1cmp_add)
    rename!(metrics_1cmp_add,:Value => :ONE_CMP_ADD)
  metrics_1cmp_prop = metrics_table(pkfit_1cmp_prop)
    rename!(metrics_1cmp_prop,:Value => :ONE_CMP_PROP)
metrics_1cmp_comb = metrics_table(pkfit_1cmp_comb)
    rename!(metrics_1cmp_comb,:Value => :ONE_CMP_COMB)
df_comp = innerjoin(metrics_1cmp_add, metrics_1cmp_prop, metrics_1cmp_comb, on=:Metric, makeunique=true)


comp_est = compare_estimates(; pkfit_1cmp_add, pkfit_1cmp_prop, pkfit_1cmp_comb)



# 2. CWRES vs PRED 
# Residuals = DV-PRED/W where W is some weight that is normalizing DV-IPRED 
figure1 = Figure(resolution = (2000, 500); fontsize = 20)
wresiduals_vs_predictions(figure1[1,1], insp_1cmp_add; observations = :dv)
wresiduals_vs_predictions(figure1[1,2], insp_1cmp_prop; observations = :dv)
wresiduals_vs_predictions(figure1[1,3], insp_1cmp_comb; observations = :dv)
figure1
#save("wres_pred.png", figure1)




# 3. IWRES vs IPRED 
# IWRES = DV-IPRED/sigma 
figure2 = Figure(resolution = (2000, 500); fontsize = 20)
iwresiduals_vs_ipredictions(figure2[1,1], insp_1cmp_add; observations = :dv)
iwresiduals_vs_ipredictions(figure2[1,2], insp_1cmp_prop; observations = :dv)
iwresiduals_vs_ipredictions(figure2[1,3], insp_1cmp_comb; observations = :dv)
figure2
#save("iwres_ipred.png", figure2)



# 4. Individual Subject Fits 
pred_1cmp_add = predict(pkfit_1cmp_add, obstimes=0:0.1:72)
pred_1cmp_prop = predict(pkfit_1cmp_prop, obstimes=0:0.1:72)
pred_1cmp_comb = predict(pkfit_1cmp_comb, obstimes=0:0.1:72)


figures_1cmp_add = 
    subject_fits(
        pred_1cmp_add;
        observations=[:dv],
        axis = (xlabel = "Time (hour)", 
                ylabel = "Concentration (mcg/L)",),
                #xticks = [0,2,4,6,8],
                #limits = (0, 10, nothing, nothing),
                #yscale = Makie.pseudolog10,),
        separate = true,
        paginate = true,
        limit=4,
        facet = (combinelabels = true,),)
#
figures_1cmp_prop = 
    subject_fits(
        pred_1cmp_prop;
        observations=[:dv],
        axis = (xlabel = "Time (hour)", 
                ylabel = "Concentration (mcg/L)",),
                #xticks = [0,2,4,6,8],
                #limits = (0, 10, nothing, nothing),
                #yscale = Makie.pseudolog10,),
        separate = true,
        paginate = true,
        limit=4,
        facet = (combinelabels = true,),)
#
figures_1cmp_comb = 
    subject_fits(
        pred_1cmp_comb;
        observations=[:dv],
        axis = (xlabel = "Time (hour)", 
                ylabel = "Concentration (mcg/L)",),
                #xticks = [0,2,4,6,8],
                #limits = (0, 10, nothing, nothing),
                #yscale = Makie.pseudolog10,),
        separate = true,
        paginate = true,
        limit=4,
        facet = (combinelabels = true,),)
#

figures_1cmp_add[1]
figures_1cmp_prop[1]
figures_1cmp_comb[1]











### TWO COMPARTMENT MODEL, 1ST ORDER ABSORPTION, 1ST ORDER ELIMINATION, COMBINATION RUV 
mdl_2cmp = @model begin

  @metadata begin
    desc = "base model: 2comp"
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
    """
    - ΩCL
    - ΩVc
    - ΩKa
    - ΩF
    - ΩVp
    - ΩQ
    """
    Ω    ∈ PDiagDomain(6)
    "Additive RUV"
    σ²_add  ∈ RealDomain(lower = 0.0001)
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
    CL = tvcl * exp(η[1])
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
    cp = @. (Central/Vc)
    """
    DrugY Concentration (ng/mL)
    """
    dv ~ @. Normal(cp, sqrt(((abs(cp)^2)*σ²_prop) + σ²_add))
  end

end

params_2cmp = (
  tvcl = 3.0,
  tvvc = 110.0,
  tvq = 4.0,
  tvvp = 50.0,
  tvka = 1.6,
  tvbio = 0.9,
  Ω = Diagonal([0.04,0.04,0.04,0.04,0.04,0.04]),
  σ²_add = 0.1,
  σ²_prop = 0.1)
#



latexify(mdl_2cmp, :dynamics)
render(latexify(mdl_2cmp, :dynamics))


## Initial estimate explorer
ee_2cmp = explore_estimates(mdl_2cmp, 
                     pop, 
                     params_2cmp)
## update coefficients based on app
#params_2cmp = coef(ee_2cmp)


## Evaluate initial loglikelihood 
pkfit_2cmp_ll = loglikelihood(mdl_2cmp,
                              pop,
                              params_2cmp,
                              Pumas.FOCE())

## explore influential individuals 
pkfit_2cmp_fi = findinfluential(mdl_2cmp,
                                  pop,
                                  params_2cmp,
                                  Pumas.FOCE())       

##

## Naive Pooled analysis
pkfit_2cmp_np = fit(mdl_2cmp,
                 pop,
                 params_2cmp,
                 Pumas.NaivePooled(), 
                 omegas=(:Ω,))
#
## Maximum likelihood estimation


params_2cmp = (
  tvcl = 3.0,
  tvvc = 67,
  tvq = 4.0,
  tvvp = 66,
  tvka = 1.6,
  tvbio = 0.9,
  Ω = Diagonal([0.04,0.04,0.04,0.04,0.04,0.04]),
  σ²_add = 0.1,
  σ²_prop = 0.1)
#
#

# combination RUV 
pkfit_2cmp_comb = fit(mdl_2cmp,
                 pop,
                 params_2cmp,
                 Pumas.FOCE())
#
serialize("pkfit_2cmp_comb.jls", pkfit_2cmp_comb)
#pkfit_2cmp_comb = deserialize("pkfit_2cmp_comb.jls")
infer_2cmp_comb = infer(pkfit_2cmp_comb) # failure due to additive RUV approaching zero (should remove additive RUV for this reason)








# run with proportional error 
# combination RUV 
pkfit_2cmp_prop = fit(mdl_2cmp,
                 pop,
                 params_2cmp,
                 Pumas.FOCE(),
                 constantcoef=(σ²_add=0.0,))
#                 
serialize("pkfit_2cmp_prop.jls", pkfit_2cmp_prop)
#pkfit_2cmp_prop = deserialize("pkfit_2cmp_prop.jls")
infer_2cmp_prop = infer(pkfit_2cmp_prop) 
insp_2cmp_prop = inspect(pkfit_2cmp_prop)








################# ASSESSMENT OF STRUCTURAL MODEL ######################
res_base = evaluate_diagnostics([
                                (pkfit_1cmp_comb, infer_1cmp_comb, 
                                insp_1cmp_comb),
                                (pkfit_2cmp_prop, infer_2cmp_prop, 
                                insp_2cmp_prop)],
                                categorical =[:doselevel, :isPM, 
                                              :isfed,     :sex])
#                 



## Manual assessment 
# 1. Parameter Plausability 
metrics_2cmp_prop = metrics_table(pkfit_2cmp_prop)
    rename!(metrics_2cmp_prop,:Value => :TWO_CMP_PROP)
df_comp = innerjoin(metrics_1cmp_comb,metrics_2cmp_prop, on=:Metric, makeunique=true)


comp_est = compare_estimates(;pkfit_1cmp_comb, pkfit_2cmp_prop)


# 2. CWRES Distribution 
figure1 = Figure(resolution = (1500, 500); fontsize = 20)
wresiduals_dist(figure1[1,1], insp_1cmp_comb; observations = :dv)
wresiduals_dist(figure1[1,2], insp_2cmp_prop; observations = :dv)
figure1
#save("wres_dist_cmps.png", figure1)



# 3. CWRES vs TIME 
figure2 = Figure(resolution = (1500, 500); fontsize = 20)
wresiduals_vs_time(figure2[1,1], insp_1cmp_comb; observations = :dv)
wresiduals_vs_time(figure2[1,2], insp_2cmp_prop; observations = :dv)
figure2
#save("wres_time_cmps.png", figure2)



# 4. Individual Subject Fits 
pred_2cmp_prop = predict(pkfit_2cmp_prop, obstimes=0:0.1:72)

#
figures_2cmp_prop = 
    subject_fits(
        pred_2cmp_prop;
        observations=[:dv],
        axis = (xlabel = "Time (hour)", 
                ylabel = "Concentration (mcg/L)",),
                #xticks = [0,2,4,6,8],
                #limits = (0, 10, nothing, nothing),
                #yscale = Makie.pseudolog10,),
        separate = true,
        paginate = true,
        limit=4,
        facet = (combinelabels = true,),)
#
figures_1cmp_comb[1]
figures_2cmp_prop[1]




# 5. GOF - Observed vs Individual Predicted
figure3 = Figure(resolution = (1500, 500); fontsize = 20)
observations_vs_ipredictions(figure3[1,1], insp_1cmp_comb; observations = :dv)
observations_vs_ipredictions(figure3[1,2], insp_2cmp_prop; observations = :dv)
figure3
#save("dv_ipred_cmps.png", figure3)


# 6. GOF - Observed vs Population Predicted 
figure4 = Figure(resolution = (1500, 500); fontsize = 20)
observations_vs_predictions(figure4[1,1], insp_1cmp_comb; observations = :dv)
observations_vs_predictions(figure4[1,2], insp_2cmp_prop; observations = :dv)
figure4
#save("dv_pred_cmps.png", figure4)


# Final structural model selected: 
# 2CMP WITH COMBINATION RUV & LINEAR ABS/ELIM 












#################################################
#                                               #
#           Simulations of Base Model           #
#                                               #
#################################################

# create a subject - given a single dose (60 mg)
dose = DosageRegimen(60000, time = 0, route=NCA.EV)
subj_1 = Subject(id = 1,
            events = dose,
            covariates = (wt = 70, doselevel=60, age = 40, sex="male", isfed="yes", isPM="yes"),
            observations = (dv = nothing,
                            cp = nothing))
#
param = coef(pkfit_2cmp_prop)
obs = simobs(mdl_2cmp, subj_1, param, obstimes = 0:0.1:120)

# simulation w/o RUV 
sim_plot(obs; observations=[:cp], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (hr)", 
               ylabel = "Predicted Concentration (mg/L)", 
               title = "Base Model: One Subject"))
# 









# create a subject - given multiple doses dose 
dose = DosageRegimen(60000, time = 0, route=NCA.EV, ii = 24, addl = 4)
subj_2 = Subject(id = 2,
            events = dose,
            covariates = (wt = 70, doselevel=60, age = 40, sex="male", isfed="yes", isPM="yes"),
            observations = (dv = nothing,
                            cp = nothing))
#
param = coef(pkfit_2cmp_prop)
obs = simobs(mdl_2cmp, subj_2, param, obstimes = 0:0.1:120)

# simulation w/o RUV 
sim_plot(obs; observations=[:cp], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (hr)", 
               ylabel = "Predicted Concentration (mg/L)", 
               title = "Base Model: One Subject"))
# 











# combine two subjects to form a population 
pop_2subjs = [subj_1, subj_2]
#typeof(subj_1)
#typeof(pop_2subjs)
param = coef(pkfit_2cmp_prop)
obs = simobs(mdl_2cmp, pop_2subjs, param, obstimes = 0:0.1:120)
f,a = sim_plot(obs; observations=[:cp], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (hr)", 
               ylabel = "Predicted Concentration (mg/L)", 
               title = "Base Model: Two Subjects"),
               color = :redsblues)
axislegend(a) 
f
# 










# create a population of random covariates 
dose = DosageRegimen(60000, time = 0, route=NCA.EV)
choose_covariates() = (wt = rand(55:80), doselevel=60, age = rand(20:60), sex=rand(["male", "female"]), isfed=rand(["yes", "no"]), isPM=rand(["yes", "no"]))
subj_with_covariates = map(1:10) do i
    Subject(id = i,
            events = dose,
            covariates = choose_covariates(),
            observations = (dv = nothing,
                            cp = nothing))
end


param = coef(pkfit_2cmp_prop)
obs = simobs(mdl_2cmp, subj_with_covariates, param, obstimes = 0:0.1:72)


# simulation w/ RUV 
sim_plot(obs; observations=[:dv], 
                figure=(; fontsize = 18), 
                #linewidth = 4,
                 axis=(;
               xlabel = "Time (hr)", 
               ylabel = "Predicted Concentration (mg/L)", 
               title = "Base Model"))
# 
# simulation w/o RUV 
sim_plot(obs; observations=[:cp], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (hr)", 
               ylabel = "Predicted Concentration (mg/L)", 
               title = "Base Model"))
#

# 1 subject: 
sim_plot(obs[1]; observations=[:cp], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (hr)", 
               ylabel = "Predicted Concentration (mg/L)", 
               title = "Base Model: Subject #1"))
#









