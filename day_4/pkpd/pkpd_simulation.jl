using CSV
using DataFramesMeta
using Dates
using Pumas
using Pumas.Latexify
using PumasUtilities
using Random
using CairoMakie
using Serialization


# Indirect 
# - stimulation of loss 
# - inhibition of loss 
# - stimulation of production 
# - inhibition of production 


#################################################
#                                               #
#             Stimulation of Loss               #
#                                               #
#################################################

mdl_pkpd_insulin = @model begin

    @metadata begin
        desc = "PKPD Model: SubQ Insulin Effect on Glucose"
        timeu = u"hr"
    end

    @param begin
        ##### PK #####
        "Clearance (L/min)"
        tvcl  ∈ RealDomain(lower = 0.0001)
        "Volume (L)"
        tvvc  ∈ RealDomain(lower = 0.0001)
        "First-order absorption rate constant (1/min)"
        tvka  ∈ RealDomain(lower = 0.0001)
        "Bioavailability"
        tvbio ∈ RealDomain(lower = 0.01)
        ##### PD #####
        "Basal zero-order production rate of glucose (mg/dL/min)"
        tvkprod   ∈ RealDomain(lower = 0.0001)
        "First-order glucose uptake rate constant (1/min)"
        tvkuptake ∈ RealDomain(lower = 0.0001)
        "Maximum fractional stimulation of glucose uptake from basal level"
        tvsmax    ∈ RealDomain(lower = 0.01)
        "Insulin concentration at half-maximumal stimulation (microU/mL)"
        tvsc50    ∈ RealDomain(lower = 0.0001)

        """
        - ΩCL
        - ΩVc
        - ΩKa
        - ΩF
        """
        Ω_pk ∈ PDiagDomain(4)
        """
        - Ωkprod
        - Ωkuptake
        - ΩSmax
        - ΩSC50
        """
        Ω_pd ∈ PDiagDomain(4)

        "Proportional RUV - PK"
        σ²_prop_pk ∈ RealDomain(lower = 0.0001)
        "Additive RUV - PK (μU/mL)"
        σ²_add_pk  ∈ RealDomain(lower = 0.0001)
        "Proportional RUV - PD"
        σ²_prop_pd ∈ RealDomain(lower = 0.0001)
        "Additive RUV - PD (mg/dL)"
        σ²_add_pd  ∈ RealDomain(lower = 0.0001)
    end
  
    @random begin
        η_pk   ~ MvNormal(Ω_pk)
        η_pd   ~ MvNormal(Ω_pd)
    end
  
    @pre begin
      # PK 
      CL   = tvcl * exp(η_pk[1])
      Vc   = tvvc * exp(η_pk[2])
      Ka   = tvka * exp(η_pk[3])
      # PD
      kprod   = tvkprod   * exp(η_pd[1])
      kuptake = tvkuptake * exp(η_pd[2])
      Smax    = tvsmax    * exp(η_pd[3])
      SC50    = tvsc50    * exp(η_pd[4])
    end

    @dosecontrol begin
        bioav = (Depot = tvbio* exp(η_pk[4]), Central=1.0)
    end

    @init begin
      Depot = 0
      Glucose = 120
    end
  
    @dynamics begin
      Depot'   = -Ka*Depot
      Central' =  Ka*Depot - (CL/Vc)*Central
      Glucose' =  kprod - kuptake*Glucose*(1+((Smax*(Central/Vc))/(SC50+(Central/Vc))))
    end
  
    @derived begin
      # PK
      CONC = @. Central / Vc
      """
      Insulin Concentration (μU/mL) 
      """
      DV   ~ @. Normal(CONC, sqrt(((abs(CONC)^2)*σ²_prop_pd) + σ²_add_pd))
      # PD
      CONC_GLUCOSE = @. Glucose
      """
      Glucose Concentration (mg/dL) 
      """
      DV_GLUCOSE ~ @. Normal(CONC_GLUCOSE, sqrt(((abs(CONC_GLUCOSE)^2)*σ²_prop_pd) + σ²_add_pd))
    end
end



params_pkpd_insulin = ( # PK 
                tvcl = 0.014,
                tvvc = 0.72,
                tvka = 0.34,
                tvbio = 0.64,
                #Ω_pk = Diagonal([0.348,0.71,0.34,0.7]),
                Ω_pk = Diagonal([0.013,0.023,0.015,0.021]),
                σ²_add_pk = 0.1, 
                σ²_prop_pk = 0.1,
                # PD 
                tvkprod = 0.81,
                tvkuptake = 0.007,
                tvsmax = 5.8, 
                tvsc50 = 80,
                Ω_pd = Diagonal([0.013,0.023,0.015,0.021]),
                σ²_add_pd = 0.1, 
                σ²_prop_pd = 0.1)
#


# subcutaneou insulin dose: 150 μU
dose = DosageRegimen(150, time = 0, route=NCA.EV)
pop_pkpd_insulin = map(1:50) do i
    Subject(id = i,
            events = dose,
            #covariates = choose_covariates(),
            observations = (CONC = nothing,
                            DV = nothing,
                            CONC_GLUCOSE = nothing, 
                            DV_GLUCOSE = nothing))
end


obs_pkpd_insulin = simobs(mdl_pkpd_insulin, pop_pkpd_insulin, params_pkpd_insulin, obstimes = 0:1:300)
# simulation w/o RUV 
f = sim_plot(obs_pkpd_insulin; observations=[:CONC], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (min)", 
               ylabel = "Predicted Insulin Concentration (μU/mL)", 
               title = "Insulin Effect on Glucose"))
g = sim_plot(obs_pkpd_insulin; observations=[:CONC_GLUCOSE], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (min)", 
               ylabel = "Predicted Glucose Concentration (mg/dL)", 
               title = "Insulin Effect on Glucose"))
sim_plot(obs_pkpd_insulin[1]; observations=[:CONC, :CONC_GLUCOSE], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (min)", 
               ylabel = "Predicted Concentrations", 
               title = "Insulin Effect on Glucose"),
               color = :redsblues)
#




# Insulin decreases glucose levels 
# Glucagon increases glucose levels 






#################################################
#                                               #
#          Stimulation of Production            #
#                                               #
#################################################

mdl_pkpd_glucagon = @model begin

    @metadata begin
        desc = "PKPD Model: Glucagon Effect on Glucose"
        timeu = u"hr"
    end

    @param begin
        ##### PK #####
        "Clearance (L/min)"
        tvcl  ∈ RealDomain(lower = 0.0001)
        "Volume (L)"
        tvvc  ∈ RealDomain(lower = 0.0001)
        "First-order absorption rate constant (1/min)"
        tvka  ∈ RealDomain(lower = 0.0001)
        "Bioavailability"
        tvbio ∈ RealDomain(lower = 0.01)
        ##### PD #####
        "Basal zero-order production rate of glucose (mg/dL/min)"
        tvkprod   ∈ RealDomain(lower = 0.0001)
        "First-order glucose uptake rate constant (1/min)"
        tvkuptake ∈ RealDomain(lower = 0.0001)
        "Maximum fractional stimulation of glucose production from basal level"
        tvsmax    ∈ RealDomain(lower = 0.01)
        "Glucagon concentration at half-maximumal stimulation (microU/mL)"
        tvsc50    ∈ RealDomain(lower = 0.0001)

        """
        - ΩCL
        - ΩVc
        - ΩKa
        - ΩF
        """
        Ω_pk ∈ PDiagDomain(4)
        """
        - Ωkprod
        - Ωkuptake
        - ΩSmax
        - ΩSC50
        """
        Ω_pd ∈ PDiagDomain(4)

        "Proportional RUV - PK"
        σ²_prop_pk ∈ RealDomain(lower = 0.0001)
        "Additive RUV - PK (μU/mL)"
        σ²_add_pk  ∈ RealDomain(lower = 0.0001)
        "Proportional RUV - PD"
        σ²_prop_pd ∈ RealDomain(lower = 0.0001)
        "Additive RUV - PD (mg/dL)"
        σ²_add_pd  ∈ RealDomain(lower = 0.0001)
    end
  
    @random begin
        η_pk   ~ MvNormal(Ω_pk)
        η_pd   ~ MvNormal(Ω_pd)
    end
  
    @pre begin
      # PK 
      CL   = tvcl * exp(η_pk[1])
      Vc   = tvvc * exp(η_pk[2])
      Ka   = tvka * exp(η_pk[3])
      # PD
      kprod   = tvkprod   * exp(η_pd[1])
      kuptake = tvkuptake * exp(η_pd[2])
      Smax    = tvsmax    * exp(η_pd[3])
      SC50    = tvsc50    * exp(η_pd[4])
    end

    @dosecontrol begin
        bioav = (Depot = tvbio* exp(η_pk[4]), Central=1.0)
    end

    @init begin
      Depot = 0
      Glucose = 20
    end
  
    @dynamics begin
      Depot'   = -Ka*Depot
      Central' =  Ka*Depot - (CL/Vc)*Central
      Glucose' =  kprod*(1+((Smax*(Central/Vc))/(SC50+(Central/Vc)))) - kuptake*Glucose
    end
  
    @derived begin
      # PK
      CONC = @. Central / Vc
      """
      Glucagon Concentration (μU/mL) 
      """
      DV   ~ @. Normal(CONC, sqrt(((abs(CONC)^2)*σ²_prop_pd) + σ²_add_pd))
      # PD
      CONC_GLUCOSE = @. Glucose
      """
      Glucose Concentration (mg/dL) 
      """
      DV_GLUCOSE ~ @. Normal(CONC_GLUCOSE, sqrt(((abs(CONC_GLUCOSE)^2)*σ²_prop_pd) + σ²_add_pd))
    end
end

params_pkpd_glucagon = ( # PK 
                tvcl = 0.014,
                tvvc = 0.72,
                tvka = 0.34,
                tvbio = 0.64,
                #Ω_pk = Diagonal([0.348,0.71,0.34,0.7]),
                Ω_pk = Diagonal([0.013,0.023,0.015,0.021]),
                σ²_add_pk = 0.1, 
                σ²_prop_pk = 0.1,
                # PD 
                tvkprod = 0.81,
                tvkuptake = 0.007,
                tvsmax = 5.8, 
                tvsc50 = 80,
                Ω_pd = Diagonal([0.013,0.023,0.015,0.021]),
                σ²_add_pd = 0.1, 
                σ²_prop_pd = 0.1)
#


# subcutaneou insulin dose: 150 μU
dose = DosageRegimen(150, time = 0, route=NCA.EV)
pop_pkpd_glucagon = map(1:50) do i
    Subject(id = i,
            events = dose,
            #covariates = choose_covariates(),
            observations = (CONC = nothing,
                            DV = nothing,
                            CONC_GLUCOSE = nothing, 
                            DV_GLUCOSE = nothing))
end

obs_pkpd_glucagon = simobs(mdl_pkpd_glucagon, pop_pkpd_glucagon, params_pkpd_glucagon, obstimes = 0:1:300)
# simulation w/o RUV 
f = sim_plot(obs_pkpd_glucagon; observations=[:CONC], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (min)", 
               ylabel = "Predicted Glucagon Concentration (μU/mL)", 
               title = "Glucagon Effect on Glucose"))
g = sim_plot(obs_pkpd_glucagon; observations=[:CONC_GLUCOSE], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (min)", 
               ylabel = "Predicted Glucose Concentration (mg/dL)", 
               title = "Glucagon Effect on Glucose"))
sim_plot(obs_pkpd_glucagon[1]; observations=[:CONC, :CONC_GLUCOSE], 
                figure=(; fontsize = 18), 
                linewidth = 4,
                 axis=(;
               xlabel = "Time (min)", 
               ylabel = "Predicted Concentrations", 
               title = "Glucagon Effect on Glucose"),
               color = :redsblues)
#










