

# Call Necessary Packages 
using CSV
using Chain
using DataFrames
using DataFramesMeta 
using Dates
using Statistics
using PharmaDatasets




#################################################
#                                               #
#             JULIA LANGUAGE BASICS             #
#                                               #
#################################################

# Understand variables & types 
x = 5 
y = 3.14
c = 'a' 
s = "My String" 

typeof(x) # Int64 - integer (whole number)
typeof(y) # Float64 - number with decimal place 
typeof(c) # Char - single character (note using '' instead of "" --> can only be 1 character )
typeof(s) # String - indices of characters



#################################################
#                                               #
#             DATA FAMILIARIZATION              #
#                                               #
#################################################

# read in a dataframe from PharmaDatasets 
df = dataset("demographics_1")
typeof(df) #type dataframe - matrix with columns that are vectors 


#### indexing --> [] #####
#df[row,column]

# subset the first column 
df[:,1] 
df[!,1]
typeof(df[:,1]) 

df[!,:AGE]
typeof(df[:,:AGE]) 

n = @select df :AGE
typeof(n)
@select(df, :AGE)


@select df :ID :AGE
@select(df, :ID, :AGE)
id_age = @select(df, :ID, :AGE)

@select df $(Not(:AGE))
not_age = @select df $(Not(:AGE)) # understanding punctuation: https://docs.julialang.org/en/v1/base/punctuation/
not_age_wt = select(df,Not([:AGE,:WEIGHT]))


# subset the first row
# df[row,column]
df[1,:]
typeof(df[1,:]) 
typeof(df[!,1]) # dataframe row 

# subset the first 5 rows 
df[1:5,:]
typeof(df[1:5,:])

# indexing - extract the 2nd value in the ISMALE column 
df[!,:ISMALE][2] #indexing an indexed column
df[2,:ISMALE] # indexing both simultaneously (equivalent to above)

# preview the first 10 rows in the REPL 
first(df,10)

# preview the last 10 rows in the REPL 
last(df,10)








#################################################
#                                               #
#             SIMPLE SUMMARY STATS              #
#                                               #
#################################################

# summary statistics one one column - simple 

using StatsBase
mean(df[!,:AGE])
mean_age = mean(df[!,:AGE])

median(df[!,:AGE])
maximum(df[!,:AGE])
minimum(df[!,:AGE])
std(df[!,:AGE])

geomean(df[!,:AGE])

unique(df[!,:ISMALE])
unique(df[!,:AGE])


# summary statistics - multiple 
df_sum = describe(df, :mean, :median, :std, cols=["AGE","WEIGHT"])
df_sum = describe(df, :mean, :median, :std, cols=[:AGE,:WEIGHT])







##############################################
#                                            #
#               DATA WRANGLING              #
#                                            #
##############################################

# https://tutorials.pumas.ai/
# Section 4

# @transform 
# @combine 
# @select 
# @subset / filter 
# @orderby 

# order by a column 
df_order_wt = @orderby df :WEIGHT 
minimum(df[!,:WEIGHT])
df_order_wt_sex = @orderby df :WEIGHT :ISMALE
df_order_sex_wt = @orderby df :ISMALE :WEIGHT 

# order by a column in-place
sort!(df, [:AGE])
df

# add a column 
#df[row,column]
df[!,:COHORT].="SAD"

# add a column and conditionally fill it (transform)
df = @rtransform df :SEX = :ISMALE == 1 ? "Male" : "Female"


# subset / filter
df_male = @rsubset df :ISMALE == 1
df_female = @rsubset df :ISMALE != 1


# create a new dataframe by selecting specific columns of an already existing dataframe 
df_select = @select df :ID :ISMALE

# rename a column 
rename!(df_select,:ISMALE => :ISSMOKER)
rename!(df_select, Dict(2 => :ISMALE))
rename!(df_select, Dict(2 => "ISSMOKER"))

# vertical concatenate 
df_new = vcat(df_male,df_female) 

# horizontal concatenate
add = DataFrame(Missing_Vals = Vector{Union{Float64, Missing}}(undef, 100))
df_new2 = hcat(df_new, add)


# join by 1 column 
df_final = leftjoin(df_new, df_select, on=:ID)

# join by multiple columns 
df_select2 = @rselect df :ID :ISMALE
df_select2[!,:EXAMPLE].=df_select2[!,:ID].+1
#    sort!(df_select2,[:ID])
df_final2 = leftjoin(df_new, df_select2, on=[:ID,:ISMALE])
#df_final2 = leftjoin(df_new, df_select2, on=[:ID]) # errors 
#df_final2 = leftjoin(df_new, df_select2, on=[:ID], makeunique=true) # solution 


# other methods: 
# innerjoin
# rightjoin 
# outerjoin 
# antijoin 
# semijoin 
# crossjoin 


# change column type 
df_final[!,:ID] = string.(df_final[!,:ID])
df_final[!,:ID] = parse.(Int, df_final[!,:ID])
df_final[!,:ID] = float.(df_final[!,:ID])







##############################################
#                                            #
#               Save / Export                #
#                                            #
##############################################

# save a dataframe as a csv file 
CSV.write("df_final.csv", df_final)


# read in a csv file 
df_read = CSV.read("df_final.csv", DataFrame)


# specify the header, skip to the 3rd row (includes header)
df_read_skipped = CSV.read("df_final.csv", DataFrame, missingstring=["."], header=1, skipto=3)











##############################################
#                                            #
#                   Loops                    #
#                                            #
##############################################

# For loop 
# While loop 

df = dataset("demographics_1")


# for loop 
df[!,:COUNT].=1 
for i = 1:100 
    df[!,:COUNT][i] = i
end 



df[!,:COUNT].=1 
for i = 1:100 
    df[i,:COUNT] = i 
end 



df[!,:COUNT].=1 
for i = 1:length(df[!,:ID])
    df[i,:COUNT] = i 
end 





# while loop 
df_example = @rtransform df :SEX = :ISMALE == 0 ? "Male" : "Female"

df[!,:SEX].="UNKNOWN"
i=1
while i <= length(df[!,:ID])
    if df[!,:ISMALE][i] == 0 
        df[!,:SEX][i] = "MALE"
    else 
        df[!,:SEX][i] = "FEMALE"
    end 
    global i += 1
end











