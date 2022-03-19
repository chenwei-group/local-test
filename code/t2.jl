using JLD
data = load("cancerData.jld")
n = data["n"]
nTest = data["nTest"]

# Compute number of groups (here, the number of training and testing groups is the same)
k = size(n,1)

# Function to compute NLL when we have a theta for each group
NLLs(theta,n) =
begin
    LL = 0
    for j in 1:k
        LL -= n[j,1]*log(theta[j]) + (n[j,2]-n[j,1])*log(1-theta[j])
    end
    return LL
end

# Show test NLL if all theta=0.5
theta = 0.5*ones(k)
@show NLLs(theta,nTest)

# Fit MLE for each training group
MLEs = n[:,1]./n[:,2]
# Show training and test NLL for MLE
@show NLLs(MLEs,nTest)

# debug
n
n[:,1]
n[:,2]


# Compute NLL of the MLE Q1.2.2
MAPs = (n[:,1]+(2-1)*ones(k))./((n[:,1]+(2-1)*ones(k))+(n[:,2]-n[:,1]+(2-1)*ones(k)))
@show NLLs(MAPs,nTest)


# Compute NLL of the posterior predictive probabilities Q1.2.3
p = (n[:,1]+2*ones(k))./((n[:,1]+2*ones(k))+(n[:,2]-n[:,1]+2*ones(k)))
@show NLLs(p,nTest)


# Q1.2.4
NLLs_single(theta,n) =
begin
    return - (sum(n[:,1])*log(theta) + sum(n[:,2]-n[:,1])*log(1-theta))
end

# MLEs
MLEs_single= sum(n[:,1]) / sum(n[:,2])
@show NLLs_single(MLEs_single,nTest)

# MAPs
MAPs_single= (sum(n[:,1])+2-1) / ((sum(n[:,1])+2-1)+sum(n[:,2]-n[:,1])+2-1)
@show NLLs_single(MAPs_single,nTest)

# Posterior predictive probabilities
p_single= (sum(n[:,1])+2) / ((sum(n[:,1])+2)+(sum(n[:,2]-n[:,1])+2))
@show NLLs_single(p_single,nTest)


# Q1.2.5
#using Pkg
#Pkg.add("SpecialFunctions")
using SpecialFunctions

n1 = sum(n[:,1])
n0 = sum(n[:,2]) - n1

# Logarithm of marginal likelihood function
logMargLik(a,b) =
begin
    return -(logbeta(a+n1,b+n0)-logbeta(a,b))
end

# fix m=0.2 and increasing k
#a=m*k
#b=(1-m)*k
m = 0.2
LM = []
for k in 1:100
    append!(LM, logMargLik(m*k,(1-m)*k))
end

using PyPlot
plot(1:100,LM)
display(gcf())


# Q1.2.6
using SpecialFunctions

n1 = sum(n[:,1])
n0 = sum(n[:,2]) - n1

# Logarithm of marginal likelihood function
logMargLik(a,b) =
begin
    return logbeta(a+n1,b+n0)-logbeta(a,b)
end

# Search objective function
SearchObj(a,b)=
begin
    m = a/(a+b)
    k = a+b
    pm = m^(0.01-1)*(1-m)^(9.9-1)
    pk = 1/(1+k)^2
    return logMargLik(a,b) + log(pm) + log(pk)
end

# find best alpha and beta values
alpha = NaN
beta = NaN
Obj=-Inf
for a in 0.1:0.1:9
    for b in 0.1:0.1:9
        if Obj < SearchObj(a,b)
            Obj = SearchObj(a,b)
            alpha = a
            beta = b
        end
    end
end

@show(Obj)
@show(alpha)
@show(beta)
@show(m = alpha/(alpha+beta))
@show(k=alpha+beta)

# Q1.2.7
# Search objective function
K = size(n,1)

# Search objective function
ReglogMargLik(a,b)=
begin
    m = a/(a+b)
    k = a+b
    pm = m^(0.01-1)*(1-m)^(9.9-1)
    pk = 1/(1+k)^2
    RegLL=0
    for j in 1:K
        RegLL += (logbeta(a+n[j,1],b+(n[j,2]-n[j,1])) - logbeta(a,b))
    end
    ObjReg = RegLL + log(pm) + log(pk) 
    return ObjReg
end

# find best alpha and beta values
alpha = NaN
beta = NaN
Obj=-Inf
for a in 1:1000
    for b in 1:1000
        if Obj < ReglogMargLik(a,b)
            Obj = ReglogMargLik(a,b)
            alpha = a
            beta = b
        end
    end
end

@show(alpha)
@show(beta)
@show(Obj)

p_ = (n[:,1]+alpha*ones(K))./((n[:,1]+alpha*ones(K))+(n[:,2]-n[:,1]+beta*ones(K)))
@show NLLs(p_,nTest)


# Q1.2.9
MAPs_ = (n[:,1]+(alpha-1)*ones(k))./((n[:,1]+(alpha-1)*ones(k))+(n[:,2]-n[:,1]+(beta-1)*ones(k)))
@show NLLs(MAPs_,nTest)