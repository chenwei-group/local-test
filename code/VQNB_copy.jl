include("misc.jl") # Includes GenericModel typedef
include("kMeans.jl")
# Load data


function VQNB_copy(X,y,k)
	# Implementation of generative classifier,
	# where a product of Bernoullis is used for p(x,y)

	(n,d) = size(X)

    #model = kMeans(X,k)	
    #z = model.predict(X)
	#@show(k)

	X3 = X[y.==1,:]
	model3 = kMeans(X3,k)	
	z3 = model3.predict(X3)

	X5 = X[y.==2,:]
    model5 = kMeans(X5,k)	
	z5 = model5.predict(X5)
	
	z = append!(z3,z5.+k)
	@show(k)
	k=2k

	# Compute p(y = 1)
	p_y = sum(y.==1)/n
	#p_y2 = sum(y.==2)/n
	# @show(p_y)
	# @show(p_y2)

	p_zy = zeros(k,2)
	for c in 1:k
		p_zy[c,1] = sum((z.==c).&&(y.==1))/sum(y.==1)
		p_zy[c,2] = sum((z.==c).&&(y.==2))/sum(y.==2)
		#@show(p_zy)
	end

	p_xyz = zeros(d,2,k)
	for j in 1:d
		for c in 1:k
			#p_xyz[j,1,c] = sum((X[:,j].==1).&&(z.==c).&&(y.==1))/sum((z.==c).&&(y.==1))
			p_xyz[j,1,c] = sum((X[:,j].>=0.5).&&(z.==c).&&(y.==1))/sum((z.==c).&&(y.==1))
			#p_xyz[j,2,c] = sum((X[:,j].==1).&&(z.==c).&&(y.==2))/sum((z.==c).&&(y.==2))
			p_xyz[j,2,c] = sum((X[:,j].>=0.5).&&(z.==c).&&(y.==2))/sum((z.==c).&&(y.==2))
		end
	end
	@show(k)
	#@show(p_xyz)
	@show(size(p_xyz))

	
	function predict(Xhat)

		(t,d) = size(Xhat)
		yhat = zeros(t)

		for i in 1:t
			prods = ones(k,2)
			for c in 1:k
				for j in 1:d
					#if Xhat[i,j] == 1
					if Xhat[i,j] >= 0.5
						prods[c,1] *= p_xyz[j,1,c]
						prods[c,2] *= p_xyz[j,2,c]
					else
						prods[c,1] *= (1-p_xyz[j,1,c])
						prods[c,2] *= (1-p_xyz[j,2,c])
					end
				end
			end
			p_yx = sum(p_zy.*prods, dims = 1)'.*[p_y;1-p_y]
		

			if p_yx[1] > p_yx[2]
				yhat[i] = 1
			else
				yhat[i] = 2
			end
		end

		return yhat
	end

	return GenericModel(predict)
	
end
