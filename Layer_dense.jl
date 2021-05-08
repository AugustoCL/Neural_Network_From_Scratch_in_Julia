### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ be7baa4f-6871-4d74-b4d2-1c2534cdfa17
# Row Orientation
begin
	using Random
	Random.seed!(0)
	
	struct Layer_dense
		w::Matrix{Float64}
		b::Matrix{Float64}
		σ::Function
	
		function Layer_dense(n_in::Int, n_out::Int; σ::Function = identity)
			w = 0.01 .* randn(n_out, n_in)
			b = reshape(zeros(n_out), (1, n_out)) 
			return new(w, b, σ)
		end
	end

	function (L::Layer_dense)(input) 
		W, b, σ = L.w, L.b, L.σ
		σ( (input * W') .+ b )
	end
end

# ╔═╡ c4ea1432-abb4-11eb-1cf5-edac0735d67d
md"# LayerDense"

# ╔═╡ 45d31083-883b-4e84-913e-c040c6fdfd5f
md"""
>No futuro, podemos pensar em novas formas de fazer esse objeto. Por exemplo, passando a matriz W como input construtor do layer.
"""

# ╔═╡ dc76be1b-292e-4545-887e-1e62faf8e1b6
md"""
### Column Orientation
"""

# ╔═╡ fe237f5f-8f90-429a-8ddc-e5933b8fd808
begin
	Random.seed!(0)
	
	struct LayerDense{T<:Real}
		W::Matrix{T}
		b::Vector{T}
		σ::Function
	
		function LayerDense(W::Matrix{T}, b::Vector{T}, σ::Function = identity) where {T<:Real}
			return new{T}(W, b, σ)
		end
	end
	
	function LayerDense(W::Matrix{T}, b::Vector{S}, σ::Function = identity) where {T<:Real, S<:Real}
		R = promote_type(T, S)
		return LayerDense(Matrix{R}(W), Vector{R}(b), σ)
	end
	
	function LayerDense(n_in::Int, n_out::Int, σ::Function = identity)
		W = 0.01 .* randn(n_out, n_in)
		b = zeros(n_out) 
		return LayerDense(W, b, σ)
	end
	
	function (L::LayerDense)(input::Vector{<:Real})
		W, b, σ = L.W, L.b, L.σ
		σ(W * input + b)
	end
	
	function (L::LayerDense)(input::Matrix{<:Real})
		W, b, σ = L.W, L.b, L.σ
		σ(W * input' .+ b)
	end
end

# ╔═╡ cda8baaf-0be6-47b7-b267-7ae084177251
promote_type(Int, Float64) #Promoção de tipo

# ╔═╡ 520df978-ec7d-4dff-a0cc-2ca2db79b491
a = [1, 2]

# ╔═╡ 1e7a5834-5938-4429-8fd4-c447ff5a02e3
b = [1.0 4.1
	 2.2 5.4]

# ╔═╡ d45caa30-c2c7-461f-b6cb-d948b95e22de
A = LayerDense(b, a)

# ╔═╡ 6bff6f83-dcd5-44bb-b707-2890aa557deb
A([5, 6])

# ╔═╡ e85bc0f7-65ab-4a41-a4ba-e63c8b7c8184
B = LayerDense(4, 3)

# ╔═╡ 142bb8f9-4188-46d2-94a2-8407d633c496
input = [2 3 4 5
		 5 3 5 7] # cada observação deve estar em uma linha.

# ╔═╡ 2377f014-0bd0-457b-b9e9-3ace1ad7c331
B(input) # cada coluna é um output

# ╔═╡ c99c7f4f-7e62-4036-ba08-4478e5480d19
md"### Adding new sigmoid functions"

# ╔═╡ 58265283-a36a-4c83-9101-3387165da021
begin
	function σ(x::Real) 
		t = exp(-abs(x))
		return ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
	end	
	
	σ(x::AbstractVecOrMat{T}) where {T<:Real} = σ.(x)
	
	const sigmoid = σ
end

# ╔═╡ 45bafada-34ec-4c41-b0a2-000f35ebffef
begin
	relu(x::Real) = max(zero(x), x) # zero function keep the type of x
	relu(x::AbstractVecOrMat{T}) where {T<:Real} = relu.(x)
end

# ╔═╡ d6fce490-1498-469d-8ecc-55d0a71d1565
begin
	softplus(x::Real) = log((1+ℯ^x))
	softplus(x::AbstractVecOrMat{T}) where {T<:Real} = softplus.(x)
end

# ╔═╡ 04cadf6a-4fd1-4b5f-aa4d-140ea6e5cff2
function softmax(x::Vector{T}) where {T<:Real}
	m = maximum(x)
	exp_val = exp.(x .- m)
	s = sum(exp_val)
	return exp_val ./ s
end

# ╔═╡ a7898ffd-1acb-4220-bbc8-3b428b95db85
function softmax(x::Matrix{T}) where {T<:Real}
	m = maximum.(eachcol(x))
	exp_val = exp.(x .- m')
	s = sum(eachrow(exp_val))
	return exp_val ./ s'
end

# ╔═╡ c116a1e4-fd4d-446f-bdea-02301f43b2c0
md"**Aplying some sigmoid functions**"

# ╔═╡ d0c91a1c-8798-4985-978b-4c47d1abc212
C = LayerDense(2, 3, σ)

# ╔═╡ 3de0841e-f0f3-436c-a891-dd4af32d2af6
#E(randn(4,3))
C([1 4; 2 3; 3 5])

# ╔═╡ 963f6276-2440-4bfe-ae0f-539d8bfae0a2
D = LayerDense(2, 3, softmax)

# ╔═╡ 1499b700-960a-4075-aa89-f5780be39728
begin
	function CE(outp, target)
		R = []
		for (i,d) in zip(target, eachcol(outp))
			push!(R, d[i])
		end
		return R
	end

	outp = D([1 4; 2 3; 3 5])
	target = [1, 2, 2]
	R = []
	for (i,d) in zip(target, eachcol(outp))
		push!(R, d[i])
	end

	(outp, R)
end

# ╔═╡ e3ec0333-14ad-42a8-9cd3-4e3937dc5ff0
begin
	outp1 = [
		0.7 0.1 0.2
		0.1 0.5 0.4
		0.02 0.9 0.08
	]
	target1 = [
		1 0 0
		0 1 0
		0 1 0
	]
	
	#( outp1, target1')
	outp1*target1'
	#sum(outp1*target1', dims=2)
end

# ╔═╡ f4b146c1-af4a-4c3e-860d-1f92c4c13a6e
( D([1 4; 2 3; 3 5]), sum(eachrow(D([1 4; 2 3; 3 5]))) )

# ╔═╡ Cell order:
# ╟─c4ea1432-abb4-11eb-1cf5-edac0735d67d
# ╟─45d31083-883b-4e84-913e-c040c6fdfd5f
# ╟─dc76be1b-292e-4545-887e-1e62faf8e1b6
# ╠═fe237f5f-8f90-429a-8ddc-e5933b8fd808
# ╠═cda8baaf-0be6-47b7-b267-7ae084177251
# ╠═520df978-ec7d-4dff-a0cc-2ca2db79b491
# ╠═1e7a5834-5938-4429-8fd4-c447ff5a02e3
# ╠═d45caa30-c2c7-461f-b6cb-d948b95e22de
# ╠═6bff6f83-dcd5-44bb-b707-2890aa557deb
# ╠═e85bc0f7-65ab-4a41-a4ba-e63c8b7c8184
# ╠═142bb8f9-4188-46d2-94a2-8407d633c496
# ╠═2377f014-0bd0-457b-b9e9-3ace1ad7c331
# ╟─c99c7f4f-7e62-4036-ba08-4478e5480d19
# ╠═58265283-a36a-4c83-9101-3387165da021
# ╠═45bafada-34ec-4c41-b0a2-000f35ebffef
# ╠═d6fce490-1498-469d-8ecc-55d0a71d1565
# ╠═04cadf6a-4fd1-4b5f-aa4d-140ea6e5cff2
# ╠═a7898ffd-1acb-4220-bbc8-3b428b95db85
# ╟─c116a1e4-fd4d-446f-bdea-02301f43b2c0
# ╠═d0c91a1c-8798-4985-978b-4c47d1abc212
# ╠═3de0841e-f0f3-436c-a891-dd4af32d2af6
# ╠═963f6276-2440-4bfe-ae0f-539d8bfae0a2
# ╠═1499b700-960a-4075-aa89-f5780be39728
# ╠═e3ec0333-14ad-42a8-9cd3-4e3937dc5ff0
# ╠═f4b146c1-af4a-4c3e-860d-1f92c4c13a6e
# ╟─be7baa4f-6871-4d74-b4d2-1c2534cdfa17
