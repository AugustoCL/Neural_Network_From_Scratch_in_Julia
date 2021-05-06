### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 6e58a278-ede3-440f-b5ad-c23d03833918
using LinearAlgebra

# ╔═╡ 78b943f0-ae7b-11eb-3de1-d5cfb31433c6
function σ(x)
	t = exp(-abs(x))
	return ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end

# ╔═╡ 6fe8d0a2-fd10-4097-95c8-fea57d2a32e8
const sigmoid = σ

# ╔═╡ 5b4291e3-1f67-46ac-89f5-d00aaefdac8a
sigmoid(4)

# ╔═╡ 46487d66-1f85-4e2e-907c-ce688dc3b02f
σ(4)

# ╔═╡ 257e3608-759e-49b2-aa9f-88211bba198b
relu(x) = max(zero(x), x)

# ╔═╡ 45ba9913-5316-457e-9f97-2622661397e1
begin
	inputs = [1, 2, 3, 2.5]
	weights = [0.2, 0.8, -0.5, 1.0]
	bias = 2.0
	
	output = σ(weights ⋅ inputs + bias)
end

# ╔═╡ 2571eef9-f812-47fc-bd5a-27042ae3751c
output_relu = relu(weights ⋅ inputs + bias)

# ╔═╡ e2f8442b-5ccd-431a-8881-3e746b0eb047
begin
	#inputs = [1, 2, 3, 2.5]
	weights_matrix = [
		 0.2   0.8  -0.5   1.0
		 0.5  -0.91  0.26 -0.5
		-0.26 -0.27  0.17  0.87
	]		
	
	bias_vec = [2.0, 3.0, 0.5]
	
	outputs = σ.(weights_matrix * inputs + bias_vec)
end

# ╔═╡ 8ba34590-f8a0-469d-9b17-48dc184ea239
outputs_relu = relu.(weights_matrix * inputs + bias_vec)

# ╔═╡ afbe3c9c-6db4-415c-921d-cdc4033e9b06
function softmax(x::Vector{T}) where {T<:Real}
	exp_val = exp.(x)
	output = exp_val ./ sum(exp_val)
	return output
end

# ╔═╡ Cell order:
# ╠═78b943f0-ae7b-11eb-3de1-d5cfb31433c6
# ╠═6fe8d0a2-fd10-4097-95c8-fea57d2a32e8
# ╠═5b4291e3-1f67-46ac-89f5-d00aaefdac8a
# ╠═46487d66-1f85-4e2e-907c-ce688dc3b02f
# ╠═257e3608-759e-49b2-aa9f-88211bba198b
# ╠═6e58a278-ede3-440f-b5ad-c23d03833918
# ╠═45ba9913-5316-457e-9f97-2622661397e1
# ╠═2571eef9-f812-47fc-bd5a-27042ae3751c
# ╠═e2f8442b-5ccd-431a-8881-3e746b0eb047
# ╠═8ba34590-f8a0-469d-9b17-48dc184ea239
# ╠═afbe3c9c-6db4-415c-921d-cdc4033e9b06
