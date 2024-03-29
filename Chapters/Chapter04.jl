### A Pluto.jl notebook ###
# v0.19.18

using Markdown
using InteractiveUtils

# ╔═╡ 6e58a278-ede3-440f-b5ad-c23d03833918
using LinearAlgebra

# ╔═╡ 81465bdf-4eb3-4dc5-8f60-2ccdca956e73
md"#### Sigmoid"

# ╔═╡ 78b943f0-ae7b-11eb-3de1-d5cfb31433c6
function σ(x)
	t = exp(-abs(x))
	return ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end

# ╔═╡ 6fe8d0a2-fd10-4097-95c8-fea57d2a32e8
const sigmoid = σ

# ╔═╡ 112f4771-632e-4c35-81bd-ec7f09b40dc0
md"#### ReLU"

# ╔═╡ 257e3608-759e-49b2-aa9f-88211bba198b
relu(x) = max(zero(x), x)

# ╔═╡ 5b4291e3-1f67-46ac-89f5-d00aaefdac8a
sigmoid(4) == σ(4)

# ╔═╡ 45ba9913-5316-457e-9f97-2622661397e1
begin
	inputs = [1, 2, 3, 2.5]
	weights = [0.2, 0.8, -0.5, 1.0]
	bias = 2.0
	
	(
		σ(weights ⋅ inputs + bias),
		relu(weights ⋅ inputs + bias)
	)
end

# ╔═╡ e2f8442b-5ccd-431a-8881-3e746b0eb047
begin
	#inputs = [1, 2, 3, 2.5]
	weights_matrix = [
		 0.2   0.8  -0.5   1.0
		 0.5  -0.91  0.26 -0.5
		-0.26 -0.27  0.17  0.87
	]		
	
	bias_vec = [2.0, 3.0, 0.5]
	
	( 
		σ.(weights_matrix * inputs + bias_vec), 
		relu.(weights_matrix * inputs + bias_vec)
	)
end

# ╔═╡ 0e4a6376-d3fa-47fc-b140-ddb65ee0c592
md"#### Softmax"

# ╔═╡ afbe3c9c-6db4-415c-921d-cdc4033e9b06
function softmax(x::Vector{T}) where {T<:Real}
	m = maximum(x)
	exp_val = exp.(x .- m)
	s = sum(exp_val)
	exp_val ./ s
end

# ╔═╡ 662251f6-03ae-4ea6-8bd1-e90ec8a10805
function softmax(x::Matrix{T}) where {T<:Real}
	m = maximum.(eachcol(x))
	exp_val = exp.(x .- m')	
	s = sum(eachrow(exp_val))
	return exp_val ./ s' 
end

# ╔═╡ 383304f8-eb30-45be-99c4-542132ecc78a
begin
	#inputs_batch = [
	#	[1, 2, 3, 2.5] [2, 5, -1, 2] [-1.5, 2.7, 3.3, -0.8]
	#]
	inputs_batch = [
		1 2 3 2.5
		2 5 -1 2
		-1.5 2.7 3.3 -0.8
	]
	
	(
		softmax(weights_matrix * inputs + bias_vec), 
		softmax(weights_matrix * inputs_batch' .+ bias_vec) # orientacao coluna
	)	
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.3"
manifest_format = "2.0"
project_hash = "ac1187e548c6ab173ac57d4e72da1620216bce54"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"
"""

# ╔═╡ Cell order:
# ╠═6e58a278-ede3-440f-b5ad-c23d03833918
# ╟─81465bdf-4eb3-4dc5-8f60-2ccdca956e73
# ╠═78b943f0-ae7b-11eb-3de1-d5cfb31433c6
# ╠═6fe8d0a2-fd10-4097-95c8-fea57d2a32e8
# ╟─112f4771-632e-4c35-81bd-ec7f09b40dc0
# ╠═257e3608-759e-49b2-aa9f-88211bba198b
# ╠═5b4291e3-1f67-46ac-89f5-d00aaefdac8a
# ╠═45ba9913-5316-457e-9f97-2622661397e1
# ╠═e2f8442b-5ccd-431a-8881-3e746b0eb047
# ╟─0e4a6376-d3fa-47fc-b140-ddb65ee0c592
# ╠═afbe3c9c-6db4-415c-921d-cdc4033e9b06
# ╠═662251f6-03ae-4ea6-8bd1-e90ec8a10805
# ╠═383304f8-eb30-45be-99c4-542132ecc78a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
