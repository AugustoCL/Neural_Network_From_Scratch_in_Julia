### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 0a5335e0-ab9e-11eb-0a85-139ff5cf0b1d
md"# Chapter 02 from NNFS book"

# ╔═╡ 0c5d667f-0ea5-4d21-80ad-58d144470a73
md"### A Single Neuron (4 inputs, 1 output)"

# ╔═╡ d85b097d-c7c7-439f-80ec-a1b75c696cb1
md""" 
for **each input** we need **one weigth** and for **each output** we need **one bias**.
So, for this simple example with 1 neuron with 4 inputs we are gonna have:

- **3** inputs
- **3** weights
- **1** bias.
"""

# ╔═╡ ef746509-f29b-42ae-b9ec-a2e18ad4b327
begin
	inputs = [1; 2; 3; 2.5]
	weights = [0.2; 0.8; -0.5; 1.0]
	bias = 2.0

	inputs'*weights + bias # or dot(inputs, weights) + bias
end

# ╔═╡ a3acafe8-6a4b-4438-84d9-16a00cb21997
md"### A Layer of 03 Neurons (4 inputs, 3 outputs)"

# ╔═╡ 9fbc1e4c-52c0-4b99-92dc-4969b55f645d
begin
	#inputs = [1, 2, 3, 2.5]
	weights1 = [0.2, 0.8, -0.5, 1.0]
	weights2 = [0.5, -0.91, 0.26, -0.5]
	weights3 = [-0.26, -0.27, 0.17, 0.87]
	bias1 = 2.0
	bias2 = 3.0
	bias3 = 0.5

	[inputs'*weights1 + bias1,
	inputs'*weights2 + bias2,
	inputs'*weights3 + bias3]
end

# ╔═╡ 3613059e-501d-4cb0-bd73-8e42e6395e16
md" with only one unique matrix of weights we use matrix multiplication from linear algebra"

# ╔═╡ 0e309e2a-59e8-4fce-819d-afeec8c21ca8
begin
	weights_array = [
		0.2    0.8  -0.5   1.0;
		0.5   -0.91  0.26 -0.5;
		-0.26 -0.27  0.17  0.87;
	]
	bias_vec = [2.0 3.0 0.5]
	
	(weights_array * inputs) .+ bias_vec
end

# ╔═╡ d27568ac-934c-4408-880d-4bd34ccead72
md"### A Layer of Neurons with multiple batch of Data"

# ╔═╡ 91d6b3e6-5124-4963-b6c2-f2c65b67b045
md"Now we are gonna need use transpose operation to match the right vectors in matrix multiplication"

# ╔═╡ 7ae39d13-e448-4c91-81be-d44a15f1a414
begin
	inputs_batch = [
		1.0 2.0 3.0 2.5
		2.0 5.0 -1.0 2.0
		-1.5 2.7 3.3 -0.8
	]
	
	(inputs_batch * weights_array') .+ bias_vec 
end

# ╔═╡ Cell order:
# ╟─0a5335e0-ab9e-11eb-0a85-139ff5cf0b1d
# ╟─0c5d667f-0ea5-4d21-80ad-58d144470a73
# ╟─d85b097d-c7c7-439f-80ec-a1b75c696cb1
# ╠═ef746509-f29b-42ae-b9ec-a2e18ad4b327
# ╟─a3acafe8-6a4b-4438-84d9-16a00cb21997
# ╠═9fbc1e4c-52c0-4b99-92dc-4969b55f645d
# ╟─3613059e-501d-4cb0-bd73-8e42e6395e16
# ╠═0e309e2a-59e8-4fce-819d-afeec8c21ca8
# ╟─d27568ac-934c-4408-880d-4bd34ccead72
# ╟─91d6b3e6-5124-4963-b6c2-f2c65b67b045
# ╠═7ae39d13-e448-4c91-81be-d44a15f1a414
