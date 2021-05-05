### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 9522185e-2666-4cf9-a228-a9ed2f1fd4f5
using LinearAlgebra

# ╔═╡ 0a5335e0-ab9e-11eb-0a85-139ff5cf0b1d
md"# Chapter 02 from NNFS book"

# ╔═╡ 0c5d667f-0ea5-4d21-80ad-58d144470a73
md"### A Single Neuron (4 inputs, 1 output)"

# ╔═╡ d85b097d-c7c7-439f-80ec-a1b75c696cb1
md""" 
For **each input** we need **one weigth** and for **each output** we need **one bias**.
So, for this simple example with 1 neuron with 4 inputs we are gonna have:

- **3** inputs
- **3** weights
- **1** bias.
"""

# ╔═╡ abb7a21d-9048-4d09-8b41-cdd519773756
md"""
#### Introduction the notation
"""

# ╔═╡ 99084569-d10d-43c7-bda3-361bc0c9881b
md"""
The output of the neuron $$i$$, $$a^{(L)}_i$$, is equal to the scalar product of weights vector $$W_i$$ by the input vector $$a^{(L-1)}$$, plus bias $$b_i$$.\
"""

# ╔═╡ 73f22e47-8671-44a9-ab74-786b281b9d69
md"""
$$a_i^{(L)} = W_i \cdot a^{(L - 1)} + b_i$$
"""

# ╔═╡ 0ebd7094-37ab-4151-b874-58b6347eb0d9
md"""
For a number $$j$$ of inputs and a number $$k$$ from outputs, each weight from $$k$$ must be in a line in $$W$$. This way, the column vector of outputs is equal to matrix multiplication of weights by the column vector inputs plus bias.\
"""

# ╔═╡ 37bbfcde-9c52-436a-96ef-bacf158ef5bd
md"""
$$a_{k × 1}^{(L)} = W_{k × j}×a_{j × 1}^{(L - 1)} + b_{k × 1}$$
"""

# ╔═╡ 280badb4-b649-4805-b84d-b18797555c2b
md"""
Applying that notation in our first layer example of 4 inputs and 1 output we got:

$$a_1^{(2)} = W_1 \cdot a^{(1)} + b_1$$
"""

# ╔═╡ ef746509-f29b-42ae-b9ec-a2e18ad4b327
begin
	inputs = [1, 2, 3, 2.5]
	weights = [0.2, 0.8, -0.5, 1.0]
	bias = 2.0

	weights ⋅ inputs + bias # \cdot<tab> or dot(w, i)
end

# ╔═╡ a3acafe8-6a4b-4438-84d9-16a00cb21997
md"### A Layer of 03 Neurons (4 inputs, 3 outputs)"

# ╔═╡ cf00a188-0f6c-484b-b793-5b1cb9f08210
md"""
$$a_1^{(2)} = W_1 \cdot a^{(1)} + b_1$$
$$a_2^{(2)} = W_2 \cdot a^{(1)} + b_2$$
$$a_3^{(2)} = W_3 \cdot a^{(1)} + b_2$$
"""

# ╔═╡ 9fbc1e4c-52c0-4b99-92dc-4969b55f645d
begin
	#inputs = [1, 2, 3, 2.5]
	weights1 = [0.2, 0.8, -0.5, 1.0]
	weights2 = [0.5, -0.91, 0.26, -0.5]
	weights3 = [-0.26, -0.27, 0.17, 0.87]
	bias1 = 2.0
	bias2 = 3.0
	bias3 = 0.5

	[weights1 ⋅ inputs + bias1,
	 weights2 ⋅ inputs + bias2,
	 weights3 ⋅ inputs + bias3]
end

# ╔═╡ d96ad6e1-65de-4150-8fdd-579b8653dd39
md"""
Applying the notation in that layer with 4 inputs and 3 output we got:

$$a_{3 × 1}^{(2)} = W_{3 × 4}×a_{4 × 1}^{(1)} + b_{3 × 1}$$
"""

# ╔═╡ 3613059e-501d-4cb0-bd73-8e42e6395e16
md" With only one unique matrix of weights we use matrix multiplication."

# ╔═╡ 0e309e2a-59e8-4fce-819d-afeec8c21ca8
begin
	weights_matrix = [
		 0.2   0.8  -0.5   1.0
		 0.5  -0.91  0.26 -0.5
		-0.26 -0.27  0.17  0.87
	]
					  
	bias_vec = [2.0, 3.0, 0.5]
	
	weights_matrix * inputs + bias_vec
end

# ╔═╡ d27568ac-934c-4408-880d-4bd34ccead72
md"### A Layer of Neurons with multiple batch of Data"

# ╔═╡ 91d6b3e6-5124-4963-b6c2-f2c65b67b045
md"""
When we evaluate more than one observation (a batch of data), we'll have a matrix in the input $$a_{j × n}$$, with $$n$$ as the number of observations. 
This way we could update the notation to:

$$a_{k × n}^{(L)} = W_{k × j}×a_{j × n}^{(L - 1)} + b_{k × n}$$
"""

# ╔═╡ 56cfbb4c-6573-485c-ac2a-6d66941ffbf8
md"""
#### About the arquitecture of a layer.
We could construct the layer by two ways: 
- Observations in Row and Parameters in Column (row orientation)
- Observations in Column and Parameters in Row. (column orientation)

This happens because languages have differents orientation. In Python which is row orientation, all the code is constructed by the first way, but in other languages like Julia, or even the math textbooks, you could use the column orientation.
"""

# ╔═╡ 277b4bcc-635e-48b2-b030-b51f159f3eb3
md"""
The **first one** is like the code above.
"""

# ╔═╡ 5a066946-e2f0-48c7-9b4b-6089caca52fb
begin
	inputs_batch01 = [
		 1.0 2.0  3.0  2.5
		 2.0 5.0 -1.0  2.0
		-1.5 2.7  3.3 -0.8
	]
	
	#=
	bias_vec = [2.0, 3.0, 0.5]
	weights_matrix = [
		 0.2   0.8  -0.5   1.0
		 0.5  -0.91  0.26 -0.5
		-0.26 -0.27  0.17  0.87
		]
	=#
	
	(input = inputs_batch01,
	 output = weights_matrix * inputs_batch01' .+ bias_vec)
end

# ╔═╡ bba52778-115b-4e0d-b63b-2b1f4802736f
md"""
The **second one** is like the code above.
"""

# ╔═╡ 7ae39d13-e448-4c91-81be-d44a15f1a414
begin
	inputs_batch02 = [
		[1.0, 2.0, 3.0, 2.5] [2.0, 5.0, -1.0, 2.0] [-1.5, 2.7, 3.3, -0.8]
	]
	
	#=
	bias_vec = [2.0, 3.0, 0.5]
	weights_matrix = [
		 0.2   0.8  -0.5   1.0
		 0.5  -0.91  0.26 -0.5
		-0.26 -0.27  0.17  0.87
		]
	=#
	
	(input = inputs_batch02,
	 output = weights_matrix * inputs_batch02 .+ bias_vec)
end

# ╔═╡ Cell order:
# ╟─0a5335e0-ab9e-11eb-0a85-139ff5cf0b1d
# ╟─0c5d667f-0ea5-4d21-80ad-58d144470a73
# ╟─d85b097d-c7c7-439f-80ec-a1b75c696cb1
# ╟─abb7a21d-9048-4d09-8b41-cdd519773756
# ╟─99084569-d10d-43c7-bda3-361bc0c9881b
# ╟─73f22e47-8671-44a9-ab74-786b281b9d69
# ╟─0ebd7094-37ab-4151-b874-58b6347eb0d9
# ╟─37bbfcde-9c52-436a-96ef-bacf158ef5bd
# ╟─280badb4-b649-4805-b84d-b18797555c2b
# ╠═9522185e-2666-4cf9-a228-a9ed2f1fd4f5
# ╠═ef746509-f29b-42ae-b9ec-a2e18ad4b327
# ╟─a3acafe8-6a4b-4438-84d9-16a00cb21997
# ╟─cf00a188-0f6c-484b-b793-5b1cb9f08210
# ╠═9fbc1e4c-52c0-4b99-92dc-4969b55f645d
# ╟─d96ad6e1-65de-4150-8fdd-579b8653dd39
# ╟─3613059e-501d-4cb0-bd73-8e42e6395e16
# ╠═0e309e2a-59e8-4fce-819d-afeec8c21ca8
# ╟─d27568ac-934c-4408-880d-4bd34ccead72
# ╟─91d6b3e6-5124-4963-b6c2-f2c65b67b045
# ╟─56cfbb4c-6573-485c-ac2a-6d66941ffbf8
# ╟─277b4bcc-635e-48b2-b030-b51f159f3eb3
# ╠═5a066946-e2f0-48c7-9b4b-6089caca52fb
# ╟─bba52778-115b-4e0d-b63b-2b1f4802736f
# ╠═7ae39d13-e448-4c91-81be-d44a15f1a414
