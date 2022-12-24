### A Pluto.jl notebook ###
# v0.19.18

using Markdown
using InteractiveUtils

# ╔═╡ a5ec2a7e-2dab-46c9-9013-9e7adbd44d78
using Random, Statistics, ForwardDiff

# ╔═╡ 61709a7e-18c5-4017-8e84-6eaf16ba210d
md"# Tipos e Funções"

# ╔═╡ bd149090-b5e2-11eb-34e9-ef542421d474
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
	
	function LayerDense(W::Matrix{<:Real}, σ::Function = identity)
		b = zeros(size(W)[1])
		return LayerDense(W, b, σ)
	end
	
	function LayerDense(n_in::Int, n_out::Int, σ::Function = identity)
		W = 0.01 .* randn(n_out, n_in)
		b = zeros(n_out) 
		return LayerDense(W, b, σ)
	end
	
	function (L::LayerDense)(input::AbstractVector{<:Real})
		W, b, σ = L.W, L.b, L.σ
		return σ(W * input + b)
	end
	
	function (L::LayerDense)(input::AbstractMatrix{<:Real})
		W, b, σ = L.W, L.b, L.σ
		return σ(W * input .+ b)
	end
end

# ╔═╡ 92a9ff09-50f1-42d1-8b12-c37771610d84
begin
	function σ(x::Real) 
		t = exp(-abs(x))
		return ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
	end	
	
	σ(x::AbstractVecOrMat{T}) where {T<:Real} = σ.(x)
	
	const sigmoid = σ
end

# ╔═╡ 468eb0e4-5c6f-49c6-9589-533071bb1271
begin
	relu(x::Real) = max(zero(x), x) # zero function keep the type of x
	relu(x::AbstractVecOrMat{<:Real}) = relu.(x)
	
	#log1p(x) is the seme as log(1 + x)
	softplus(x::Real) = ifelse(x > 0, x + log(1 + exp(-x)), log(1 + exp(x)))
	softplus(x::AbstractVecOrMat{<:Real}) = softplus.(x)
	
	function softmax(x::Vector{T}) where {T<:Real}
		m = maximum(x)
		exp_val = exp.(x .- m)
		s = sum(exp_val)
		return exp_val ./ s
	end
	function softmax(x::Matrix{T}) where {T<:Real}
		m = maximum(x, dims = 1)
		exp_val = exp.(x .- m)
		s = sum(exp_val, dims = 1)
		return exp_val ./ s
	end
end

# ╔═╡ 65358f5c-04a5-4618-a78e-1fd02d72060d
begin
	function xlogy(x::Real, y::Real) 
		result = x * log(y)
		ifelse(iszero(x) && !isnan(y), zero(result), result)
	end
	
	function crossentropy(ŷ::AbstractVecOrMat{<:Real},
						  y::AbstractVecOrMat{<:Real};
						  dims::Int = 1,
						  agg::Function = Statistics.mean)
    	agg(.-sum(xlogy.(y, ŷ), dims = dims))
	end
end

# ╔═╡ c66c7488-8667-4694-915e-b022488ccd8b
begin
	struct Chain{T<:Tuple}
		layers::T
		Chain(layers...) = new{typeof(layers)}(layers)		
	end
	
	function (L::Chain)(input::AbstractVecOrMat{<:Real})
		return ∘(reverse(L.layers)...)(input)
	end
end

# ╔═╡ bd4c6ca8-93a2-4344-b598-fa533ad7e169
function params(C::Chain)
	p = []
	for layer in C.layers
		#push!(p, layer.W[:], layer.b )
		push!(p, layer.W, layer.b )
	end
	p #return vcat(p...)
end

# ╔═╡ b10cf718-528a-4433-9ae4-25ca78256ec8
md"# Exemplos"

# ╔═╡ 5fefc40a-dbc9-42e5-b9ea-bb2eed97a805
begin
	Random.seed!(1998)
	
	L1 = LayerDense(10, 5, σ)
	L2 = LayerDense(5, 2, softmax)
	
	r = rand(10)
		
	m = Chain(L1, L2)				# C1 = L2(L1(r))
	
	label = [1.0, 0.0]
	
	#function loss(p::Vector) 
		# atualiza o modelo com os parametros p
		#update(m, p)
		# aplica na função de erro
		#crossentropy(m(r), label)
	#end
	
	#∇G = x -> ForwardDiff.gradient(loss, x); # g = ∇f
	
	p = params(m)
	# for iter in iteracoes
		# calcula o loss (crossentropy)
		#loss(r, label)
		# acha o gradiente negativo dos params (p) com o ForwardDiff
		#grad = ∇G(m(r))
		# atualiza o parametro (p) substraindo o gradiente negativo
		#p .-= grad
	# end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
ForwardDiff = "~0.10.34"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.3"
manifest_format = "2.0"
project_hash = "9dc5c1876c245eb2e0dc94271cfdc7608c9718ee"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "00a2cccc7f098ff3b66806862d275ca3db9e6e5a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.5.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "c5b6685d53f933c11404a3ae9822afe30d522494"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.12.2"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "a69dd6db8a809f78846ff259298678f0d6212180"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.34"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "946607f84feb96220f480e0422d3484c49c00239"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.19"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "ffc098086f35909741f71ce21d03dadf0d2bfa76"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.11"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─61709a7e-18c5-4017-8e84-6eaf16ba210d
# ╠═a5ec2a7e-2dab-46c9-9013-9e7adbd44d78
# ╠═bd149090-b5e2-11eb-34e9-ef542421d474
# ╠═92a9ff09-50f1-42d1-8b12-c37771610d84
# ╠═468eb0e4-5c6f-49c6-9589-533071bb1271
# ╠═65358f5c-04a5-4618-a78e-1fd02d72060d
# ╠═bd4c6ca8-93a2-4344-b598-fa533ad7e169
# ╠═c66c7488-8667-4694-915e-b022488ccd8b
# ╟─b10cf718-528a-4433-9ae4-25ca78256ec8
# ╠═5fefc40a-dbc9-42e5-b9ea-bb2eed97a805
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
