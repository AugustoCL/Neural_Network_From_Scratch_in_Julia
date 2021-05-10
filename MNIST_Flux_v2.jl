using Flux
using Flux: train!, throttle, @epochs, onecold
using Flux.Losses: crossentropy
using Flux.Data: DataLoader
using MLDatasets: MNIST
using Statistics

tensor_train, labels_train = MNIST.traindata(Float64)
tensor_test, labels_test = MNIST.testdata(Float64)

xtrain = reshape(tensor_train, 784, :)
xtest = reshape(tensor_test, 784, :)

#ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)
ytrain = hcat([ [i ≠ j ? 0.0 : 1.0 for j in 0:9] for i in labels_train ]...)
ytest = hcat([ [i ≠ j ? 0.0 : 1.0 for j in 0:9] for i in labels_test ]...)

train = DataLoader((xtrain, ytrain), batchsize = 3000, shuffle = true)
test = (xtest, ytest)

model = Chain(
    Dense(784, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 10), softmax
)

loss(x, y) = crossentropy(model(x), y)

ps = params(model)

opt = Descent(0.1)

train!(loss, ps, train, opt)

evalcb() = @show(loss(xtrain, ytrain))
throtle_cb = throttle(evalcb, 3)
@epochs 5 train!(loss, ps, train, opt, cb = throtle_cb)

accuracy(x, y) = mean(onecold(model(x)) .== onecold(y)) # cute way to find average of correct guesses
accuracy(xtrain, ytrain)

function prediction(x)
    X = model(x) 
    predict = findfirst(x -> x == maximum(X), X) - 1
    return predict
end

prediction(xtest[:, 2])

labels_test[2]